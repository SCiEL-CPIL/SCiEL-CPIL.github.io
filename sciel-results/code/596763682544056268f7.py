"""Differential Evolution search for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    """Check that all coordinates are finite and pairwise distances >= min_dist."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Use squared distances for speed, min_dist^2 = 0.1225
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    return np.all(dists_sq >= 0.1225)

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:min(n-1, 12)])
    
    if n > 13:
        shell2_radius = 1.8
        rng = np.random.default_rng(42)
        for _ in range(min(n - len(pts), 42)):
            theta = rng.uniform(0, 2 * np.pi)
            cos_phi = rng.uniform(-1, 1)
            sin_phi = np.sqrt(1 - cos_phi**2)
            p = shell2_radius * np.array([sin_phi * np.cos(theta), sin_phi * np.sin(theta), cos_phi])
            pts.append(p)
    
    return np.array(pts[:n])

def _generate_random(n, seed):
    """Generate a random cluster."""
    rng = np.random.default_rng(seed)
    # Box size based on n
    box_size = 2 * (n ** (1/3)) + 2
    pts = rng.uniform(-box_size/2, box_size/2, (n, 3))
    return pts

def _evaluate_worker(args):
    """Worker function for parallel evaluation."""
    pos, kind, n = args
    # Ensure valid
    if not _is_valid(pos):
        return np.inf, pos
    
    # Relax
    try:
        energy, relaxed = relax(pos, 'sutton_chen', kind)
        if not np.isfinite(energy) or not _is_valid(relaxed):
            return np.inf, pos
        return energy, relaxed
    except Exception:
        return np.inf, pos

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # Initialize population
    # Use 16 diverse starting points
    pop_size = 16
    population = []
    
    # 1. FCC
    fcc_pos = _generate_fcc(n)
    population.append(fcc_pos)
    
    # 2. BCC
    bcc_pos = _generate_bcc(n)
    population.append(bcc_pos)
    
    # 3. Icosahedron
    ico_pos = _generate_icosahedron(n)
    population.append(ico_pos)
    
    # 4-16. Random
    for i in range(pop_size - 3):
        rand_pos = _generate_random(n, seed=100 + i)
        population.append(rand_pos)
    
    # Evaluate initial population in parallel
    tasks = [(pos, kind, n) for pos in population]
    
    # Use up to 4 workers
    num_workers = min(4, mp.cpu_count(), pop_size)
    
    # If time is tight, do it sequentially or with fewer workers
    if _time_left() < 10.0:
        # Fallback to sequential
        results = []
        for task in tasks:
            res = _evaluate_worker(task)
            results.append(res)
            if _time_left() < 5.0:
                break
    else:
        with mp.Pool(num_workers) as pool:
            try:
                results = pool.map(_evaluate_worker, tasks)
            except Exception:
                # Fallback if pool fails
                results = [_evaluate_worker(task) for task in tasks]
    
    # Update best
    for energy, pos in results:
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
            
    # Differential Evolution Loop
    # Parameters
    F = 0.8  # Scaling factor
    CR = 0.9  # Crossover rate
    max_iter = 50 # Max DE iterations
    
    # Track population energies
    pop_energies = [res[0] for res in results]
    pop_positions = [res[1] for res in results]
    
    # Ensure at least one valid
    valid_mask = [e < np.inf for e in pop_energies]
    if not any(valid_mask):
        # All invalid, try harder
        # Just return the best invalid? No, must return valid.
        # Fallback to a simple valid structure
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
             best_pos = _generate_bcc(n)
        return best_pos

    for iter in range(max_iter):
        if _time_left() < 15.0:
            break
            
        # Generate new population
        new_population = []
        new_energies = []
        
        # Sort current population
        sorted_indices = np.argsort(pop_energies)
        sorted_positions = [pop_positions[i] for i in sorted_indices]
        sorted_energies = [pop_energies[i] for i in sorted_indices]
        
        for i in range(pop_size):
            # DE/rand/1/bin mutation
            # Pick 3 distinct individuals different from i
            # Simple strategy: pick random 3
            a = np.random.randint(0, pop_size)
            b = np.random.randint(0, pop_size)
            c = np.random.randint(0, pop_size)
            # Ensure they are distinct
            while a == i or b == i or c == i or a == b or a == c or b == c:
                a = np.random.randint(0, pop_size)
                b = np.random.randint(0, pop_size)
                c = np.random.randint(0, pop_size)
                
            # Use the best available if possible? 
            # Standard DE uses random. Let's use random but prefer better ones slightly?
            # For now, standard random.
            
            x_a = pop_positions[a]
            x_b = pop_positions[b]
            x_c = pop_positions[c]
            x_i = pop_positions[i]
            
            # Mutation
            v = x_i + F * (x_a - x_b) + F * (x_c - x_i) # Standard DE/rand/1
            
            # Crossover
            # Random vector for crossover
            j_rand = np.random.randint(0, 3)
            cross_mask = np.random.rand(n, 3) < CR
            cross_mask[j_rand, :] = True
            
            u = np.where(cross_mask, v, x_i)
            
            new_population.append(u)
            
        # Evaluate new population in parallel
        tasks = [(pos, kind, n) for pos in new_population]
        
        if _time_left() < 10.0:
            results = []
            for task in tasks:
                res = _evaluate_worker(task)
                results.append(res)
                if _time_left() < 5.0:
                    break
        else:
            with mp.Pool(num_workers) as pool:
                try:
                    results = pool.map(_evaluate_worker, tasks)
                except Exception:
                    results = [_evaluate_worker(task) for task in tasks]
        
        # Selection: keep better of trial and target
        for i in range(len(results)):
            trial_energy, trial_pos = results[i]
            target_energy = pop_energies[i]
            
            if trial_energy < target_energy:
                pop_energies[i] = trial_energy
                pop_positions[i] = trial_pos
                if trial_energy < best_energy:
                    best_energy = trial_energy
                    best_pos = trial_pos
        
        # Check for convergence or stagnation?
        # If best energy doesn't change for 5 iterations, break?
        # DE usually runs for fixed iterations.
        
    # Final Polish
    # Try to relax the best found one more time with tight tolerances?
    # relax already does L-BFGS-B.
    
    if best_pos is None or not _is_valid(best_pos):
        # Fallback
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            # Last resort
            rng = np.random.default_rng(42)
            best_pos = rng.uniform(-2, 2, (n, 3))
            for _ in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng.uniform(-2, 2, (n, 3))
                
    return best_pos

# EVOLVE-BLOCK-END
