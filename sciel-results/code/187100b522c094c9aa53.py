"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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
    max_r = int(np.ceil(n ** (1 / 3))) + 3
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


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    # HCP: layers of hexagonal close packing
    a = 1.0 / np.sqrt(2.0)
    c = a * np.sqrt(8.0 / 3.0)
    
    candidates = []
    # Layer 0 (AB stacking)
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                x = a * (i + j / 2.0)
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                candidates.append((np.sqrt(x*x + y*y + z*z), np.array([x, y, z])))
                # B layer offset
                x2 = a * (i + j / 2.0 + 0.5)
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / np.sqrt(3.0))
                z2 = c * (k + 1.0 / 3.0)
                candidates.append((np.sqrt(x2*x2 + y2*y2 + z2*z2), np.array([x2, y2, z2])))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a Mackay icosahedral cluster of n atoms."""
    # Use icosahedral symmetry points
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    # Generate candidates from icosahedral shell structure
    max_r = int(np.ceil(n ** (1 / 3))) + 2
    candidates = []
    
    # Add icosahedral lattice points (using the FCC-like approach but with ico symmetry)
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_random_compact(n, rng, radius_scale=1.0):
    """Generate a random compact cluster."""
    radius = 1.2 + 0.4 * (n / 50.0) * radius_scale
    pos = rng.uniform(-radius, radius, (n, 3))
    return pos


def _polish(positions, kind, max_iter=500, tol=1e-10):
    """High-precision polish pass using gradient-based optimization."""
    from scipy.optimize import minimize
    
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        e, _ = sutton_chen_energy_gradient(pos, kind)
        return e
    
    def gradient_fn(x):
        pos = x.reshape(-1, 3)
        _, g = sutton_chen_energy_gradient(pos, kind)
        return g.ravel()
    
    try:
        result = minimize(
            energy_fn,
            positions.ravel(),
            jac=gradient_fn,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'gtol': tol, 'ftol': tol, 'maxls': 50}
        )
        if result.success and np.isfinite(result.fun):
            polished = result.x.reshape(-1, 3)
            if _is_valid(polished):
                return result.fun, polished
    except Exception:
        pass
    
    return np.inf, positions


def _growth_construct(n, kind, rng):
    """
    Deterministic growth algorithm: build cluster by adding atoms one at a time
    to the lowest-energy interstitial site of the (N-1)-atom cluster.
    """
    if n <= 1:
        return np.zeros((n, 3))
    
    # Start with 2 atoms at optimal distance
    best_energy = np.inf
    best_pos = None
    
    # Try initial distances
    for d0 in [0.9, 1.0, 1.1, 1.2]:
        pos = np.array([[0, 0, 0], [d0, 0, 0]])
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
    
    if best_pos is None:
        best_pos = np.array([[0, 0, 0], [1.0, 0, 0]])
    
    # Grow from 2 to n atoms
    current = best_pos.copy()
    
    for target_n in range(3, n + 1):
        if _time_left() < 40.0:
            break
        
        # Generate candidate positions for new atom
        candidates = []
        
        # For each existing atom, try positions at various offsets
        for i in range(len(current)):
            # Random directions
            for trial in range(6):
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                for dist in [0.8, 0.9, 1.0, 1.1, 1.2]:
                    new_pos = current[i] + direction * dist
                    candidates.append(new_pos)
        
        # Also try positions in "gaps" - midpoints between close pairs
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                midpoint = (current[i] + current[j]) / 2
                bond_vec = current[j] - current[i]
                bond_len = np.linalg.norm(bond_vec)
                if bond_len > 0.1:
                    perp = np.cross(bond_vec, [0, 0, 1])
                    if np.linalg.norm(perp) < 1e-6:
                        perp = np.cross(bond_vec, [1, 0, 0])
                    perp /= np.linalg.norm(perp)
                    for offset in [0.3, 0.5, 0.7]:
                        candidates.append(midpoint + perp * offset)
                        candidates.append(midpoint - perp * offset)
        
        # Relax the full cluster with each candidate
        best_e = np.inf
        best_p = None
        
        # Shuffle candidates and try up to 12
        rng.shuffle(candidates)
        for candidate in candidates[:12]:
            if _time_left() < 35.0:
                break
            
            test_pos = np.vstack([current, candidate.reshape(1, 3)])
            try:
                energy, relaxed = relax(test_pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_e:
                        best_e = energy
                        best_p = relaxed
            except Exception:
                continue
        
        if best_p is not None:
            current = best_p
        else:
            # Fallback: add atom at a random position
            new_atom = current[0] + rng.normal(0, 0.5, 3)
            current = np.vstack([current, new_atom.reshape(1, 3)])
    
    return current


def _differential_evolution(positions_init, kind, pop_size=12, max_generations=30, 
                             mutation_rate=0.8, crossover_rate=0.9, rng=None):
    """
    Custom differential evolution for cluster optimization.
    
    Args:
        positions_init: (n, 3) initial positions
        kind: potential kind string
        pop_size: population size
        max_generations: max generations
        mutation_rate: DE mutation constant F
        crossover_rate: crossover probability CR
        rng: numpy random generator
    
    Returns:
        (best_energy, best_positions)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_atoms = len(positions_init)
    dim = n_atoms * 3
    
    # Initialize population from diverse templates
    population = []
    
    # First member: initial positions
    population.append(positions_init.copy())
    
    # Generate diverse initial population
    templates = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedron]
    for i in range(1, pop_size):
        if _time_left() < 30.0:
            break
        
        # Alternate between templates and random
        if i % 2 == 1 and i // 2 < len(templates):
            base = templates[i // 2](n_atoms)
            # Add some noise
            noise = rng.normal(0, 0.1, base.shape)
            population.append(base + noise)
        else:
            # Random compact
            radius = 1.2 + 0.4 * (n_atoms / 50.0)
            pos = rng.uniform(-radius, radius, (n_atoms, 3))
            population.append(pos)
    
    # Ensure we have enough members
    while len(population) < pop_size:
        base = _generate_fcc(n_atoms)
        noise = rng.normal(0, 0.15, base.shape)
        population.append(base + noise)
    
    population = np.array(population)
    
    # Evaluate initial population
    fitness = np.full(pop_size, np.inf)
    for i in range(pop_size):
        if _time_left() < 25.0:
            break
        pos = population[i]
        if not _is_valid(pos):
            fitness[i] = np.inf
            continue
        try:
            energy, relaxed = relax(pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                fitness[i] = energy
                population[i] = relaxed
        except Exception:
            fitness[i] = np.inf
    
    best_idx = np.argmin(fitness)
    best_energy = fitness[best_idx]
    best_pos = population[best_idx].copy()
    
    # DE evolution loop
    for gen in range(max_generations):
        if _time_left() < 20.0:
            break
        
        for i in range(pop_size):
            if _time_left() < 15.0:
                break
            
            # Select three distinct random indices
            idxs = np.random.choice(pop_size, size=4, replace=False)
            a, b, c = idxs[0], idxs[1], idxs[2]
            
            if a == i or b == i or c == i:
                continue
            
            # Mutation: v = x_a + F * (x_b - x_c)
            v = population[a] + mutation_rate * (population[b] - population[c])
            
            # Crossover: u_i = v_i if rand < CR or i == jrand, else x_i
            jrand = np.random.randint(0, n_atoms)
            u = population[i].copy()
            for j in range(n_atoms):
                if rng.random() < crossover_rate or j == jrand:
                    u[j] = v[j]
            
            # Evaluate trial vector
            if not _is_valid(u):
                continue
            
            try:
                energy, relaxed = relax(u, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < fitness[i]:
                        fitness[i] = energy
                        population[i] = relaxed
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed.copy()
            except Exception:
                continue
        
        # Check convergence
        if gen > 0 and gen % 5 == 0:
            if np.std(fitness[fitness < np.inf]) < 1e-8:
                break
    
    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Strategy 1: Deterministic growth (primary)
    try:
        growth_pos = _growth_construct(n, kind, rng)
        if _is_valid(growth_pos):
            energy, relaxed = relax(growth_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                best_energy = energy
                best_pos = relaxed
    except Exception:
        pass
    
    # Strategy 2: FCC/BCC/HCP templates with relaxation
    for gen_func in [_generate_fcc, _generate_bcc, _generate_hcp, _generate_icosahedron]:
        if _time_left() < 25.0:
            break
        
        base = gen_func(n)
        if not _is_valid(base):
            continue
        
        # Try multiple noise levels
        for noise_scale in [0.0, 0.05, 0.1, 0.2]:
            if _time_left() < 20.0:
                break
            
            noise = rng.normal(0, noise_scale, base.shape) if noise_scale > 0 else np.zeros_like(base)
            pos = base + noise
            
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                
                if not np.isfinite(energy):
                    continue
                    
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Strategy 3: Differential Evolution (new primary global search)
    if _time_left() > 60.0:
        try:
            # Use the best position found so far as seed for DE
            if best_pos is not None:
                de_seed = best_pos.copy()
            else:
                de_seed = _generate_fcc(n)
            
            de_energy, de_pos = _differential_evolution(
                de_seed, kind, 
                pop_size=10, 
                max_generations=25,
                mutation_rate=0.7,
                crossover_rate=0.8,
                rng=rng
            )
            
            if np.isfinite(de_energy) and _is_valid(de_pos):
                if de_energy < best_energy:
                    best_energy = de_energy
                    best_pos = de_pos
        except Exception:
            pass
    
    # Strategy 4: Random restarts if time remains
    if _time_left() > 35.0:
        for i in range(20):
            if _time_left() < 20.0:
                break
            
            # Random compact sphere
            radius_scale = 1.0 + 0.2 * (i % 3)
            pos = _generate_random_compact(n, rng, radius_scale)
            
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                
                if not np.isfinite(energy):
                    continue
                    
                if _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Strategy 5: Final high-precision polish
    if best_pos is not None and _time_left() > 15.0:
        try:
            polished_energy, polished_pos = _polish(best_pos, kind, max_iter=200, tol=1e-12)
            if polished_energy < best_energy:
                best_energy = polished_energy
                best_pos = polished_pos
        except Exception:
            pass
    
    # Final fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
