"""
Dzugutov cluster global minima search using GA with polytetrahedral templates.
Key insight: GA with icosahedral templates + blend crossover explores rugged landscape better.
Templates seed near-global basins, crossover/mutation escape local minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize

_START_TIME = time.time()
_HARD_LIMIT = 315.0

def _remaining():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy_gradient(positions):
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    energy = 0.0
    gradient = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            r_vec = diff / r
            force_mag = 0.0
            
            if r < a:
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
                energy += B * np.exp(d / (r - b))
            
            gradient[i] += force_mag * r_vec
            gradient[j] -= force_mag * r_vec

    return energy, gradient

def _fibonacci_sphere(n, rng):
    """Generate n points on a sphere using Fibonacci spiral."""
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    if n == 2:
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * i / (n - 1)
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    return np.array(points)

def _generate_icosahedron(n_atoms, rng):
    """Generate icosahedral structure with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    
    # First shell: 12 atoms at icosahedral vertices
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
    
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
    
    # Additional shells using Fibonacci sphere
    remaining = n_atoms - 13
    shell_radii = [1.9 * 1.7, 1.9 * 2.4, 1.9 * 3.1, 1.9 * 3.8]
    
    for radius in shell_radii:
        if remaining <= 0:
            break
        # Add up to 30 atoms per shell (or remaining)
        n_this_shell = min(remaining, 30)
        shell_pts = _fibonacci_sphere(n_this_shell, rng) * radius
        positions.extend(shell_pts)
        remaining -= n_this_shell
    
    # If still need more, add random points
    while remaining > 0:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v) * 1.9 * (4 + remaining * 0.1)
        positions.append(v)
        remaining -= 1
    
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.2):
    """Generate random compact structure."""
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.9
    points = []
    max_attempts = 5000
    attempts = 0
    while len(points) < n_atoms and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _local_minimize(positions, max_iter=150):
    """Local minimization using L-BFGS-B."""
    try:
        res = minimize(
            lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[0],
            positions.flatten(),
            jac=lambda x: _dzugutov_energy_gradient(x.reshape(-1, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxls': 80, 'ftol': 1e-9, 'gtol': 1e-7}
        )
        if res.fun < np.inf:
            return res.fun, res.x.reshape(-1, 3)
    except Exception:
        pass
    return _dzugutov_energy_gradient(positions)[0], positions

def _blend_crossover(parent1, parent2, rng):
    """Blend crossover for cluster structures."""
    # Align by sorting by distance from centroid
    c1 = np.mean(parent1, axis=0)
    c2 = np.mean(parent2, axis=0)
    p1 = parent1 - c1
    p2 = parent2 - c2
    
    # Sort by radius
    r1 = np.linalg.norm(p1, axis=1)
    r2 = np.linalg.norm(p2, axis=1)
    idx1 = np.argsort(r1)
    idx2 = np.argsort(r2)
    p1 = p1[idx1]
    p2 = p2[idx2]
    
    # Blend
    alpha = rng.uniform(0.3, 0.7)
    child = alpha * p1 + (1 - alpha) * p2
    
    # Add small mutation
    child += rng.normal(0, 0.1, child.shape)
    
    # Center back
    child += c1
    return child

def _ga_search(n_atoms, rng, time_budget):
    """Genetic algorithm search."""
    pop_size = 8
    population = []
    
    # Initialize population with diverse structures
    population.append(_generate_icosahedron(n_atoms, rng))
    population.append(_random_compact(n_atoms, rng))
    population.append(_random_compact(n_atoms, rng))
    
    # Add perturbed icosahedron
    icosa = _generate_icosahedron(n_atoms, rng)
    population.append(icosa + rng.normal(0, 0.3, icosa.shape))
    
    # Fill rest with random
    while len(population) < pop_size:
        population.append(_random_compact(n_atoms, rng))
    
    # Evaluate initial population
    energies = []
    for pos in population:
        e, _ = _local_minimize(pos, max_iter=50)
        energies.append(e)
    
    best_energy = min(energies)
    best_idx = np.argmin(energies)
    best_positions = population[best_idx].copy()
    
    start_time = time.time()
    gen = 0
    
    while time.time() - start_time < time_budget:
        if _remaining() < 10:
            break
        
        gen += 1
        new_population = []
        
        # Elitism: keep best
        new_population.append(population[best_idx].copy())
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            # Tournament selection
            if rng.random() < 0.5:
                p1 = population[best_idx]
            else:
                p1 = population[rng.integers(pop_size)]
            
            if rng.random() < 0.5:
                p2 = population[best_idx]
            else:
                p2 = population[rng.integers(pop_size)]
            
            if p1 is p2:
                child = p1 + rng.normal(0, 0.4, p1.shape)
            else:
                child = _blend_crossover(p1, p2, rng)
            
            # Relax
            try:
                e, _ = _local_minimize(child, max_iter=40)
                new_population.append(child)
            except Exception:
                new_population.append(child)
        
        # Evaluate new population
        new_energies = []
        for pos in new_population:
            e, _ = _local_minimize(pos, max_iter=30)
            new_energies.append(e)
        
        # Update best
        if min(new_energies) < best_energy:
            best_idx = np.argmin(new_energies)
            best_energy = new_energies[best_idx]
            best_positions = new_population[best_idx].copy()
        
        population = new_population
        energies = new_energies
    
    return best_energy, best_positions

def _basin_hopping(positions, rng, time_limit):
    """Simple basin hopping."""
    current_energy, current_positions = _local_minimize(positions, max_iter=60)
    best_energy, best_positions = current_energy, current_positions.copy()
    
    start_time = time.time()
    step_size = 0.4
    
    while time.time() - start_time < time_limit and _remaining() > 5:
        perturbed = current_positions + rng.normal(0, step_size, current_positions.shape)
        try:
            new_energy, new_positions = _local_minimize(perturbed, max_iter=40)
        except Exception:
            continue
        
        if new_energy < current_energy or (rng.random() < np.exp(-(new_energy - current_energy) / 0.4)):
            current_energy = new_energy
            current_positions = new_positions
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        if rng.random() < 0.08:
            step_size *= 1.15
        elif rng.random() < 0.08:
            step_size = max(0.15, step_size * 0.85)
    
    return best_energy, best_positions

def construct(instance):
    size = instance["n"]
    start_time = time.time()
    time_budget = min(48.0, _remaining() - 5)
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    best_energy = np.inf
    best_positions = None
    lock = threading.Lock()
    
    def _update_best(e, pos):
        nonlocal best_energy, best_positions
        with lock:
            if e < best_energy:
                best_energy = e
                best_positions = pos.copy()
    
    # Phase 1: GA search (main workhorse)
    try:
        ga_energy, ga_positions = _ga_search(size, rng, time_budget * 0.45)
        _update_best(ga_energy, ga_positions)
    except Exception:
        pass
    
    # Phase 2: Parallel basin hopping from diverse starts
    remaining_time = min(time_budget * 0.55, _remaining() - 5)
    if remaining_time > 3:
        starts = []
        
        # Icosahedral seed
        try:
            starts.append(_generate_icosahedron(size, rng))
        except Exception:
            pass
        
        # Random compacts
        starts.append(_random_compact(size, rng))
        starts.append(_random_compact(size, rng))
        
        # Perturbed best
        if best_positions is not None:
            starts.append(best_positions + rng.normal(0, 0.25, best_positions.shape))
        
        per_chain_time = remaining_time / max(1, len(starts))
        
        def bh_worker(start_pos):
            try:
                e, p = _basin_hopping(start_pos, rng, per_chain_time)
                _update_best(e, p)
            except Exception:
                pass
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(bh_worker, s) for s in starts]
            for f in as_completed(futures):
                if _remaining() < 8:
                    break
                try:
                    f.result()
                except Exception:
                    pass
    
    # Final polish: a few more basin hopping iterations from best
    if best_positions is not None and _remaining() > 12:
        try:
            e, p = _basin_hopping(best_positions, rng, min(8, _remaining() - 10))
            _update_best(e, p)
        except Exception:
            pass
    
    if best_positions is None:
        # Fallback
        return _random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
