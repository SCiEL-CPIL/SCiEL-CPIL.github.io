"""
Strategy: Parallel Multi-Start Global Optimization with Physics-Informed Initialization.

Key Differences from Parent:
1. Parallelization: Uses concurrent.futures.ThreadPoolExecutor to run 4 independent
   optimization chains simultaneously. This utilizes the 4-thread allowance to explore
   4 distinct basins in parallel, effectively quadrupling the search coverage per unit time
   compared to the serial basin hopping in the parent.
2. Initialization: Instead of relying heavily on lattice fragments (FCC/HCP) which may be
   too rigid for certain cluster sizes, we use a "Packed Sphere" initialization that places
   atoms with a minimum distance constraint (soft-sphere packing). This creates diverse,
   liquid-like structures that are closer to the local minima of the Sutton-Chen potential
   than purely random points, reducing the number of L-BFGS-B steps required for convergence.
3. Optimization Loop: Each thread runs a sequence of "restarts". For each restart, it takes
   the best position found so far in that thread, adds Gaussian noise, and runs L-BFGS-B.
   This is a simplified, parallel version of Basin Hopping, but without the serial bottleneck
   of waiting for each hop to finish before starting the next.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 295.0  # Safety margin before 300s hard kill
_START_TIME = None
_LOCK = None
import threading
_LOCK = threading.Lock()

def _init_global():
    global _START_TIME, _LOCK
    _START_TIME = time.time()
    _LOCK = threading.Lock()

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _packed_sphere_init(n, seed, min_dist=0.6):
    """
    Generate a compact cluster of n atoms with a minimum distance constraint.
    This mimics a liquid/solid interface and provides a better starting point
    than uniform random.
    """
    rng = np.random.default_rng(seed)
    points = []
    # Estimate radius based on packing density
    # Volume ~ n * (min_dist/2)^3 / packing_efficiency (0.74 for FCC)
    # R ~ (n * (min_dist/2)^3 / 0.74)^(1/3)
    R = 1.2 * n ** (1.0 / 3.0) * min_dist
    
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        # Sample in sphere
        u = rng.random(3)
        r = R * rng.random() ** (1.0 / 3.0)
        theta = rng.random() * 2 * np.pi
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        # Check distance to existing points
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < min_dist:
                attempts += 1
                continue
        
        points.append(p)
        attempts += 1
    
    # If we failed to place all atoms, fill with random points (relaxer will fix)
    if len(points) < n:
        remaining = n - len(points)
        extra = rng.uniform(-R, R, (remaining, 3))
        points.extend(extra.tolist())
        
    return np.array(points)

def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    if n < 2:
        return float('inf')
    # Vectorized calculation for speed, though O(N^2)
    # For N~80, this is fast enough.
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)

def _worker(thread_id, n, kind, potential, initial_positions, results_container):
    """
    Worker function for parallel optimization.
    Runs a series of restarts: perturb current best, relax, update.
    """
    rng = np.random.default_rng(42 + thread_id)
    current_pos = initial_positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    if not np.isfinite(current_energy):
        # Try to fix initial position
        try:
            current_energy, current_pos = relax(current_pos, potential, kind)
        except:
            return

    step_size = 0.2
    max_restarts = 50  # Cap restarts per thread to ensure diversity
    
    for restart in range(max_restarts):
        if _time_left() < 2:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, (n, 3))
        perturbed = current_pos + noise
        
        try:
            energy, positions = relax(perturbed, potential, kind)
        except Exception:
            continue
            
        if not np.isfinite(energy):
            step_size *= 1.1
            continue
            
        min_d = _min_distance(positions)
        if min_d < 0.35:
            # Invalid configuration, increase step size to escape
            step_size *= 1.2
            step_size = min(step_size, 0.8)
            continue
            
        if energy < current_energy:
            current_energy = energy
            current_pos = positions
            step_size *= 0.9 # Slightly reduce step size on success
        else:
            step_size *= 1.05 # Slightly increase on failure
            step_size = min(step_size, 0.8)
            
        # Update global best
        with _LOCK:
            if energy < results_container['best_energy']:
                results_container['best_energy'] = energy
                results_container['best_positions'] = positions.copy()

def construct(instance):
    global _START_TIME
    _init_global()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    results_container = {
        'best_energy': float('inf'),
        'best_positions': None
    }
    
    # Generate 4 diverse initial configurations
    # We use different seeds and slight variations in min_dist to ensure diversity
    initial_configs = []
    for i in range(4):
        # Vary min_dist slightly to get different packing densities
        min_dist = 0.55 + 0.05 * i
        config = _packed_sphere_init(n, seed=100 + i, min_dist=min_dist)
        initial_configs.append(config)
        
        # Also add a pure random one for the last thread to maximize diversity
        if i == 3:
            rng = np.random.default_rng(999)
            R = 1.2 * n ** (1.0 / 3.0)
            config = rng.uniform(-R, R, (n, 3))
            initial_configs[3] = config

    # Run 4 parallel workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            future = executor.submit(_worker, i, n, kind, potential, initial_configs[i], results_container)
            futures.append(future)
            
        # Wait for all to finish or timeout
        # We rely on the _time_left check inside workers to stop them.
        # We need to wait for them to actually return.
        try:
            for f in concurrent.futures.as_completed(futures, timeout=_time_left()):
                f.result()
        except concurrent.futures.TimeoutError:
            pass
        except Exception:
            pass

    best_positions = results_container['best_positions']
    best_energy = results_container['best_energy']
    
    # Fallback if parallel execution failed or found nothing
    if best_positions is None or not np.isfinite(best_energy):
        # Try a single serial relaxation on the first initial config
        try:
            best_energy, best_positions = relax(initial_configs[0], potential, kind)
            if not np.isfinite(best_energy):
                # Last resort: return a valid random config
                best_positions = initial_configs[0]
                # Ensure validity
                md = _min_distance(best_positions)
                if md < 0.35:
                    best_positions = best_positions * 1.5
        except Exception:
            best_positions = initial_configs[0]
            md = _min_distance(best_positions)
            if md < 0.35:
                best_positions = best_positions * 1.5

    # Final safety check
    if best_positions is not None:
        md = _min_distance(best_positions)
        if md < 0.35:
            # Try to relax again with tighter constraints if possible, or just scale
            # Scaling is safer for time limits
            best_positions = best_positions * 1.2
            md = _min_distance(best_positions)
            if md < 0.35:
                best_positions = best_positions * 1.5

    return best_positions
# EVOLVE-BLOCK-END
