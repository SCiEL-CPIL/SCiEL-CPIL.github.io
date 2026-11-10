import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def _initial_positions(n, rng, min_dist=0.45):
    """Generate initial positions using a structured approach with rejection."""
    radius = 1.2 * n ** (1.0 / 3.0)
    points = [np.zeros(3)]
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        vec = rng.normal(size=3)
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            continue
        # Sample radius with cube-root distribution for uniform volume sampling
        r = radius * (0.3 + 0.7 * rng.random() ** (1.0 / 3.0))
        p = vec / norm * r
        if len(points) == 1:
            points.append(p)
        else:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) > min_dist:
                points.append(p)
    if len(points) < n:
        # Fallback: random positions
        pts = rng.normal(size=(n, 3)) * 0.5
        return pts
    return np.array(points)

def _run_bh_restart(n, kind, rng_seed, max_bh_steps=200, base_perturb=0.25):
    """
    Runs a single Basin Hopping restart with adaptive perturbation.
    """
    rng = np.random.default_rng(rng_seed)
    
    # 1. Initialization
    points = _initial_positions(n, rng, min_dist=0.4)
    
    # 2. Initial Relaxation
    try:
        current_energy, current_points = relax(points, 'sutton_chen', kind)
    except Exception:
        return np.inf, None
        
    if not np.isfinite(current_energy):
        return np.inf, None
        
    # Center the cluster
    current_points = current_points - np.mean(current_points, axis=0)
    
    best_energy = current_energy
    best_points = current_points.copy()
    
    # 3. Basin Hopping Loop with adaptive parameters
    # Scale perturbation with cluster size
    scale_factor = 1.0 + 0.3 * np.log10(n / 10.0)
    T = 0.4 * scale_factor
    cooling_rate = 0.97
    min_T = 0.03
    
    for step in range(max_bh_steps):
        # Adaptive perturbation: larger early, smaller late
        progress = step / max_bh_steps
        scale = base_perturb * scale_factor * (1.0 - 0.5 * progress)
        perturbation = rng.normal(scale=scale, size=(n, 3))
        perturbed_points = current_points + perturbation
        
        # Relax
        try:
            new_energy, new_points = relax(perturbed_points, 'sutton_chen', kind)
        except Exception:
            continue
            
        if not np.isfinite(new_energy):
            continue
            
        # Center
        new_points = new_points - np.mean(new_points, axis=0)
        
        # Accept/Reject with Metropolis criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_energy = new_energy
            current_points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
                
        T *= cooling_rate
        if T < min_T:
            break
            
    return best_energy, best_points

def _run_bh_restart_large_perturb(n, kind, rng_seed, max_bh_steps=150, base_perturb=0.5):
    """
    Basin Hopping with larger perturbations for escaping deep local minima.
    """
    rng = np.random.default_rng(rng_seed)
    
    points = _initial_positions(n, rng, min_dist=0.35)
    
    try:
        current_energy, current_points = relax(points, 'sutton_chen', kind)
    except Exception:
        return np.inf, None
        
    if not np.isfinite(current_energy):
        return np.inf, None
        
    current_points = current_points - np.mean(current_points, axis=0)
    
    best_energy = current_energy
    best_points = current_points.copy()
    
    scale_factor = 1.0 + 0.3 * np.log10(n / 10.0)
    T = 0.6 * scale_factor
    cooling_rate = 0.96
    min_T = 0.05
    
    for step in range(max_bh_steps):
        progress = step / max_bh_steps
        scale = base_perturb * scale_factor * (1.0 - 0.4 * progress)
        perturbation = rng.normal(scale=scale, size=(n, 3))
        perturbed_points = current_points + perturbation
        
        try:
            new_energy, new_points = relax(perturbed_points, 'sutton_chen', kind)
        except Exception:
            continue
            
        if not np.isfinite(new_energy):
            continue
            
        new_points = new_points - np.mean(new_points, axis=0)
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_energy = new_energy
            current_points = new_points
            if new_energy < best_energy:
                best_energy = new_energy
                best_points = new_points.copy()
                
        T *= cooling_rate
        if T < min_T:
            break
            
    return best_energy, best_points

def construct(instance):
    n = instance['n']
    kind = instance['kind']
    
    start_time = time.time()
    time_limit = 45.0  # Seconds per instance (6 instances * 45s = 270s < 300s)
    
    # Number of restarts: more for smaller clusters
    if n <= 40:
        n_restarts_small = 8
        n_restarts_large = 4
    elif n <= 60:
        n_restarts_small = 6
        n_restarts_large = 4
    else:
        n_restarts_small = 5
        n_restarts_large = 3
    
    base_seed = instance['n'] * 10000
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Standard BH restarts
    seeds_small = [base_seed + i * 7919 for i in range(n_restarts_small)]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for seed in seeds_small:
            futures.append(executor.submit(_run_bh_restart, n, kind, seed))
            
        for future in as_completed(futures):
            try:
                e, p = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = p
            except Exception:
                pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.5:
        # Phase 2: Large perturbation BH restarts
        seeds_large = [base_seed + 1000 + i * 104729 for i in range(n_restarts_large)]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for seed in seeds_large:
                futures.append(executor.submit(_run_bh_restart_large_perturb, n, kind, seed))
                
            for future in as_completed(futures):
                try:
                    e, p = future.result()
                    if e < best_energy:
                        best_energy = e
                        best_positions = p
                except Exception:
                    pass
    
    # Check time again - if we have more time, do a final refinement pass
    elapsed = time.time() - start_time
    if elapsed < time_limit * 0.7 and best_positions is not None:
        # Final refinement: small perturbations around best solution
        rng = np.random.default_rng(base_seed + 99999)
        current_energy, current_points = best_energy, best_positions.copy()
        T = 0.1
        for step in range(30):
            perturbation = rng.normal(scale=0.05, size=(n, 3))
            perturbed_points = current_points + perturbation
            try:
                new_energy, new_points = relax(perturbed_points, 'sutton_chen', kind)
            except Exception:
                continue
            if not np.isfinite(new_energy):
                continue
            new_points = new_points - np.mean(new_points, axis=0)
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
                current_energy = new_energy
                current_points = new_points
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_positions = new_points.copy()
            T *= 0.95
    
    if best_positions is None:
        # Fallback
        rng = np.random.default_rng(n)
        points = rng.normal(size=(n, 3)) * 0.5
        return points
        
    return best_positions
