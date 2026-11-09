"""Improved program for Sutton-Chen cluster global minima search using Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import math

def _generate_random_sphere(n, radius, rng):
    """Generate n random points inside a sphere of given radius."""
    points = []
    for _ in range(n):
        u = rng.random(3)
        r = radius * (u[0] ** (1/3.0))
        theta = 2 * np.pi * u[1]
        phi = np.arccos(2 * u[2] - 1)
        points.append([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
    return np.array(points)

def _generate_shell(n, radius, rng):
    """Generate n random points on a sphere surface."""
    points = []
    for _ in range(n):
        u = rng.random(3)
        theta = 2 * np.pi * u[1]
        phi = np.arccos(2 * u[2] - 1)
        points.append([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi)
        ])
    return np.array(points)

def _basin_hopping(x_init, potential, kind, seed, n_steps, sigma_init, sigma_final, t_init, t_final, time_limit, start_time):
    """Perform a basin hopping search starting from x_init."""
    rng = np.random.default_rng(seed)
    # Initial relaxation
    try:
        E_curr, X_curr = relax(x_init, potential, kind)
        if not np.isfinite(E_curr):
            return np.inf, x_init
    except Exception:
        return np.inf, x_init

    best_E = E_curr
    best_X = X_curr.copy()
    
    n = len(X_curr)
    
    for i in range(n_steps):
        # Check time
        if time.time() - start_time > time_limit:
            break
            
        # Cooling schedule
        progress = i / max(1, n_steps - 1)
        sigma = sigma_init * (sigma_final / sigma_init) ** progress
        T = t_init * (t_final / t_init) ** progress
        
        # Perturb
        noise = rng.normal(0, sigma, X_curr.shape)
        X_pert = X_curr + noise
        
        # Relax
        try:
            E_new, X_new = relax(X_pert, potential, kind)
            if not np.isfinite(E_new):
                continue
        except Exception:
            continue
            
        # Metropolis acceptance
        dE = E_new - E_curr
        if dE < 0:
            X_curr = X_new
            E_curr = E_new
        else:
            # Avoid log(0)
            prob = np.exp(-dE / T) if T > 1e-10 else 0.0
            if rng.random() < prob:
                X_curr = X_new
                E_curr = E_new
                
        # Update best
        if E_curr < best_E:
            best_E = E_curr
            best_X = X_curr.copy()
            
    return best_E, best_X

def construct(instance):
    """Basin Hopping optimization for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    start_time = time.time()
    time_limit = 45.0 # Allow ~75s per instance for 6 instances in 300s total
    
    # Estimate radius
    radius = (size / (4.0/3.0 * np.pi * 0.74)) ** (1/3.0) * 1.1
    
    # Generate diverse initial structures
    n_starts = 8
    initial_structures = []
    for i in range(n_starts):
        if i % 3 == 0:
            x = _generate_random_sphere(size, radius, rng)
        elif i % 3 == 1:
            x = _generate_shell(size, radius, rng)
        else:
            x = _generate_random_sphere(size, radius * 1.2, rng)
        initial_structures.append(x)
        
    best_energy = np.inf
    best_positions = None
    
    # Run basin hopping in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, x_init in enumerate(initial_structures):
            futures.append(executor.submit(
                _basin_hopping, x_init, potential, kind, size * 1000 + i * 100 + 42, 
                50, 0.5, 0.05, 1.0, 0.01, time_limit, start_time
            ))
            
        for f in as_completed(futures):
            try:
                E, X = f.result()
                if E < best_energy:
                    best_energy = E
                    best_positions = X
            except Exception:
                pass
                
    # Fallback
    if best_positions is None or not np.isfinite(best_energy):
        x0 = _generate_random_sphere(size, radius, rng)
        try:
            E, X = relax(x0, potential, kind)
            if np.isfinite(E):
                best_energy = E
                best_positions = X
        except Exception:
            best_positions = x0

    return best_positions
# EVOLVE-BLOCK-END
