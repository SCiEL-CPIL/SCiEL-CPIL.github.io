"""Morse cluster global minima search with basin hopping and structured initialization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# Aim to finish by 290s to be safe
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    """Compute Morse energy. Returns huge value if any pair is closer than 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        # Strong penalty for violation
        return 1e6
    # Vectorized Morse energy
    # E = sum exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    # Avoid double counting: sum over i<j
    # d is symmetric. We can sum upper triangle.
    # Or sum all and divide by 2? No, diagonal is inf.
    # Let's use a mask or just sum unique pairs.
    # For n up to 80, n^2 is 6400, which is fine.
    # d contains all pairwise distances.
    # We need to sum for i < j.
    # idx = np.triu_indices(n, 1)
    # r = d[idx]
    # x = rho * (1 - r)
    # e = np.exp(x)
    # return np.sum(e * (e - 2))
    
    # Faster: use full matrix, sum, divide by 2 (since diagonal is inf, we must mask it)
    # Actually, cdist returns 0 on diagonal. We filled with inf.
    # exp(inf) is inf. Sum will be inf.
    # So we must mask diagonal before exp.
    d = np.where(np.eye(n, dtype=bool), np.inf, d)
    x = rho * (1 - d)
    # Clip x to avoid overflow if d is very small (though we check min_d)
    # If min_d >= 0.5, x <= rho * 0.5. For rho=10, x <= 5. exp(5) ~ 148. Safe.
    e = np.exp(x)
    # Sum over all pairs i!=j, then divide by 2
    # The diagonal is inf, so we need to exclude it.
    # Let's just use triu indices for clarity and safety.
    idx = np.triu_indices(n, 1)
    r = d[idx]
    x = rho * (1 - r)
    e = np.exp(x)
    return np.sum(e * (e - 2))


def _morse_grad(rho, coords):
    """Compute Morse gradient."""
    n = len(coords)
    if n < 2:
        return np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        return np.zeros_like(coords) # Gradient undefined or huge, return 0 to stop
    
    grad = np.zeros_like(coords)
    # Loop over pairs for gradient accumulation
    # For n=80, 3160 pairs. This is fast enough in Python? 
    # Vectorized gradient is complex. Let's try a simple loop first, or use scipy's gradient?
    # No, we need analytic.
    # dE/dr = exp(rho(1-r)) * (exp(rho(1-r)) - 2) * (-rho) * 2? No.
    # E_pair = e^x (e^x - 2) where x = rho(1-r)
    # dE/dr = dE/dx * dx/dr
    # dE/dx = e^x(e^x - 2) + e^x(e^x) = 2e^{2x} - 2e^x
    # dx/dr = -rho
    # dE/dr = -rho * (2e^{2x} - 2e^x)
    
    # grad_i = sum_{j!=i} dE/dr_ij * (r_i - r_j) / r_ij
    
    # Vectorized approach:
    # r_ij vector = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    # dist = np.linalg.norm(r_ij, axis=-1)
    # This creates n x n x 3 array. For n=80, 80*80*3 = 19200 floats. Fine.
    
    r_vec = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist = np.linalg.norm(r_vec, axis=-1)
    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    dist = np.where(mask, dist, 1.0) # Avoid div by 0
    
    x = rho * (1 - dist)
    e = np.exp(x)
    dEdr = -rho * (2 * e * e - 2 * e)
    
    # dE/dx_i = sum_j dEdr_ij * (x_i - x_j) / dist_ij
    # dEdr is n x n. r_vec is n x n x 3. dist is n x n.
    # factor = dEdr / dist
    factor = np.where(mask, dEdr / dist, 0.0)
    
    # grad = sum_j factor_ij * r_vec_ij
    # r_vec_ij = coords_i - coords_j
    # grad_i = sum_j factor_ij * (coords_i - coords_j)
    # = sum_j factor_ij * coords_i - sum_j factor_ij * coords_j
    # = coords_i * sum_j factor_ij - sum_j factor_ij * coords_j
    
    sum_factor = np.sum(factor, axis=1)
    grad = coords * sum_factor[:, np.newaxis] - np.sum(factor[:, :, np.newaxis] * coords[np.newaxis, :, :], axis=1)
    
    return grad


def _local_minimize(coords, rho, max_iter=50):
    """Local minimization using L-BFGS-B."""
    def obj_and_grad(x):
        e = _morse_energy(rho, x)
        g = _morse_grad(rho, x)
        return e, g
    
    # Use L-BFGS-B
    res = minimize(
        obj_and_grad, 
        coords, 
        method='L-BFGS-B', 
        jac=True, 
        options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-5}
    )
    return res.x, res.fun


def _fibonacci_sphere(n, radius):
    """Generate n points on a sphere using Fibonacci lattice."""
    phi = np.pi * (3. - np.sqrt(5.))
    i = np.arange(0, n)
    z = (i / (n - 1) * 2. - 1.) if n > 1 else np.array([0.])
    r = np.sqrt(1. - z * z)
    theta = phi * i
    x = np.cos(theta) * r
    y = np.sin(theta) * r
    points = np.vstack([x, y, z]).T
    return points * radius


def _random_cluster(n, radius, min_dist, rng):
    """Generate random cluster with minimum distance constraint."""
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fill with random points if failed
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)


def _icosahedral_cluster(n, rho, rng):
    """Generate icosahedral-like shells."""
    # Approximate number of atoms per shell
    # 1, 12, 42, 92...
    # For n=38, 1+12+25? No, 1+12+42 is 55.
    # Let's just use a dense random packing or Fibonacci scaled.
    # For now, fallback to random or Fibonacci.
    return _random_cluster(n, 1.0 * n**(1/3) * 0.8, 0.8, rng)


def construct(instance):
    """Find the lowest-energy arrangement."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter
    
    # Time check
    if _time_remaining() < 5:
        # Emergency return
        return _fibonacci_sphere(size, 1.0 * size**(1/3) * 0.8)
        
    rng = np.random.default_rng(size + 42)
    
    # Estimate radius
    radius = 1.0 * size**(1.0/3.0) * 0.8
    
    # Initial diverse starts
    starts = []
    
    # 1. Fibonacci Sphere
    starts.append(_fibonacci_sphere(size, radius))
    
    # 2. Random Compact
    starts.append(_random_cluster(size, radius, 0.8, rng))
    
    # 3. Another Random
    starts.append(_random_cluster(size, radius, 0.8, rng))
    
    # 4. "Shelled" - try to pack in shells
    # Simple heuristic: place atoms on concentric spheres
    # Just use random for now to save time in code generation
    
    best_energy = np.inf
    best_pos = starts[0]
    
    # Phase 1: Local Minimize each start
    for start in starts:
        if _time_remaining() < 10:
            break
        try:
            pos, energy = _local_minimize(start, rho, max_iter=50)
            if energy < best_energy and np.min(cdist(pos, pos)[np.triu_indices(size, 1)]) >= 0.5:
                best_energy = energy
                best_pos = pos
        except Exception:
            pass
    
    # Phase 2: Basin Hopping
    # We will run basin hopping on the best solution found so far
    # and also occasionally try new random starts
    
    bh_steps = 0
    max_bh_steps = 500 # Limit steps
    
    while _time_remaining() > 10 and bh_steps < max_bh_steps:
        bh_steps += 1
        
        # Perturbation
        # Scale perturbation with cluster size
        sigma = 0.1 * (size / 38.0) ** 0.5
        perturbation = rng.normal(0, sigma, (size, 3))
        new_pos = best_pos + perturbation
        
        # Relax
        try:
            relaxed_pos, relaxed_energy = _local_minimize(new_pos, rho, max_iter=30)
            
            # Check validity
            if np.min(cdist(relaxed_pos, relaxed_pos)[np.triu_indices(size, 1)]) < 0.5:
                continue
                
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos
                # If we found a significantly better one, we might restart BH
                # but for now just continue
            else:
                # If stuck, try a fresh random start occasionally
                if bh_steps % 10 == 0:
                    fresh_start = _random_cluster(size, radius, 0.8, rng)
                    try:
                        fp, fe = _local_minimize(fresh_start, rho, max_iter=30)
                        if fe < best_energy and np.min(cdist(fp, fp)[np.triu_indices(size, 1)]) >= 0.5:
                            best_energy = fe
                            best_pos = fp
                    except Exception:
                        pass
                        
        except Exception:
            pass
            
    return best_pos
