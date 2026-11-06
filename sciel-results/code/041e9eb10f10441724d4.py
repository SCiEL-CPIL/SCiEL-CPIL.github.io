"""Packing 50 equal circles in a unit circle; uses local search with multiple seeds."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
TIME_LIMIT = 110.0  # seconds, leaving margin before 120s kill

def compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if len(pts) < 2:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    radii = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary_dist = np.min(1.0 - radii)
    
    if min_boundary_dist < 0:
        return 0.0  # Invalid: center outside container
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r

def seed_hexagonal():
    """Generate a hexagonal lattice packing seed."""
    # Try different spacings to get at least N points
    for spacing in np.linspace(0.12, 0.20, 200):
        k = int(1.0 / spacing) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    rows.append([x, y])
        if len(rows) >= N:
            pts = np.array(rows)
            # Sort by distance from center, take closest N
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists, kind="stable")[:N]
            return pts[idx]
    # Fallback: random
    return np.random.uniform(-0.8, 0.8, size=(N, 2))

def seed_random():
    """Generate random points inside the unit circle."""
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.random.uniform(0, 0.9, N)
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def seed_ring():
    """Generate points on concentric rings."""
    # Ring 0: center
    pts = [[0.0, 0.0]]
    # Ring 1: 6 points
    r1 = 0.25
    for i in range(6):
        angle = 2 * np.pi * i / 6
        pts.append([r1 * np.cos(angle), r1 * np.sin(angle)])
    # Ring 2: 12 points
    r2 = 0.50
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        pts.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    # Ring 3: 18 points
    r3 = 0.75
    for i in range(18):
        angle = 2 * np.pi * i / 18
        pts.append([r3 * np.cos(angle), r3 * np.sin(angle)])
    # Ring 4: 13 points
    r4 = 0.95
    for i in range(13):
        angle = 2 * np.pi * i / 13 + np.pi / 13
        pts.append([r4 * np.cos(angle), r4 * np.sin(angle)])
    
    pts = np.array(pts[:N])
    return pts

def optimize_from_seed(pts, time_budget):
    """Optimize a seed using local search with gradient-free methods."""
    start_time = time.time()
    best_pts = pts.copy()
    best_r = compute_radius(pts)
    
    # Use Nelder-Mead on the negative radius objective
    def objective(x):
        x = x.reshape(N, 2)
        return -compute_radius(x)
    
    # Try multiple local optimizations with different initial perturbations
    n_attempts = 5
    for attempt in range(n_attempts):
        if time.time() - start_time > time_budget * 0.8:
            break
        
        if attempt == 0:
            x0 = pts.flatten()
        else:
            # Perturb the best solution
            noise = np.random.normal(0, 0.05 * best_r, size=(N, 2))
            x0 = (pts + noise).flatten()
        
        # Clamp to ensure validity
        for i in range(N):
            r = np.hypot(x0[2*i], x0[2*i+1])
            if r > 0.99:
                scale = 0.99 / r
                x0[2*i] *= scale
                x0[2*i+1] *= scale
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if result.success or result.fun < objective(x0):
                new_pts = result.x.reshape(N, 2)
                # Validate
                new_r = compute_radius(new_pts)
                if new_r > best_r:
                    best_pts = new_pts
                    best_r = new_r
        except Exception:
            pass
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    
    # Generate multiple seeds
    seeds = [
        seed_hexagonal(),
        seed_ring(),
        seed_random(),
        seed_random(),
        seed_random(),
    ]
    
    best_pts = None
    best_r = 0.0
    
    # Time budget per seed
    n_seeds = len(seeds)
    per_seed_time = (TIME_LIMIT - 5.0) / n_seeds  # 5s margin
    
    for i, seed in enumerate(seeds):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - 5.0 - elapsed
        if remaining < 1.0:
            break
        
        # Optimize this seed
        time_budget = min(per_seed_time, remaining * 0.9)
        opt_pts, opt_r = optimize_from_seed(seed, time_budget)
        
        if opt_r > best_r:
            best_r = opt_r
            best_pts = opt_pts
    
    # If we still have time, do more local refinement on the best
    if best_pts is not None:
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - 3.0 - elapsed
        if remaining > 5.0:
            best_pts, best_r = optimize_from_seed(best_pts, remaining * 0.8)
    
    # Final validation
    if best_pts is None:
        # Fallback
        best_pts = seed_hexagonal()
        best_r = compute_radius(best_pts)
    
    # Ensure all points are strictly inside
    for i in range(N):
        r = np.hypot(best_pts[i, 0], best_pts[i, 1])
        if r >= 1.0:
            scale = 0.999 / max(r, 1e-10)
            best_pts[i] *= scale
    
    return best_pts
