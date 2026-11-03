"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _score(x):
    """Compute the packing radius for a flat array of 2N coordinates."""
    pts = x.reshape(-1, 2)
    # distances between all pairs
    # Use broadcasting for pairwise distances - but N=200 means 200x200 = 40000 elements, fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist) / 2.0
    # distance to walls: square is [-0.5, 0.5]^2
    wall = np.min(np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1])))
    return min(min_pair, wall)


def _soft_min_obj(x, beta=50.0):
    """Smooth approximation of -min(packing radius) using soft-min.
    We maximize r = min(pair_dist/2, wall_dist).
    Equivalent to minimizing -r.
    Use log-sum-exp style soft-min for smoothness.
    """
    pts = x.reshape(-1, 2)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_r = dist / 2.0
    # Wall distances
    wall = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
    # Combine all constraints into one array
    # For soft-min: we want to approximate min of all values
    # Use: -1/beta * log(sum(exp(-beta * v_i))) approximates min(v_i) as beta->inf
    # But this can overflow. Let's use a different approach.
    # Actually, let's just compute the raw min and use a penalty-based approach.
    # For L-BFGS-B, we need a smooth objective. Let's use:
    # f(x) = -min(pair_r, wall) + penalty for violations
    # But min is non-smooth. Let's use soft-min.
    
    # Collect all "radius candidates"
    pair_vals = pair_r.flatten()
    # Filter out inf
    pair_vals = pair_vals[np.isfinite(pair_vals)]
    all_vals = np.concatenate([pair_vals, wall])
    
    # Soft-min approximation
    min_val = np.min(all_vals)
    # Shift for numerical stability
    shifted = all_vals - min_val
    # Soft min = min_val - (1/beta) * log(sum(exp(-beta * shifted)))
    # Since shifted >= 0, exp(-beta*shifted) <= 1
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    soft_min_val = min_val - log_sum / beta
    
    return -soft_min_val  # We want to minimize negative of radius


def _hex_lattice_init(n_circles, seed=42):
    """Generate a hexagonal lattice initialization clipped to the unit square."""
    rng = np.random.RandomState(seed)
    
    # Hexagonal packing: spacing s, rows offset
    # Area per circle in hex packing: s^2 * sqrt(3)/2
    # For N circles in area 1: s^2 * sqrt(3)/2 * N ≈ 1 => s ≈ sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = []
    row = 0
    while len(points) < n_circles + 20:
        y = -0.5 + (row + 0.5) * s * np.sqrt(3) / 2
        if y > 0.5:
            break
        offset = 0 if row % 2 == 0 else s / 2
        col = 0
        while True:
            x = -0.5 + (col + 0.5) * s + offset
            if x > 0.5:
                break
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
            col += 1
        row += 1
    
    points = np.array(points)
    
    if len(points) < n_circles:
        # Fill remaining with random points
        while len(points) < n_circles:
            p = rng.uniform(-0.45, 0.45, size=2)
            points = np.vstack([points, p])
    else:
        points = points[:n_circles]
    
    # Add small jitter
    points += rng.normal(0, s * 0.05, size=points.shape)
    # Clip to square
    points = np.clip(points, -0.49, 0.49)
    
    return points


def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin before 150s
    
    best_x = None
    best_score = 0.0
    
    # Try multiple seeds for hex lattice initialization
    seeds = [42, 123, 777, 2024, 999, 555, 111, 888]
    
    for seed in seeds:
        if time.time() > deadline:
            break
        
        x0 = _hex_lattice_init(N, seed=seed).flatten()
        
        # Try L-BFGS-B with soft-min objective
        try:
            result = minimize(
                _soft_min_obj,
                x0,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < 0:
                s = _score(result.x)
                if s > best_score:
                    best_score = s
                    best_x = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline:
            break
    
    # If we have a good result, try to polish it
    if best_x is not None:
        # Try Nelder-Mead polish on the best result
        if time.time() < deadline - 20:
            try:
                result_nm = minimize(
                    _soft_min_obj,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                s = _score(result_nm.x)
                if s > best_score:
                    best_score = s
                    best_x = result_nm.x.copy()
            except Exception:
                pass
        
        # Final high-precision polish with tighter beta
        if time.time() < deadline - 10:
            def _soft_min_obj_tight(x, beta=200.0):
                pts = x.reshape(-1, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dist, np.inf)
                pair_r = dist / 2.0
                wall = np.minimum(0.5 - np.abs(pts[:, 0]), 0.5 - np.abs(pts[:, 1]))
                pair_vals = pair_r.flatten()
                pair_vals = pair_vals[np.isfinite(pair_vals)]
                all_vals = np.concatenate([pair_vals, wall])
                min_val = np.min(all_vals)
                shifted = all_vals - min_val
                log_sum = np.log(np.sum(np.exp(-beta * shifted)))
                soft_min_val = min_val - log_sum / beta
                return -soft_min_val
            
            try:
                result_tight = minimize(
                    _soft_min_obj_tight,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (2 * N),
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                s = _score(result_tight.x)
                if s > best_score:
                    best_score = s
                    best_x = result_tight.x.copy()
            except Exception:
                pass
    
    if best_x is None:
        # Fallback: return hex lattice
        best_x = _hex_lattice_init(N, seed=42).flatten()
    
    pts = best_x.reshape(-1, 2)
    # Ensure points are strictly inside the square
    pts = np.clip(pts, -0.4999, 0.4999)
    
    return pts
# EVOLVE-BLOCK-END
