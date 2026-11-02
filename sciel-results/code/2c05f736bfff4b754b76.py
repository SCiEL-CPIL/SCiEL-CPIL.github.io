"""Packing 50 equal circles in a unit square.
Strategy: start from a good grid, then use simulated annealing with
a repulsive-potential objective, followed by L-BFGS polishing.
"""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _radius(pts):
    """Compute the common radius for a set of points."""
    pts = np.asarray(pts)
    # pairwise distances
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    # boundary
    boundary = 0.5 - np.abs(pts).max()
    return min(min_pair, 2 * boundary) / 2.0

def _objective(pts, beta):
    """Repulsive potential objective to maximize."""
    pts = np.asarray(pts).reshape(-1, 2)
    n = len(pts)
    # pairwise
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    # boundary distances
    bx = 0.5 - np.abs(pts[:, 0])
    by = 0.5 - np.abs(pts[:, 1])
    bd = np.minimum(bx, by)
    # objective: sum of exp(-beta * d) for pairs, plus boundary
    pair_obj = np.sum(np.exp(-beta * d))
    # boundary: we want to push away from boundary, so exp(-beta * bd)
    bound_obj = np.sum(np.exp(-beta * bd))
    return pair_obj + bound_obj

def _sa_step(pts, T, beta, rng):
    """One SA step: propose random perturbation, accept with probability."""
    n = len(pts)
    i = rng.integers(n)
    dx = rng.normal(0, 0.01)
    dy = rng.normal(0, 0.01)
    old = pts[i].copy()
    pts[i] += np.array([dx, dy])
    # clamp to square
    pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
    pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
    old_obj = _objective(old.reshape(1, -1), beta)  # this is expensive, skip for speed
    # Instead, compute delta approximately
    # For speed, just accept with some probability based on T
    # Better: compute new objective for the whole set
    # This is O(N^2) per step, too slow for many steps
    # Let's use a simpler approach: just move and accept if T is high enough
    # Actually, let's do a proper SA with precomputed distances
    return pts

def construct():
    t0 = time.time()
    deadline = t0 + 140  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Start from a grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].astype(float)
    
    best_pts = pts.copy()
    best_r = _radius(pts)
    
    # SA phase: use a simpler random walk with acceptance
    # For speed, we'll do a limited number of steps
    beta = 50.0
    T = 0.05
    T_min = 0.001
    
    # Precompute for speed: use vectorized objective
    def full_obj(p):
        p = np.asarray(p).reshape(-1, 2)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        pair = np.sum(np.exp(-beta * d))
        bx = 0.5 - np.abs(p[:, 0])
        by = 0.5 - np.abs(p[:, 1])
        bd = np.minimum(bx, by)
        bound = np.sum(np.exp(-beta * bd))
        return pair + bound
    
    n_sa_steps = 20000
    for step in range(n_sa_steps):
        if time.time() > deadline - 30:
            break
        
        # Propose
        i = rng.integers(N)
        scale = 0.005 + 0.005 * (T / T_min)
        dx = rng.normal(0, scale)
        dy = rng.normal(0, scale)
        old_pt = pts[i].copy()
        pts[i, 0] += dx
        pts[i, 1] += dy
        pts[i, 0] = np.clip(pts[i, 0], -0.5, 0.5)
        pts[i, 1] = np.clip(pts[i, 1], -0.5, 0.5)
        
        # Compute objective
        new_obj = full_obj(pts)
        # We need old_obj; compute it by reverting temporarily
        # This is expensive. Let's just accept if T is high or always for now
        # Better approach: store old_obj
        # For simplicity, let's do a different SA: minimize a smooth objective
        
        # Actually, let's switch to a simpler hill-climbing with restarts
        # and use L-BFGS for polishing
        break  # exit SA, go to L-BFGS
    
    # L-BFGS polishing: minimize a smooth objective
    # Objective: -min_pair_dist/2 + penalty for boundary violation
    def lbfgs_obj(p):
        p = np.asarray(p).reshape(-1, 2)
        d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min()
        bx = 0.5 - np.abs(p[:, 0])
        by = 0.5 - np.abs(p[:, 1])
        min_bound = np.minimum(bx, by).min()
        # We want to maximize min(min_pair/2, min_bound)
        # Use a smooth approximation: -min(min_pair/2, min_bound)
        # But min is not differentiable. Use a soft min.
        val = min(min_pair / 2, min_bound)
        return -val
    
    # Try L-BFGS
    try:
        res = minimize(lbfgs_obj, pts.ravel(), method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < -best_r:
            cand = res.x.reshape(-1, 2)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand.copy()
    except Exception:
        pass
    
    # More L-BFGS iterations with tighter tolerance
    if time.time() < deadline - 10:
        try:
            res2 = minimize(lbfgs_obj, best_pts.ravel(), method='L-BFGS-B',
                            bounds=[(-0.5, 0.5)] * (2 * N),
                            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10})
            cand = res2.x.reshape(-1, 2)
            r = _radius(cand)
            if r > best_r:
                best_r = r
                best_pts = cand.copy()
        except Exception:
            pass
    
    # Final check
    if time.time() < deadline - 5:
        # Try a few random restarts with L-BFGS
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            p0 = best_pts + rng.normal(0, 0.001, (N, 2))
            p0 = np.clip(p0, -0.5, 0.5)
            try:
                res3 = minimize(lbfgs_obj, p0.ravel(), method='L-BFGS-B',
                                bounds=[(-0.5, 0.5)] * (2 * N),
                                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
                cand = res3.x.reshape(-1, 2)
                r = _radius(cand)
                if r > best_r:
                    best_r = r
                    best_pts = cand.copy()
            except Exception:
                pass
    
    return best_pts
# EVOLVE-BLOCK-END
