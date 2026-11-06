"""Evolved program: optimize 8 points in the plane to maximize product of pairwise distances at fixed diameter 2."""

import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _log_product_distances(points):
    """Compute log of product of all pairwise distances (unordered pairs)."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < 1e-15:
                return -1e30  # penalty for coincident points
            total += np.log(d)
    return total

def _score(points):
    """Score = Delta / n^n where Delta = product over ordered pairs of distances,
    after scaling set to diameter 2. Regular octagon scores 1."""
    n = len(points)
    # Compute all pairwise distances
    dists = []
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = points[i, 0] - points[j, 0]
                dy = points[i, 1] - points[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                dists.append(d)
    
    if len(dists) == 0:
        return 0.0
    
    # Diameter = max distance
    diam = max(dists)
    if diam < 1e-15:
        return 0.0
    
    # Scale to diameter 2
    scale = 2.0 / diam
    # Product over ordered pairs (each unordered pair appears twice)
    # Delta = product of all 56 ordered pair distances (for n=8, 8*7=56 ordered pairs)
    log_delta = 0.0
    for d in dists:
        if d < 1e-15:
            return 0.0
        log_delta += np.log(d * scale)
    
    score = np.exp(log_delta - n * np.log(n))
    return score

def _optimize_local(x0, timeout=10):
    """Local optimization using Nelder-Mead."""
    def neg_log_score(params):
        pts = params.reshape(8, 2)
        # Check distinctness
        for i in range(8):
            for j in range(i+1, 8):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 1e-8:
                    return 1e10
        s = _score(pts)
        if s <= 0:
            return 1e10
        return -np.log(s)
    
    try:
        result = minimize(neg_log_score, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12})
        return result.x
    except:
        return x0

def construct():
    start_time = time.time()
    timeout = 135  # Safety margin from 150s limit
    
    n = 8
    # Regular octagon initial guess (diameter 2 => radius 1)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    octagon = np.column_stack([np.cos(angles), np.sin(angles)])
    best_pts = octagon.copy()
    best_score = _score(octagon)
    
    # Strategy: Use differential evolution on a geometric parameterization
    # Represent points as r_k * (cos(theta_k), sin(theta_k)) for k=0..7
    # This gives 16 parameters but we can reduce by fixing one angle and one radius
    # Actually, let's just do DE on 16 parameters (x,y for each point)
    
    # DE bounds: points should be within [-2, 2] for reasonable diameter
    bounds = [(-2.0, 2.0)] * (2 * n)
    
    def de_objective(params):
        pts = params.reshape(n, 2)
        s = _score(pts)
        return -s  # minimize negative score
    
    # Run differential evolution
    try:
        de_result = differential_evolution(
            de_objective, bounds,
            seed=42, maxiter=300, popsize=15,
            tol=1e-8, mutation=(0.5, 1.0), recombination=0.8,
            polish=False, updating='immediate'
        )
        if time.time() - start_time < timeout - 10:
            cand = de_result.x.reshape(n, 2)
            s = _score(cand)
            if s > best_score:
                best_pts = cand.copy()
                best_score = s
    except Exception:
        pass
    
    # Local polish from best DE result
    if time.time() - start_time < timeout - 15:
        polished = _optimize_local(best_pts.flatten(), timeout=timeout - time.time() - 5)
        cand_pts = polished.reshape(n, 2)
        s = _score(cand_pts)
        if s > best_score:
            best_pts = cand_pts.copy()
            best_score = s
    
    # Try a few local starts from perturbed octagon
    if time.time() - start_time < timeout - 20:
        rng = np.random.RandomState(42)
        for trial in range(20):
            if time.time() - start_time > timeout - 20:
                break
            perturbation = 0.1 * rng.randn(n, 2)
            x0 = (octagon + perturbation).flatten()
            polished = _optimize_local(x0, timeout=min(5, timeout - time.time() - 5))
            cand_pts = polished.reshape(n, 2)
            s = _score(cand_pts)
            if s > best_score:
                best_pts = cand_pts.copy()
                best_score = s
    
    # Final check
    if time.time() - start_time > timeout - 5:
        return best_pts.tolist()
    
    # Final high-precision polish
    polished = _optimize_local(best_pts.flatten(), timeout=min(10, timeout - time.time() - 2))
    cand_pts = polished.reshape(n, 2)
    s = _score(cand_pts)
    if s > best_score:
        best_pts = cand_pts.copy()
        best_score = s
    
    return best_pts.tolist()
