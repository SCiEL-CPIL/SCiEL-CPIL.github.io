"""Heilbronn convex regions: Maximize minimum triangle area / hull area."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 13

# EVOLVE-BLOCK-START
import random

def _all_tri_areas(pts):
    """Vectorized computation of all triangle areas."""
    n = len(pts)
    areas = np.empty((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                areas[i, j, k] = 0.5 * abs(cross)
    # Only keep i < j < k
    mask = np.zeros((n, n, n), dtype=bool)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                mask[i, j, k] = True
    return areas[mask]

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-9

def _score(pts):
    """Compute min triangle area / hull area."""
    pts = np.asarray(pts)
    if pts.shape != (N, 2):
        return -1e9
    areas = _all_tri_areas(pts)
    min_area = np.min(areas)
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return -1e9
    return min_area / h_area

def _soft_min_objective(params, tau=0.01):
    """
    Smooth surrogate for the negative of min triangle area / hull area.
    Uses log-sum-exp to approximate the minimum.
    """
    pts = params.reshape(N, 2)
    areas = _all_tri_areas(pts)
    h_area = _hull_area(pts)
    if h_area < 1e-9:
        return 1e9
    
    # We want to maximize min(areas)/h_area
    # Equivalent to minimizing -min(areas)/h_area
    # min(areas) = -tau * log(sum(exp(-areas/tau)))
    # So -min(areas)/h_area = tau/h_area * log(sum(exp(-areas/tau)))
    
    neg_areas = -areas
    max_neg = np.max(neg_areas)
    if max_neg > 700:
        max_neg = 700
    sum_exp = np.sum(np.exp(neg_areas - max_neg))
    log_sum_exp = max_neg + np.log(sum_exp)
    
    soft_min = -tau * log_sum_exp
    return -soft_min / h_area

def _exact_objective(params):
    """Exact objective: negative of min triangle area / hull area."""
    pts = params.reshape(N, 2)
    return -_score(pts)

def _halton_sequence(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def _halton_points_2d(n):
    pts = np.array([[_halton_sequence(i, 2), _halton_sequence(i, 3)] for i in range(1, n+1)])
    pts -= np.mean(pts, axis=0)
    return pts

def _regular_polygon(n):
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack((np.cos(angles), np.sin(angles)))

def _concentric_ring_init(n, n_outer=8, n_inner=5):
    """Initialize with points on two concentric rings."""
    pts = []
    # Outer ring
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False)
    outer = np.column_stack((np.cos(angles_outer), np.sin(angles_outer))) * 1.0
    pts.extend(outer.tolist())
    # Inner ring
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + np.pi/n_inner
    inner = np.column_stack((np.cos(angles_inner), np.sin(angles_inner))) * 0.5
    pts.extend(inner.tolist())
    pts = np.array(pts[:n])
    pts -= np.mean(pts, axis=0)
    return pts

def _nelder_mead_polish(pts, tau=0.005, maxiter=2000):
    """Polish with Nelder-Mead on the soft-min objective."""
    try:
        res = minimize(
            _soft_min_objective,
            pts.flatten(),
            args=(tau,),
            method='Nelder-Mead',
            options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _lbfgs_optimize(pts, tau=0.02, maxiter=5000):
    """Optimize with L-BFGS-B."""
    try:
        res = minimize(
            _soft_min_objective,
            pts.flatten(),
            args=(tau,),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter*10, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x.reshape(N, 2)
    except Exception:
        return pts

def _grid_init(n):
    """Grid-based initialization for small N."""
    # For 13 points, use a grid-like pattern
    pts = []
    # Center
    pts.append([0.0, 0.0])
    # Ring 1: 4 points at 45 degrees
    for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        pts.append([0.6*np.cos(angle), 0.6*np.sin(angle)])
    # Ring 2: 8 points
    for i in range(8):
        angle = 2*np.pi*i/8 + np.pi/8
        pts.append([1.0*np.cos(angle), 1.0*np.sin(angle)])
    pts = np.array(pts[:n])
    pts -= np.mean(pts, axis=0)
    return pts

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    rng = np.random.default_rng(42)
    
    best_score = -1e9
    best_pts = None
    
    # Generate diverse initializations
    inits = []
    
    # Regular polygon
    inits.append(_regular_polygon(N))
    
    # Halton sequence
    inits.append(_halton_points_2d(N))
    
    # Concentric rings
    inits.append(_concentric_ring_init(N))
    
    # Grid-based
    inits.append(_grid_init(N))
    
    # Random starts
    for _ in range(8):
        rand_pts = rng.random((N, 2)) * 2 - 1
        rand_pts -= np.mean(rand_pts, axis=0)
        inits.append(rand_pts)
    
    # Perturbed regular polygon
    for _ in range(4):
        noise = rng.normal(0, 0.3, size=N*2)
        perturbed = _regular_polygon(N).flatten() + noise
        perturbed = perturbed.reshape(N, 2)
        perturbed -= np.mean(perturbed, axis=0)
        inits.append(perturbed)
    
    # Phase 1: Coarse optimization with multiple starts
    for i, init in enumerate(inits):
        if time.time() - start_time > time_limit - 30.0:
            break
        
        # Use higher temperature for coarse search
        tau = 0.05 if i < 6 else 0.02
        result = _lbfgs_optimize(init, tau=tau, maxiter=3000)
        score = _score(result)
        
        if score > best_score:
            best_score = score
            best_pts = result.copy()
    
    # Phase 2: Fine optimization
    if best_pts is not None:
        if time.time() - start_time < time_limit - 20.0:
            # Continue with the best found so far
            result = _lbfgs_optimize(best_pts, tau=0.01, maxiter=5000)
            score = _score(result)
            if score > best_score:
                best_score = score
                best_pts = result.copy()
        
        # Try a few more refined starts
        for _ in range(4):
            if time.time() - start_time > time_limit - 15.0:
                break
            noise = rng.normal(0, 0.1, size=N*2)
            perturbed = best_pts.flatten() + noise
            perturbed = perturbed.reshape(N, 2)
            perturbed -= np.mean(perturbed, axis=0)
            
            result = _lbfgs_optimize(perturbed, tau=0.005, maxiter=3000)
            score = _score(result)
            if score > best_score:
                best_score = score
                best_pts = result.copy()
    
    # Phase 3: Nelder-Mead polish
    if best_pts is not None and time.time() - start_time < time_limit - 10.0:
        polished = _nelder_mead_polish(best_pts, tau=0.005, maxiter=2000)
        score = _score(polished)
        if score > best_score:
            best_score = score
            best_pts = polished.copy()
    
    # Phase 4: Final refinement with very small tau
    if best_pts is not None and time.time() - start_time < time_limit - 5.0:
        final = _lbfgs_optimize(best_pts, tau=0.001, maxiter=3000)
        score = _score(final)
        if score > best_score:
            best_score = score
            best_pts = final.copy()
        
        final2 = _nelder_mead_polish(best_pts, tau=0.001, maxiter=1000)
        score = _score(final2)
        if score > best_score:
            best_score = score
            best_pts = final2.copy()
    
    if best_pts is None:
        # Fallback to regular polygon
        return _regular_polygon(N)
        
    return best_pts

# EVOLVE-BLOCK-END
