"""Heilbronn convex regions n=14 — centrally symmetric parameterization + optimization."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 14
HALF = N // 2  # 7 points, the rest are their negatives
DEADLINE = 140.0  # safety margin before 150s kill

# Precompute all triple indices for the full 14-point set
# We'll build the full set from 7 base points + their negatives
def _build_full_points(base7):
    """base7: shape (7,2). Returns 14x2 array: base7 then -base7."""
    return np.vstack([base7, -base7])

def _score(points14):
    """Min triangle area / convex hull area. Higher is better."""
    n = points14.shape[0]
    if n < 3:
        return 0.0
    # Convex hull area
    try:
        hull = ConvexHull(points14)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area <= 0:
        return 0.0
    # All triples
    min_area = np.inf
    for i in range(n):
        pi = points14[i]
        for j in range(i + 1, n):
            pj = points14[j]
            for k in range(j + 1, n):
                pk = points14[k]
                # Area = 0.5 * |cross(pj-pi, pk-pi)|
                ax = pj[0] - pi[0]
                ay = pj[1] - pi[1]
                bx = pk[0] - pi[0]
                by = pk[1] - pi[1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
    return min_area / hull_area

def _neg_score(base7_flat):
    """Objective for minimization: negative of score."""
    base7 = base7_flat.reshape(HALF, 2)
    full = _build_full_points(base7)
    return -_score(full)

def _neg_score_fast(base7_flat):
    """Faster version with early exit if hull area is tiny."""
    base7 = base7_flat.reshape(HALF, 2)
    full = _build_full_points(base7)
    # Quick hull area check
    try:
        hull = ConvexHull(full)
        hull_area = hull.volume
    except Exception:
        return 1e10
    if hull_area < 1e-12:
        return 1e10
    min_area = np.inf
    for i in range(N):
        pi = full[i]
        for j in range(i + 1, N):
            pj = full[j]
            for k in range(j + 1, N):
                pk = full[k]
                ax = pj[0] - pi[0]
                ay = pj[1] - pi[1]
                bx = pk[0] - pi[0]
                by = pk[1] - pi[1]
                area = 0.5 * abs(ax * by - ay * bx)
                if area < min_area:
                    min_area = area
                    if min_area < 1e-12:
                        break
            if min_area < 1e-12:
                break
        if min_area < 1e-12:
            break
    return -min_area / hull_area

def construct():
    start_time = time.time()
    
    # Initial guess: regular heptagon vertices (centrally symmetric pair)
    # Actually, for central symmetry we need 7 points whose negatives are the other 7.
    # A good starting point: vertices of a regular 14-gon, take 7 of them.
    # Regular 14-gon vertices at angles 2*pi*k/14 for k=0..13
    # Take k=0,1,2,3,4,5,6 -> their negatives are k=7,8,9,10,11,12,13
    angles = 2 * np.pi * np.arange(HALF) / N
    r = 1.0
    init = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    best_flat = init.flatten()
    best_score = _score(_build_full_points(init))
    
    # Bounds: keep points in a reasonable range
    bounds = [(-2.0, 2.0)] * (HALF * 2)
    
    # Phase 1: Differential Evolution (global search)
    try:
        remaining = DEADLINE - (time.time() - start_time)
        if remaining > 20:
            de_result = differential_evolution(
                _neg_score_fast,
                bounds=bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=True,
                updating='deferred',
                workers=1,
                callback=lambda x, convergence: None,
            )
            if de_result.success or de_result.fun < 0:
                cand = de_result.x
                s = _score(_build_full_points(cand.reshape(HALF, 2)))
                if s > best_score:
                    best_score = s
                    best_flat = cand.copy()
    except Exception:
        pass
    
    # Phase 2: Local refinement with L-BFGS-B
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 10:
        try:
            res = minimize(
                _neg_score,
                best_flat,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10},
            )
            cand = res.x
            s = _score(_build_full_points(cand.reshape(HALF, 2)))
            if s > best_score:
                best_score = s
                best_flat = cand.copy()
        except Exception:
            pass
    
    # Phase 3: Additional local refinement from multiple perturbations
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        rng = np.random.RandomState(42)
        for _ in range(3):
            if time.time() - start_time > DEADLINE - 3:
                break
            perturbed = best_flat + rng.normal(0, 0.05, size=best_flat.shape)
            try:
                res = minimize(
                    _neg_score,
                    perturbed,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10},
                )
                cand = res.x
                s = _score(_build_full_points(cand.reshape(HALF, 2)))
                if s > best_score:
                    best_score = s
                    best_flat = cand.copy()
            except Exception:
                pass
    
    # Final deadline guard
    if time.time() - start_time > DEADLINE:
        pass  # Return best so far
    
    return _build_full_points(best_flat.reshape(HALF, 2))
