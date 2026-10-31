"""Heilbronn convex regions: 13 points maximizing min triangle area / hull area."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import ConvexHull


def _compute_vertices(offsets):
    """
    Compute the 13 vertices of the convex polygon defined by 13 half-planes.
    offsets: array of 13 distances from origin to support lines.
    Normals are fixed at evenly spaced angles.
    """
    n = len(offsets)
    angles = 2.0 * np.pi * np.arange(n) / n
    # Normals
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    vertices = np.zeros((n, 2))
    for i in range(n):
        n1 = normals[i]
        n2 = normals[(i + 1) % n]
        d1 = offsets[i]
        d2 = offsets[(i + 1) % n]
        
        # Solve:
        # n1[0]*x + n1[1]*y = d1
        # n2[0]*x + n2[1]*y = d2
        A = np.array([n1, n2])
        b = np.array([d1, d2])
        try:
            vertices[i] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Should not happen for distinct angles
            return None
            
    return vertices

def _min_triangle_area(pts):
    """Min area over all C(n,3) triples."""
    n = len(pts)
    min_a = np.inf
    # Use vectorized calculation for speed if possible, but n=13 is small
    # C(13,3) = 286 triangles. Simple loop is fast enough.
    for i in range(n - 2):
        pi = pts[i]
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pi
            for k in range(j + 1, n):
                v2 = pts[k] - pi
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def _polygon_area(pts):
    """Shoelace formula for polygon area. Assumes CCW order."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return 0.5 * abs(area)

def _score(params):
    """
    params: 12 values. The 13th offset is fixed to 1.0 to break scale invariance.
    Returns min_triangle_area / hull_area.
    """
    # Reconstruct full offsets
    offsets = np.ones(N)
    offsets[1:] = params # params are offsets 1..12, offset 0 is 1.0
    
    # Ensure offsets are positive (convex region containing origin? No, just convex polygon)
    # If any offset <= 0, the polygon might not contain the origin or be degenerate.
    # For a convex polygon defined by support functions, offsets should be positive if origin is inside.
    # If origin is not inside, this representation is tricky. 
    # Standard support function h(u) >= 0 for all u if origin is inside.
    # Let's enforce positive offsets.
    if np.any(offsets <= 0):
        return 0.0
        
    vertices = _compute_vertices(offsets)
    if vertices is None:
        return 0.0
        
    # Check if points are distinct
    # Calculate min distance between vertices
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(vertices[i] - vertices[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 1e-6:
        return 0.0
        
    ta = _min_triangle_area(vertices)
    ha = _polygon_area(vertices)
    
    if ha < 1e-10:
        return 0.0
        
    return ta / ha

def _score_neg(params):
    return -_score(params)

def construct():
    t_start = time.time()
    DEADLINE = 140.0
    
    # Bounds for the 12 relative offsets.
    # If offset 0 is 1.0, others should be in a reasonable range to keep polygon non-degenerate.
    # If an offset is too small, the polygon shrinks. If too large, it expands.
    # Since we normalize by area, scale doesn't matter, but relative shape does.
    # Let's search in [0.5, 1.5] for simplicity, centered around regular polygon (all 1s).
    bounds = [(0.5, 1.5)] * (N - 1)
    
    best_params = np.ones(N - 1)
    best_score = _score(best_params)
    
    # 1. Differential Evolution
    if time.time() < DEADLINE - 30:
        try:
            result = differential_evolution(
                _score_neg, bounds,
                seed=42, maxiter=50, tol=1e-6,
                popsize=15, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, init='latinhypercube',
                callback=lambda x, convergence: (time.time() > DEADLINE - 35)
            )
            s = _score(result.x)
            if s > best_score:
                best_score = s
                best_params = result.x.copy()
        except Exception:
            pass
    
    # 2. Nelder-Mead Refinement
    if time.time() < DEADLINE - 10:
        try:
            res = minimize(
                _score_neg, best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            s = _score(res.x)
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
            
    # 3. Final Polish / Local Search
    if time.time() < DEADLINE - 5:
        # Try a few random perturbations around best
        rng = np.random.RandomState(123)
        for _ in range(5):
            if time.time() > DEADLINE - 5:
                break
            pert = best_params + rng.normal(0, 0.01, size=N-1)
            try:
                res = minimize(
                    _score_neg, pert,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                s = _score(res.x)
                if s > best_score:
                    best_score = s
                    best_params = res.x.copy()
            except Exception:
                pass

    # Reconstruct final points
    offsets = np.ones(N)
    offsets[1:] = best_params
    vertices = _compute_vertices(offsets)
    
    # Fallback if something went wrong
    if vertices is None or _polygon_area(vertices) < 1e-10:
        angles = 2.0 * np.pi * np.arange(N) / N
        vertices = np.column_stack([np.cos(angles), np.sin(angles)])
        
    return vertices
# EVOLVE-BLOCK-END
