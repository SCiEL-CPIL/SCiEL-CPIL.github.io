"""Heilbronn convex regions - maximize minimum triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples of points."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area of triangle using cross product
                v1 = pts[j] - pts[i]
                v2 = pts[k] - pts[i]
                area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                if area < min_area:
                    min_area = area
    return min_area

def _convex_hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume  # In 2D, volume is area
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _neg_score(pts):
    return -_score(pts)

def _eval_neg(x):
    """Flatten/unflatten for scipy optimizer."""
    pts = x.reshape(-1, 2)
    return _neg_score(pts)

def _fermat_spiral(n, radius=1.0):
    """Generate points using Fermat's spiral (Vogel's model)."""
    i = np.arange(n)
    r = radius * np.sqrt(i / n)
    theta = i * 2.399963229728653  # Golden angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.column_stack((x, y))
    pts -= pts.mean(axis=0)
    return pts

def _regular_polygon(n, radius=1.0):
    """Generate regular polygon vertices."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y))

def _concentric_polygons():
    """Generate concentric polygon structure for 13 points.
    13 = 1 + 5 + 7 (center, pentagon, heptagon)
    or 13 = 1 + 6 + 6 (center, two hexagons)
    or 13 = 2 + 5 + 6
    """
    # Try: center + inner hexagon (6) + outer hexagon (6) = 13
    # Or: center + inner pentagon (5) + outer heptagon (7) = 13
    
    candidates = []
    
    # Option 1: center + 6 inner + 6 outer
    for r1 in np.linspace(0.3, 0.6, 8):
        for r2 in np.linspace(0.8, 1.2, 8):
            for phase1 in np.linspace(0, np.pi, 5, endpoint=False):
                for phase2 in np.linspace(0, np.pi, 5, endpoint=False):
                    pts = np.zeros((1, 2))
                    theta1 = np.linspace(0, 2*np.pi, 6, endpoint=False) + phase1
                    inner = np.column_stack((r1*np.cos(theta1), r1*np.sin(theta1)))
                    theta2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + phase2
                    outer = np.column_stack((r2*np.cos(theta2), r2*np.sin(theta2)))
                    pts = np.vstack([pts, inner, outer])
                    candidates.append(pts)
    
    # Option 2: center + 5 inner + 7 outer
    for r1 in np.linspace(0.3, 0.6, 8):
        for r2 in np.linspace(0.8, 1.2, 8):
            for phase1 in np.linspace(0, 2*np.pi/5, 5, endpoint=False):
                for phase2 in np.linspace(0, 2*np.pi/7, 5, endpoint=False):
                    pts = np.zeros((1, 2))
                    theta1 = np.linspace(0, 2*np.pi, 5, endpoint=False) + phase1
                    inner = np.column_stack((r1*np.cos(theta1), r1*np.sin(theta1)))
                    theta2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + phase2
                    outer = np.column_stack((r2*np.cos(theta2), r2*np.sin(theta2)))
                    pts = np.vstack([pts, inner, outer])
                    candidates.append(pts)
    
    # Option 3: 2 center-ish + 5 + 6
    for r_c in np.linspace(0.1, 0.3, 4):
        for r1 in np.linspace(0.4, 0.7, 6):
            for r2 in np.linspace(0.9, 1.3, 6):
                pts = np.array([[0, r_c], [0, -r_c]])
                theta1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
                inner = np.column_stack((r1*np.cos(theta1), r1*np.sin(theta1)))
                theta2 = np.linspace(0, 2*np.pi, 6, endpoint=False)
                outer = np.column_stack((r2*np.cos(theta2), r2*np.sin(theta2)))
                pts = np.vstack([pts, inner, outer])
                candidates.append(pts)
    
    return candidates

def _optimize_from_start(pts0, time_budget=30):
    """Optimize from a starting point using L-BFGS-B."""
    x0 = pts0.flatten()
    start_time = time.time()
    
    result = minimize(_eval_neg, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
    
    elapsed = time.time() - start_time
    return result.x.reshape(-1, 2), elapsed

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    best_pts = None
    best_score = -np.inf
    
    # Phase 1: Generate diverse starting points and evaluate
    starts = []
    
    # Fermat spiral
    starts.append(_fermat_spiral(N))
    
    # Regular polygon
    starts.append(_regular_polygon(N))
    
    # Random points in unit circle
    for seed in range(20):
        rng = np.random.RandomState(seed)
        r = rng.uniform(0, 1, N)
        theta = rng.uniform(0, 2*np.pi, N)
        pts = np.column_stack((r*np.cos(theta), r*np.sin(theta)))
        starts.append(pts)
    
    # Concentric polygon structures
    conc_starts = _concentric_polygons()
    # Sample a subset to keep time manageable
    if len(conc_starts) > 50:
        idx = np.random.RandomState(42).choice(len(conc_starts), 50, replace=False)
        conc_starts = [conc_starts[i] for i in idx]
    starts.extend(conc_starts)
    
    # Evaluate all starts
    for pts in starts:
        if time.time() > deadline - 60:
            break
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Phase 2: Local optimization from best starts
    # Sort by score and optimize top candidates
    scored_starts = []
    for pts in starts:
        s = _score(pts)
        scored_starts.append((s, pts))
    scored_starts.sort(key=lambda x: -x[0])
    
    # Optimize top 10 starts
    for s, pts in scored_starts[:10]:
        if time.time() > deadline - 30:
            break
        try:
            opt_pts, elapsed = _optimize_from_start(pts, time_budget=20)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Phase 3: If time remains, do differential evolution on a smaller space
    if time.time() < deadline - 10 and best_pts is not None:
        # Use a parameterized form to reduce dimensionality
        # Represent points relative to centroid, scale to unit hull area
        def _param_to_pts(params):
            # params: 26 values (13 x 2)
            pts = params.reshape(-1, 2)
            # Center
            pts -= pts.mean(axis=0)
            return pts
        
        # DE bounds: points in [-2, 2] x [-2, 2]
        bounds = [(-2, 2)] * (2 * N)
        
        try:
            de_result = differential_evolution(
                _eval_neg, bounds, 
                maxiter=50, popsize=15, 
                tol=1e-6, seed=42,
                update_interval=1,
                polish=True
            )
            de_pts = de_result.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # Final: ensure we return something valid
    if best_pts is None:
        best_pts = _fermat_spiral(N)
    
    return best_pts

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
