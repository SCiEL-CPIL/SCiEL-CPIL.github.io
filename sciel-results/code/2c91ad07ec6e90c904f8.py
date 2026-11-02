"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 140.0  # safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)

_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized computation of all triangle areas
    # p_i, p_j, p_k
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    # Cross product (pi->pj) x (pi->pk)
    # (pj - pi) x (pk - pi) = (pj_x - pi_x)*(pk_y - pi_y) - (pj_y - pi_y)*(pk_x - pi_x)
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _make_two_layer(inner_n, outer_n, r_inner=0.5, r_outer=1.0):
    """
    Create a two-layer configuration.
    inner_n points on a circle of radius r_inner.
    outer_n points on a circle of radius r_outer.
    Total points must be N.
    """
    if inner_n + outer_n != N:
        raise ValueError("inner_n + outer_n must equal N")
    
    pts = []
    # Inner ring
    if inner_n > 0:
        angles_in = np.linspace(0, 2 * np.pi, inner_n, endpoint=False)
        # Offset inner ring by half step to interleave with outer? 
        # Or just start at 0. Let's start at 0 for simplicity, 
        # but maybe offsetting helps symmetry breaking? 
        # For N=13, 1+12 or 7+6 or 1+6+6 (3 layers) are common.
        # Let's stick to 2 layers for now, or 1 center + 12 outer.
        x_in = r_inner * np.cos(angles_in)
        y_in = r_inner * np.sin(angles_in)
        pts.append(np.column_stack([x_in, y_in]))
        
    # Outer ring
    if outer_n > 0:
        angles_out = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_out = r_outer * np.cos(angles_out)
        y_out = r_outer * np.sin(angles_out)
        pts.append(np.column_stack([x_out, y_out]))
        
    return np.vstack(pts)

def _make_three_layer(center_n, mid_n, outer_n, r_mid=0.5, r_outer=1.0):
    """
    Create a three-layer configuration.
    center_n points at center (or small ring).
    mid_n points on circle radius r_mid.
    outer_n points on circle radius r_outer.
    """
    if center_n + mid_n + outer_n != N:
        raise ValueError("center_n + mid_n + outer_n must equal N")
        
    pts = []
    # Center
    if center_n == 1:
        pts.append(np.array([[0.0, 0.0]]))
    elif center_n > 1:
        # Small ring
        angles_c = np.linspace(0, 2 * np.pi, center_n, endpoint=False)
        r_c = 0.1
        x_c = r_c * np.cos(angles_c)
        y_c = r_c * np.sin(angles_c)
        pts.append(np.column_stack([x_c, y_c]))
        
    # Mid ring
    if mid_n > 0:
        angles_m = np.linspace(0, 2 * np.pi, mid_n, endpoint=False)
        x_m = r_mid * np.cos(angles_m)
        y_m = r_mid * np.sin(angles_m)
        pts.append(np.column_stack([x_m, y_m]))
        
    # Outer ring
    if outer_n > 0:
        angles_o = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_o = r_outer * np.cos(angles_o)
        y_o = r_outer * np.sin(angles_o)
        pts.append(np.column_stack([x_o, y_o]))
        
    return np.vstack(pts)

def construct():
    t_start = time.time()
    
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -np.inf
    
    # List of initializations to try
    # 1. Regular 13-gon
    # 2. 1 center + 12 outer
    # 3. 1 center + 6 mid + 6 outer
    # 4. 7 mid + 6 outer (no center)
    # 5. Random
    
    inits = []
    
    # 1. Regular 13-gon
    p1 = _regular_polygon(N)
    inits.append(p1)
    
    # 2. 1 center + 12 outer
    p2 = _make_three_layer(1, 0, 12)
    inits.append(p2)
    
    # 3. 1 center + 6 mid + 6 outer
    p3 = _make_three_layer(1, 6, 6)
    inits.append(p3)
    
    # 4. 7 mid + 6 outer
    p4 = _make_two_layer(7, 6)
    inits.append(p4)
    
    # 5. Random uniform
    rng = np.random.RandomState(42)
    p5 = rng.uniform(-1, 1, size=(N, 2))
    inits.append(p5)
    
    # Evaluate initial scores
    for p in inits:
        s = _score_points(p)
        if s > best_score:
            best_score = s
            best_params = p.flatten().copy()
            
    # Optimization Strategy:
    # 1. Differential Evolution (Global)
    # 2. Local Refinement (L-BFGS-B) from best DE and best Inits
    
    def time_check():
        return (time.time() - t_start) > (DEADLINE - 10.0)
        
    # Phase 1: Differential Evolution
    # We use a larger population and more iterations because scoring is fast.
    try:
        result = differential_evolution(
            _neg_score,
            bounds,
            seed=42,
            maxiter=50,  # Increased from 30
            popsize=20,  # Increased from 15
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.8,
            polish=True,
            init='sobol',
            callback=None,
            updating='deferred',
            workers=1
        )
        
        if result.fun < -best_score:
            candidate = result.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = result.x.copy()
    except Exception:
        pass
        
    # Phase 2: Local Refinement from multiple starts
    # We try to polish the best DE result and the best initial guesses.
    # This helps escape shallow local minima that DE might land in.
    
    starts_to_refine = [best_params]
    for p in inits:
        if p.flatten().shape == best_params.shape:
            starts_to_refine.append(p.flatten())
            
    # Deduplicate starts roughly
    unique_starts = []
    for s in starts_to_refine:
        is_unique = True
        for u in unique_starts:
            if np.allclose(s, u, atol=1e-3):
                is_unique = False
                break
        if is_unique:
            unique_starts.append(s)
            
    for start in unique_starts:
        if time_check():
            break
            
        try:
            res2 = minimize(
                _neg_score,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'maxfun': 20000}
            )
            score2 = _score_points(res2.x.reshape(N, 2))
            if score2 > best_score:
                best_score = score2
                best_params = res2.x.copy()
        except Exception:
            pass
            
    # Phase 3: If time remains, try a few random restarts around the best solution
    # to see if we can jump to a better basin.
    attempts = 0
    while (time.time() - t_start) < (DEADLINE - 5.0) and attempts < 10:
        attempts += 1
        # Random perturbation around best
        # Scale perturbation by the "size" of the current configuration
        scale = np.std(best_params.reshape(N, 2))
        if scale < 1e-3: scale = 0.1
        perturbation = rng.normal(0, scale * 0.1, size=(2 * N))
        start = best_params + perturbation
        
        # Project back to bounds roughly
        start = np.clip(start, -1.0, 1.0)
        
        try:
            res4 = minimize(
                _neg_score,
                start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 10000}
            )
            score4 = _score_points(res4.x.reshape(N, 2))
            if score4 > best_score:
                best_score = score4
                best_params = res4.x.copy()
        except Exception:
            pass
            
    return best_params.reshape(N, 2)
