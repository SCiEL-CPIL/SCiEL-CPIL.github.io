import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 13
DEADLINE = 115.0  # Safety margin before 120s kill

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

def _compute_areas(pts):
    """Compute all triangle areas for a point set."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-12

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _compute_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _neg_score(params):
    """Negative score for minimization."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _make_symmetric_inits():
    """
    Generate diverse initial configurations based on symmetry.
    Optimal Heilbronn configurations often have points on concentric regular polygons.
    """
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    p1 = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(p1.flatten())
    
    # 2. 1 center + 12 outer (Regular 12-gon)
    p2 = np.zeros((N, 2))
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    p2[1:, 0] = np.cos(angles)
    p2[1:, 1] = np.sin(angles)
    inits.append(p2.flatten())
    
    # 3. 1 center + 4 inner (Square) + 8 outer (Octagon)
    p3 = np.zeros((N, 2))
    # 4 inner
    angles_i = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    p3[1:5, 0] = 0.5 * np.cos(angles_i)
    p3[1:5, 1] = 0.5 * np.sin(angles_i)
    # 8 outer
    angles_o = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    p3[5:, 0] = np.cos(angles_o)
    p3[5:, 1] = np.sin(angles_o)
    inits.append(p3.flatten())
    
    # 4. 1 center + 6 inner (Hexagon) + 6 outer (Hexagon)
    p4 = np.zeros((N, 2))
    # 6 inner
    angles_i = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    p4[1:7, 0] = 0.5 * np.cos(angles_i)
    p4[1:7, 1] = 0.5 * np.sin(angles_i)
    # 6 outer
    angles_o = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6
    p4[7:, 0] = np.cos(angles_o)
    p4[7:, 1] = np.sin(angles_o)
    inits.append(p4.flatten())
    
    # 5. 1 center + 3 inner (Triangle) + 9 outer (Nonagon)
    p5 = np.zeros((N, 2))
    angles_i = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    p5[1:4, 0] = 0.5 * np.cos(angles_i)
    p5[1:4, 1] = 0.5 * np.sin(angles_i)
    angles_o = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    p5[4:, 0] = np.cos(angles_o)
    p5[4:, 1] = np.sin(angles_o)
    inits.append(p5.flatten())
    
    # 6. Random uniform
    rng = np.random.RandomState(42)
    p6 = rng.uniform(-1, 1, size=(N, 2)).flatten()
    inits.append(p6)
    
    # 7. Random symmetric perturbation of regular 13-gon
    p7 = p1.copy()
    p7 += rng.normal(0, 0.1, size=p7.shape)
    inits.append(p7.flatten())
    
    # 8. Random symmetric perturbation of 1+12
    p8 = p2.copy()
    p8 += rng.normal(0, 0.1, size=p8.shape)
    inits.append(p8.flatten())
    
    return inits

def construct():
    t_start = time.time()
    
    best_params = None
    best_score = -np.inf
    
    # 1. Evaluate initial guesses
    inits = _make_symmetric_inits()
    for p in inits:
        pts = p.reshape(N, 2)
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = p.copy()
            
    # 2. Multi-start Nelder-Mead optimization
    # We will try to refine each initial guess using Nelder-Mead.
    # Nelder-Mead is derivative-free and handles non-differentiable functions well.
    
    # Sort inits by score to try the best ones first
    init_scores = []
    for p in inits:
        s = _score_points(p.reshape(N, 2))
        init_scores.append((s, p))
    init_scores.sort(key=lambda x: x[0], reverse=True)
    
    for s, p in init_scores:
        if (time.time() - t_start) > (DEADLINE - 10.0):
            break
            
        # Run Nelder-Mead from this start
        try:
            res = minimize(
                _neg_score,
                p,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            # Evaluate the result
            res_score = _score_points(res.x.reshape(N, 2))
            if res_score > best_score:
                best_score = res_score
                best_params = res.x.copy()
                
        except Exception:
            pass
            
    # 3. Final local refinement with higher precision
    # Try to polish the best result found so far
    if best_params is not None:
        if (time.time() - t_start) < (DEADLINE - 5.0):
            try:
                res = minimize(
                    _neg_score,
                    best_params,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                res_score = _score_points(res.x.reshape(N, 2))
                if res_score > best_score:
                    best_score = res_score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    return best_params.reshape(N, 2)
