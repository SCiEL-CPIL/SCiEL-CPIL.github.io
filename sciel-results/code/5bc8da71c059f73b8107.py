"""Combined program: polar dual (support function) parametrization + differential evolution + soft-min L-BFGS-B refinement."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

# Precompute all triangle index triplets for vectorized area computation
_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]
_I, _J, _K = _TRIPLETS.T

# Fixed angular normals for the polar dual representation
_ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False)
_NORMALS = np.column_stack([np.cos(_ANGLES), np.sin(_ANGLES)])

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    a = pts[_I]
    b = pts[_J]
    c = pts[_K]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                          (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        if len(pts) < 3:
            return 0.0
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return float(np.min(areas)) / hull_area

def _soft_min_objective(x, beta=50.0):
    """
    Soft-min of triangle areas / hull area, negative for minimization.
    x is a flat array of length 2*N.
    """
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    # Soft min using log-sum-exp trick for numerical stability
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _exact_objective(x):
    """Exact negative score for Nelder-Mead refinement."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _polytope_from_support(log_dists):
    """
    Reconstruct the convex polygon from positive support distances.
    log_dists: array of length N, log of the distance from origin to each support line.
    The polygon is the intersection of half-planes: n_i . x <= d_i for i=0..N-1.
    Returns array of shape (M, 2) where M is the number of vertices.
    """
    d = np.exp(log_dists)
    # Each edge i is defined by normal n_i and distance d_i from origin.
    # The edge line: n_i . x = d_i
    # Vertex i is the intersection of edges i and (i+1) mod N.
    verts = np.zeros((N, 2))
    for i in range(N):
        n1 = _NORMALS[i]
        n2 = _NORMALS[(i + 1) % N]
        d1 = d[i]
        d2 = d[(i + 1) % N]
        # Solve: n1 . x = d1, n2 . x = d2
        A = np.array([n1, n2])  # 2x2
        b = np.array([d1, d2])
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        if abs(det) < 1e-15:
            # Degenerate: use midpoint fallback
            verts[i] = 0.5 * (d1 * n1 + d2 * n2)
        else:
            verts[i, 0] = (d1 * A[1, 1] - d2 * A[0, 1]) / det
            verts[i, 1] = (A[0, 0] * d2 - A[1, 0] * d1) / det
    return verts

def _support_objective(log_dists, beta=50.0):
    """
    Negative soft-min score for the polar dual parametrization.
    log_dists: array of length N.
    """
    pts = _polytope_from_support(log_dists)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 1e-10:
        return 1e6
    
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    
    neg_beta_areas = -beta * areas
    max_val = np.max(neg_beta_areas)
    log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
    soft_min_area = -log_sum_exp / beta
    
    return -soft_min_area / hull_area

def _support_exact_objective(log_dists):
    """Exact negative score for the polar dual parametrization."""
    pts = _polytope_from_support(log_dists)
    return -_score(pts)

def _initial_log_dists(seed=0):
    """Generate initial log support distances: regular polygon (all equal)."""
    # Regular N-gon inscribed in unit circle has inradius cos(pi/N)
    inradius = np.cos(np.pi / N)
    return np.full(N, np.log(inradius))

def _random_log_dists(seed=0, scale=0.5):
    """Random perturbation of log support distances."""
    rng = np.random.RandomState(seed)
    base = np.cos(np.pi / N)
    log_base = np.log(base)
    return log_base + rng.normal(0, scale, N)

def _halton_sequence(n, base):
    """Generate first n Halton sequence values in [0,1)."""
    seq = np.zeros(n)
    f = 1.0
    for i in range(1, n + 1):
        f /= base
        seq[i - 1] = 0.0
        j = i
        while j > 0:
            seq[i - 1] += (j % base) * f
            j //= base
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    best_score = 0.0
    best_pts = _polytope_from_support(_initial_log_dists(0))
    
    # Phase 1: Differential evolution on the polar dual parametrization
    # This structurally enforces convexity
    de_bounds = [(np.log(0.1), np.log(2.0))] * N
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _support_objective,
                bounds=de_bounds,
                args=(50.0,),
                seed=42,
                maxiter=50,
                popsize=15,
                tol=1e-8,
                mut=(0.5, 1.0),
                recomb=0.7,
                polish=True,
                update_interval=1,
                callback=None,
                init='sobol'
            )
            
            if de_result.success or de_result.fun < 1e5:
                de_pts = _polytope_from_support(de_result.x)
                de_score = _score(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_pts = de_pts.copy()
                    
        except Exception:
            pass
    
    # Phase 2: Multi-start L-BFGS-B refinement on the Cartesian parametrization
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Regular 13-gon
    angles_reg = np.linspace(0, 2 * np.pi, N, endpoint=False)
    candidates.append(np.column_stack([np.cos(angles_reg), np.sin(angles_reg)]))
    
    # 2. Current best
    candidates.append(best_pts.copy())
    
    # 3. Concentric rings (various splits)
    for n_outer in [7, 8, 9, 10]:
        n_inner = N - n_outer
        if n_inner >= 3:
            for r_inner in [0.4, 0.5, 0.6]:
                angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
                angles_inner = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
                outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
                inner = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
                candidates.append(np.vstack([outer, inner]))
    
    # 4. Perturbed regular polygon
    rng = np.random.default_rng(42)
    for i in range(10):
        noise = rng.normal(0, 0.05, (N, 2))
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        base = np.column_stack([np.cos(angles), np.sin(angles)])
        candidates.append(base + noise)
    
    # 5. Random points on/near circle
    for i in range(10):
        angles = np.sort(rng.uniform(0, 2 * np.pi, N))
        radii = rng.uniform(0.7, 1.0, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        candidates.append(pts)
    
    # Evaluate all candidates with exact score
    scored_candidates = []
    for pts in candidates:
        if time.time() > deadline:
            break
        s = _score(pts)
        scored_candidates.append((s, pts))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    scored_candidates.sort(key=lambda x: -x[0])
    top_candidates = [pts for _, pts in scored_candidates[:min(6, len(scored_candidates))]]
    
    # Progressive beta: start smooth, get sharper
    betas = [30.0, 80.0, 150.0]
    
    for beta in betas:
        if time.time() > deadline - 25:
            break
        
        for init_pts in top_candidates:
            if time.time() > deadline - 20:
                break
            
            try:
                res = minimize(
                    _soft_min_objective,
                    init_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
                )
                
                res_pts = res.x.reshape(N, 2)
                res_score = _score(res_pts)
                
                if res_score > best_score:
                    best_score = res_score
                    best_pts = res_pts.copy()
                    
            except Exception:
                pass
        
        # Update top candidates with current best
        if best_pts is not None:
            top_candidates = [best_pts.copy()]
    
    # Phase 3: Final high-precision polish with Nelder-Mead
    if best_pts is not None and time.time() < deadline - 5:
        try:
            res = minimize(
                _exact_objective,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 5000}
            )
            res_pts = res.x.reshape(N, 2)
            res_score = _score(res_pts)
            
            if res_score > best_score:
                best_score = res_score
                best_pts = res_pts.copy()
                
        except Exception:
            pass
    
    return best_pts
# EVOLVE-BLOCK-END
