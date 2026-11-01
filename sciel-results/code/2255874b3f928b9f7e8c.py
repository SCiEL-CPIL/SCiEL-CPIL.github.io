"""Optimized covering of the sphere with 50 points using soft-max surrogate + L-BFGS + Lloyd polish."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms


def _fib_sample(n, rng=None):
    """Fibonacci spiral sampling on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def _random_sample(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    return _normalize(pts)


def _compute_hull_data(pts):
    """
    Compute convex hull and per-facet circumcenter + max angle to vertices.
    Returns (hull, facet_ccs, facet_max_angs, facet_vertices_idx).
    facet_ccs: (F,3) array of circumcenters on sphere
    facet_max_angs: (F,) array of max angular distance from cc to facet vertices
    facet_vertices_idx: (F,3) integer array of vertex indices
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None, None

    facets = hull.simplices  # (F, 3)
    F = len(facets)
    facet_ccs = np.zeros((F, 3))
    facet_max_angs = np.zeros(F)
    valid = np.zeros(F, dtype=bool)

    for i in range(F):
        tri = facets[i]
        a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
        bc = np.cross(b, c)
        ca = np.cross(c, a)
        ab = np.cross(a, b)
        denom = 2.0 * np.dot(a, bc)
        if abs(denom) < 1e-12:
            continue
        cc = (bc + ca + ab) / denom
        cc_norm = np.linalg.norm(cc)
        if cc_norm < 1e-12:
            continue
        cc = cc / cc_norm
        facet_ccs[i] = cc
        # Max angle from cc to the three vertices
        dots = np.clip(np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)]), -1.0, 1.0)
        angs = np.arccos(dots)
        facet_max_angs[i] = np.max(angs)
        valid[i] = True

    return hull, facet_ccs, facet_max_angs, facets, valid


def _covering_radius_deg_hull(pts):
    """Exact covering radius in degrees via convex hull facet circumradii."""
    _, _, max_angs, _, valid = _compute_hull_data(pts)
    if max_angs is None:
        return 180.0
    if not np.any(valid):
        return 180.0
    return np.degrees(np.max(max_angs[valid]))


def _soft_max_objective(pts_flat, beta):
    """
    Soft-max surrogate for the covering radius.
    pts_flat: (150,) flattened array of 50 unit vectors.
    Returns (objective_value, gradient_flat).
    Uses log-sum-exp over facet circumradii with temperature 1/beta.
    """
    pts = _normalize(pts_flat.reshape(N, 3))
    hull, facet_ccs, facet_max_angs, facets, valid = _compute_hull_data(pts)
    if hull is None:
        return np.pi, np.zeros_like(pts_flat)

    if not np.any(valid):
        return np.pi, np.zeros_like(pts_flat)

    angs = facet_max_angs[valid]
    # Soft max via log-sum-exp: t * log(sum(exp(angs/t))) with t=1/beta
    # For large beta, this approaches max(angs)
    t = 1.0 / beta
    shifted = angs - np.max(angs)
    log_sum_exp = np.max(angs) + t * np.log(np.sum(np.exp(shifted / t)))
    obj = log_sum_exp

    # Gradient: d(obj)/d(angs_i) = exp(angs_i / t) / sum(exp(angs_j / t))
    weights = np.exp(shifted / t)
    weights = weights / weights.sum()  # (F_valid,)

    # Now compute gradient with respect to pts
    # For each valid facet, the circumcenter cc depends on the three vertices a,b,c.
    # d(cc)/d(v) for v in {a,b,c}:
    # cc = (|a|^2 b×c + |b|^2 c×a + |c|^2 a×b) / (2 a·(b×c))
    # Since |a|=|b|=|c|=1:
    # cc = (b×c + c×a + a×b) / (2 a·(b×c))
    # Let D = 2 a·(b×c), N = b×c + c×a + a×b
    # cc = N/D
    # d(cc)/d(a) = [dN/da * D - N * dD/da] / D^2
    # dN/da = d(b×c)/da + d(c×a)/da + d(a×b)/da = 0 + c×1 + 1×b = c + b  (treating a as variable)
    # Wait, need to be more careful. N = b×c + c×a + a×b
    # dN/da = 0 + c + b (derivative of c×a w.r.t. a is c, derivative of a×b w.r.t. a is b)
    # Actually d(c×a)/da = c (since d(x×y)/dx = -y×... no. d(a×b)/da = b, d(c×a)/da = ?
    # (c×a) derivative w.r.t. a: using d(u×v)/dv = -u×... hmm.
    # Let me use: d(a×b)/da = b, d(a×b)/db = -a
    # d(c×a)/da = ?  c×a, derivative w.r.t. a: the formula is d(x×y)/dx = -y×(d... 
    # Actually, for fixed c: d(c×a)/da = -a×c^T... no.
    # c×a is a vector. Its j-th component is eps_jkl c_k a_l. d/da_m = eps_jml c_m.
    # This equals (c × I) acting... Actually d(c×a)/da_m = c × e_m (cross product of c with unit vector e_m).
    # So d(c×a)/da as a matrix: the column m is c×e_m. This is -[c]_x (skew-symmetric cross product matrix of c, transposed).
    # Similarly d(a×b)/da_m = b × e_m, so the matrix is -[b]_x.
    # d(b×c)/da = 0.
    # So dN/da = -[c]_x - [b]_x = -([b]+[c])_x  ... wait, [b]_x + [c]_x = [b+c]_x, so dN/da = -[b+c]_x.
    # Hmm, let me just use the formula directly.
    # dN/da: N = b×c + c×a + a×b
    # d(b×c)/da = 0
    # d(c×a)/da: for each component, d/da (c×a) = c × (da/d...) no.
    # Let me just compute numerically or use a different approach.
    
    # Alternative: use the fact that cc is the circumcenter. The gradient of the angle
    # from cc to vertex v can be computed via the chain rule.
    # angle = arccos(cc · v), d(angle)/d(cc) = -v / sqrt(1-(cc·v)^2)
    # d(angle)/d(v) = -cc / sqrt(1-(cc·v)^2)
    
    # And d(cc)/d(pts) can be computed analytically. Let me derive it properly.
    # cc = N / D where N = b×c + c×a + a×b, D = 2 a·(b×c)
    # d(cc)/d(a) = (dN/da * D - N * dD/da) / D^2
    # dN/da = d(c×a)/da + d(a×b)/da
    # d(c×a)/da: the derivative of c×a with respect to a. 
    # (c×a)_i = eps_ijk c_j a_k, d/da_l = eps_ijl c_j = (c × e_l)_i ... 
    # Actually (c × v)_i = eps_ijk c_j v_k, so d(c×a)/da_l has components eps_ijl c_j.
    # This is the l-th column of the matrix representing cross product with c from the left... 
    # It's -[c]_x where [c]_x is the skew-symmetric matrix such that [c]_x v = c × v.
    # Because [c]_x v = c×v, and (c×a)_i = ([c]_x a)_i, so d(c×a)/da = [c]_x.
    # Wait: [c]_x a = c × a. So d(c×a)/da = [c]_x (as a matrix acting on da).
    # Similarly d(a×b)/da: [a]_x b = a×b, so d(a×b)/da = ? 
    # (a×b)_i = eps_ijk a_j b_k, d/da_l = eps_ijl b_j = ([b]_x^T ... ) 
    # [b]_x a = b×a = -(a×b). So d(a×b)/da = -[b]_x.
    # Therefore dN/da = [c]_x - [b]_x.
    
    # dD/da = 2 d(a·(b×c))/da = 2 (b×c)
    
    # So d(cc)/d(a) = (([c]_x - [b]_x) * D - N * 2(b×c)) / D^2
    
    # This is getting complex. Let me use a simpler approach: 
    # compute the gradient numerically per facet, or use a known formula.
    
    # Actually, for the purpose of this optimization, let me use a simpler gradient:
    # The key insight is that to reduce the circumradius of a facet, we should move
    # the vertices toward the circumcenter. The tangent direction from vertex v to cc
    # is: t_v = cc - (cc·v) v, normalized.
    
    # For the soft-max, the gradient w.r.t. each vertex is a weighted sum over facets
    # containing that vertex.
    
    grad = np.zeros((N, 3))
    valid_indices = np.where(valid)[0]
    for idx_in_valid, fi in enumerate(valid_indices):
        w = weights[idx_in_valid]
        if w < 1e-15:
            continue
        tri = facets[fi]
        cc = facet_ccs[fi]
        for vi in tri:
            v = pts[vi]
            # Tangent direction from v toward cc
            proj = np.dot(cc, v) * v
            tangent = cc - proj
            tn = np.linalg.norm(tangent)
            if tn > 1e-12:
                grad[vi] += w * tangent / tn
    
    # Project gradient to tangent space (remove radial component)
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return obj, grad.flatten()


def _lloyd_polish(pts, max_iters=200, deadline=None):
    """
    Lloyd's algorithm on the sphere: move each point to the centroid of its
    Voronoi cell (spherical centroid = normalized sum of cell vertices weighted by area).
    Approximation: move to normalized circumcenter of the largest adjacent facet.
    """
    pts = _normalize(pts)
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        hull, facet_ccs, facet_max_angs, facets, valid = _compute_hull_data(pts)
        if hull is None:
            break
        
        new_pts = pts.copy()
        # For each point, compute the average of circumcenters of adjacent facets
        # weighted by the solid angle (or just uniform for simplicity)
        # Simple approach: move each point toward the centroid of its cell
        # Cell centroid approx: average of facet circumcenters for facets containing that point
        cell_sum = np.zeros((N, 3))
        cell_count = np.zeros(N)
        
        for i in range(len(facets)):
            if not valid[i]:
                continue
            cc = facet_ccs[i]
            for vi in facets[i]:
                cell_sum[vi] += cc
                cell_count[vi] += 1
        
        for i in range(N):
            if cell_count[i] > 0:
                target = cell_sum[i] / cell_count[i]
                tn = np.linalg.norm(target)
                if tn > 1e-12:
                    new_pts[i] = target / tn
        
        new_pts = _normalize(new_pts)
        # Check for convergence
        if np.max(np.linalg.norm(new_pts - pts, axis=1)) < 1e-10:
            pts = new_pts
            break
        pts = new_pts
    
    return pts


def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_pts = None
    best_score = np.pi  # worst case: 180 degrees in radians
    
    def _update_best(pts):
        nonlocal best_pts, best_score
        score = _covering_radius_deg_hull(pts)
        if score < np.degrees(best_score):
            best_score = np.radians(score)
            best_pts = _normalize(pts)
    
    # Phase 1: L-BFGS with progressive soft-max sharpening from Fibonacci start
    pts = _fib_sample(N)
    _update_best(pts)
    
    # Progressive beta values: start smooth, end sharp
    beta_schedule = [10, 30, 100, 300, 1000, 3000]
    for beta in beta_schedule:
        if time.time() > deadline - 10:
            break
        
        def _obj_grad(x):
            val, grad = _soft_max_objective(x, beta)
            return val, grad
        
        result = minimize(
            _obj_grad,
            pts.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        pts = _normalize(result.x.reshape(N, 3))
        _update_best(pts)
    
    # Phase 2: Lloyd polish
    pts = _lloyd_polish(pts, max_iters=100, deadline=deadline - 10)
    _update_best(pts)
    
    # Phase 3: More L-BFGS at high beta from multiple starts
    starts = [_fib_sample(N)]
    for s in range(3):
        starts.append(_random_sample(N, seed=42 + s * 137))
    
    for start_pts in starts:
        if time.time() > deadline - 10:
            break
        
        for beta in [100, 500, 2000]:
            if time.time() > deadline - 10:
                break
            
            def _obj_grad(x, b=beta):
                val, grad = _soft_max_objective(x, b)
                return val, grad
            
            result = minimize(
                _obj_grad,
                start_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 400}
            )
            pts = _normalize(result.x.reshape(N, 3))
            pts = _lloyd_polish(pts, max_iters=50, deadline=deadline - 10)
            _update_best(pts)
    
    if best_pts is None:
        best_pts = _fib_sample(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
