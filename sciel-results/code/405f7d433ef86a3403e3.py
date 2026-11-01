"""Optimized covering of the sphere with 50 points using spherical Voronoi + gradient descent."""
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


def _facet_circumdata(pts, tri):
    """Compute circumcenter (on sphere) and max angle to vertices for a hull facet."""
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    bc = np.cross(b, c)
    ca = np.cross(c, a)
    ab = np.cross(a, b)
    denom = 2.0 * np.dot(a, bc)
    if abs(denom) < 1e-12:
        return None, 0.0
    cc = (bc + ca + ab) / denom
    cc_norm = np.linalg.norm(cc)
    if cc_norm < 1e-12:
        return None, 0.0
    cc = cc / cc_norm
    # Max angle from cc to vertices
    max_ang = 0.0
    for v in (a, b, c):
        dot = np.clip(np.dot(cc, v), -1.0, 1.0)
        ang = np.arccos(dot)
        if ang > max_ang:
            max_ang = ang
    return cc, max_ang


def _covering_radius(pts):
    """Compute the exact covering radius via convex hull facet circumradii."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi

    max_ang = 0.0
    for tri in hull.simplices:
        _, ang = _facet_circumdata(pts, tri)
        if ang > max_ang:
            max_ang = ang
    return max_ang


def _soft_max_radius_and_grad(pts, beta=20.0):
    """
    Smooth surrogate: soft-max of facet circumradii using log-sum-exp.
    Returns (smooth_objective, gradient w.r.t. pts).
    
    For each facet f with circumcenter cc_f and max angle a_f,
    we compute the soft-max: S = (1/beta) * log(sum_f exp(beta * a_f))
    Gradient: dS/d(pts) = (1/beta) * sum_f exp(beta*a_f) / sum_f exp(beta*a_f) * d(a_f)/d(pts)
    
    d(a_f)/d(pts) for vertex v of facet f: 
      a_f = arccos(cc_f . v) for the farthest vertex
      da_f/dv = -cc_f / sqrt(1 - (cc_f.v)^2)  (tangent component)
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, np.zeros_like(pts)

    facets = hull.simplices
    n_facets = len(facets)
    
    # Compute all facet data
    ccs = []
    max_angs = []
    facet_vertices = []  # list of (vertex_idx, vertex_array) for the farthest vertex per facet
    
    for tri in facets:
        cc, max_ang = _facet_circumdata(pts, tri)
        if cc is None:
            ccs.append(None)
            max_angs.append(0.0)
            facet_vertices.append(None)
        else:
            ccs.append(cc)
            max_angs.append(max_ang)
            # Find which vertex gives the max angle
            best_v_idx = tri[0]
            best_dot = -2.0
            for vi in tri:
                v = pts[vi]
                dot = np.dot(cc, v)
                if dot < best_dot:
                    best_dot = dot
                    best_v_idx = vi
            facet_vertices.append((best_v_idx, pts[best_v_idx]))

    # Log-sum-exp for numerical stability
    max_a = max(max_angs) if max_angs else 0.0
    if max_a == 0.0:
        return 0.0, np.zeros_like(pts)
    
    shifted = np.array([max_a - a for a in max_angs])
    exps = np.exp(beta * shifted)
    log_sum = np.log(np.sum(exps))
    soft_obj = max_a + log_sum / beta
    
    # Weights for each facet
    weights = exps / np.sum(exps)
    
    # Compute gradient
    grad = np.zeros_like(pts)
    for i, tri in enumerate(facets):
        w = weights[i]
        if w < 1e-10:
            continue
        cc = ccs[i]
        if cc is None:
            continue
        fv = facet_vertices[i]
        if fv is None:
            continue
        v_idx, v = fv
        dot = np.dot(cc, v)
        dot = np.clip(dot, -1.0 + 1e-10, 1.0 - 1e-10)
        sqrt_term = np.sqrt(1.0 - dot * dot)
        if sqrt_term < 1e-10:
            continue
        # d(a_f)/d(v) = -cc / sqrt(1 - dot^2), projected to tangent space
        # The gradient contribution to pts[v_idx] is:
        # -w * (1/beta) * beta * (-cc / sqrt_term) = w * cc / sqrt_term
        # But we need to project to tangent space and account for normalization
        tangent_contrib = cc / sqrt_term
        # Project to tangent space at v
        tangent_contrib = tangent_contrib - np.dot(tangent_contrib, v) * v
        grad[v_idx] += w * tangent_contrib

    return soft_obj, grad


def _random_initialization(seed=None):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(N, 3))
    return _normalize(pts)


def _fibonacci_initialization():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))


def _gradient_descent(pts, deadline, lr=0.05, max_iters=2000, beta=20.0):
    """Run gradient descent with smooth surrogate objective."""
    best_pts = pts.copy()
    best_score = _covering_radius(pts)
    current_lr = lr
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        score, grad = _soft_max_radius_and_grad(pts, beta)
        if score < best_score:
            # Also check exact score occasionally
            exact_score = _covering_radius(pts)
            if exact_score < best_score:
                best_score = exact_score
                best_pts = pts.copy()
            else:
                best_score = min(best_score, score)
                best_pts = pts.copy()
        # Gradient step on the sphere
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        pts = pts + current_lr * tangent_grad
        pts = _normalize(pts)
        # Adaptive learning rate
        if it % 50 == 0 and it > 0:
            current_lr *= 0.95
            beta = min(beta * 1.05, 100.0)  # Increase sharpness over time

    # Final exact check
    final_score = _covering_radius(pts)
    if final_score < best_score:
        best_score = final_score
        best_pts = pts.copy()
    return best_pts, best_score


def _local_polish(pts, deadline):
    """High-precision polish using L-BFGS-B on the smooth surrogate."""
    best_pts = pts.copy()
    best_score = _covering_radius(pts)
    
    def objective(x):
        p = _normalize(x.reshape(N, 3))
        score, _ = _soft_max_radius_and_grad(p, beta=50.0)
        return score
    
    def gradient(x):
        p = _normalize(x.reshape(N, 3))
        _, grad = _soft_max_radius_and_grad(p, beta=50.0)
        # Project to tangent space
        tangent_grad = grad - np.sum(grad * p, axis=1, keepdims=True) * p
        return tangent_grad.ravel()
    
    x0 = pts.ravel()
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500}
        )
        if result.success or result.fun < best_score:
            p = _normalize(result.x.reshape(N, 3))
            exact_score = _covering_radius(p)
            if exact_score < best_score:
                best_score = exact_score
                best_pts = p
    except Exception:
        pass
    
    return best_pts, best_score


def construct():
    deadline = time.time() + 115  # safety margin before 120s limit
    best_pts = None
    best_score = np.pi

    # Phase 1: Fibonacci spiral initialization + gradient descent
    pts = _fibonacci_initialization()
    best_pts, best_score = _gradient_descent(pts, deadline, lr=0.05, max_iters=1500)

    # Phase 2: Random restarts with gradient descent
    rng = np.random.default_rng(42)
    n_restarts = 0
    while time.time() < deadline - 30 and n_restarts < 10:
        seed = rng.integers(0, 2**31)
        pts = _random_initialization(seed)
        cand_pts, cand_score = _gradient_descent(pts, deadline, lr=0.03, max_iters=300)
        if cand_score < best_score:
            best_score = cand_score
            best_pts = cand_pts
        n_restarts += 1

    # Phase 3: Local polish on the best candidate
    if time.time() < deadline - 10:
        best_pts, best_score = _local_polish(best_pts, deadline)

    # Final verification
    final_score = _covering_radius(best_pts)
    if final_score < best_score:
        best_score = final_score

    return _normalize(best_pts)
# EVOLVE-BLOCK-END
