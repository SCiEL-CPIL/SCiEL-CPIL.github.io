"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_DEADLINE = time.time() + 143.0  # explicit wall-clock guard (150s kill, 7s safety margin)

def _time_ok(margin=0.0):
    return time.time() < _DEADLINE - margin

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        if m < 2:
            continue
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            local_min = 0.5 * np.min(np.abs(crosses))
            if local_min < min_area:
                min_area = local_min
    return min_area

def _all_triangle_areas(pts):
    """Return array of all triangle areas (n choose 3)."""
    n = len(pts)
    areas = []
    for i in range(n - 2):
        vi = pts[i]
        rest = pts[i + 1:]
        m = len(rest)
        if m < 2:
            continue
        for j in range(m - 1):
            vj = rest[j] - vi
            vk = rest[j + 1:] - vi
            crosses = vj[0] * vk[:, 1] - vj[1] * vk[:, 0]
            areas.append(0.5 * np.abs(crosses))
    if not areas:
        return np.array([])
    return np.concatenate(areas)

def _hull_area(pts):
    try:
        return ConvexHull(pts).volume
    except Exception:
        return 0.0

def _score(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=60.0):
    """Smooth approximation of the minimum triangle area via log-sum-exp."""
    n = len(pts)
    if n < 3:
        return 0.0
    a = _all_triangle_areas(pts)
    if len(a) == 0:
        return 0.0
    a = a[a > 1e-12]
    if len(a) == 0:
        return 0.0
    amin = np.min(a)
    shifted = -alpha * (a - amin)
    shifted = shifted - np.max(shifted)
    log_sum = np.log(np.sum(np.exp(shifted))) + np.max(shifted) - alpha * amin
    return -log_sum / alpha

def _neg_smooth_score(params, alpha=60.0):
    pts = params.reshape(-1, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    return -(_soft_min_triangle_area(pts, alpha) / hull_a)

def _newton_equiarea(pts, k_active=5, max_iter=50, tol=1e-10):
    """
    Damped Newton-Raphson: enforce that the k smallest triangle areas are all equal
    to a target A, and that the convex hull area = 1.
    Variables: 2*N point coordinates + 1 target area A = 2N+1 unknowns.
    Constraints: k_active equality constraints (areas equal A) + 1 (hull area = 1) = k_active+1.
    We use a least-squares approach since 2N+1 > k_active+1 generally.
    """
    n = len(pts)
    if n < 3:
        return pts.copy()
    
    pts = pts.astype(float).copy()
    x = pts.flatten().copy()  # 2N variables
    A = _min_triangle_area(pts)  # target area
    
    for it in range(max_iter):
        if not _time_ok(2.0):
            break
        
        # Compute all triangle areas and their gradients
        areas = _all_triangle_areas(pts)
        if len(areas) == 0:
            break
        
        # Find the k_active smallest triangles
        sorted_idx = np.argsort(areas)
        active_idx = sorted_idx[:k_active]
        active_areas = areas[active_idx]
        
        # Target: all active areas should equal A
        # Residuals: active_areas - A (k_active residuals)
        # Plus: hull_area - 1 (1 residual)
        
        # Build Jacobian for active areas w.r.t. point coordinates
        # For triangle (i, j, k): area = 0.5 * |cross(pj-pi, pk-pi)|
        # d(area)/d(pi) = 0.5 * cross(pj-pi, pk-pi) rotated... 
        # Actually for area = 0.5*|v1 x v2| where v1=pj-pi, v2=pk-pi
        # dA/dpi = -0.5*(v2[0], -v2[1]) - 0.5*(-v1[0], v1[1]) ... let me be more careful
        # A = 0.5 * |x1*y2 - x2*y1| where v1=(x1,y1)=pj-pi, v2=(x2,y2)=pk-pi
        # dA/dpi_x = 0.5*(-y2 + y1) ... no
        # Let me use: A = 0.5*|(pj-pi) x (pk-pi)| = 0.5*|pj_x*pk_y - pj_y*pk_x + ...|
        # Easier: A = 0.5*|det([pj-pi, pk-pi])|
        # dA/dpi = -0.5 * cross(pk-pi, pj-pi) ... 
        # Let me just do finite differences for simplicity and robustness
        
        # Finite difference Jacobian
        J = np.zeros((k_active, 2*n))
        eps = 1e-7
        for d in range(2*n):
            x_pert = x.copy()
            x_pert[d] += eps
            pts_pert = x_pert.reshape(-1, 2)
            areas_pert = _all_triangle_areas(pts_pert)
            if len(areas_pert) != len(areas):
                break
            J[:, d] = (areas_pert[active_idx] - active_areas) / eps
        
        # Hull area gradient (finite difference)
        hull_a = _hull_area(pts)
        J_hull = np.zeros(2*n)
        for d in range(2*n):
            x_pert = x.copy()
            x_pert[d] += eps
            pts_pert = x_pert.reshape(-1, 2)
            J_hull[d] = (_hull_area(pts_pert) - hull_a) / eps
        
        # Full residual vector and Jacobian
        r = np.concatenate([active_areas - A, [hull_a - 1.0]])
        J_full = np.vstack([J, J_hull.reshape(1, -1)])
        
        # Damped least squares: solve (J^T J + lambda*I) dx = -J^T r
        lambda_damp = 0.1
        JTJ = J_full.T @ J_full + lambda_damp * np.eye(2*n)
        JTr = J_full.T @ r
        
        try:
            dx = np.linalg.solve(JTJ, -JTr)
        except np.linalg.LinAlgError:
            dx = np.linalg.lstsq(JTJ, -JTr, rcond=None)[0]
        
        # Damping: try full step, halve if objective worsens
        step = dx
        for damp in [1.0, 0.5, 0.25, 0.1, 0.05]:
            x_new = x + damp * step
            pts_new = x_new.reshape(-1, 2)
            # Check improvement in score
            s_old = _score(pts)
            s_new = _score(pts_new)
            if s_new >= s_old - 1e-12:
                x = x_new
                pts = pts_new
                break
        else:
            break
        
        # Check convergence
        if np.max(np.abs(step)) < tol:
            break
    
    return pts.reshape(-1, 2)

def _neg_ring_score(params, ring_counts):
    pts = _ring_params_to_points(params, ring_counts)
    return -_score(pts)

def _neg_ring_smooth_score(params, ring_counts, alpha=60.0):
    pts = _ring_params_to_points(params, ring_counts)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    return -(_soft_min_triangle_area(pts, alpha) / hull_a)

def _build_from_rings(ring_params):
    """
    Build N points from a list of (count, radius, phase) tuples.
    """
    pts = []
    for count, radius, phase in ring_params:
        if count == 0:
            continue
        if count == 1:
            pts.append(np.array([[radius * np.cos(phase), radius * np.sin(phase)]]))
        else:
            angles = phase + 2 * np.pi * np.arange(count) / count
            pts.append(radius * np.column_stack([np.cos(angles), np.sin(angles)]))
    if not pts:
        return np.zeros((N, 2))
    return np.vstack(pts)

def _ring_params_to_points(params, ring_counts):
    """params: interleaved [r0, p0, r1, p1, ...] for each ring."""
    ring_params = []
    for i, count in enumerate(ring_counts):
        ring_params.append((count, abs(params[2 * i]), params[2 * i + 1]))
    return _build_from_rings(ring_params)

def _anneal_ring(ring_counts, x0, max_iter=120):
    """Progressive-annealing L-BFGS-B on the smooth ring objective, then exact NM polish."""
    pts = _ring_params_to_points(x0, ring_counts)
    cur_score = _score(pts)
    cur_params = x0.copy()
    for alpha in [15.0, 40.0, 90.0, 200.0]:
        if not _time_ok(8.0):
            break
        try:
            res = minimize(
                lambda p, a=alpha: _neg_ring_smooth_score(p, ring_counts, a),
                cur_params,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            cand_pts = _ring_params_to_points(res.x, ring_counts)
            cand_score = _score(cand_pts)
            if cand_score >= cur_score:
                cur_score = cand_score
                cur_params = res.x.copy()
        except Exception:
            pass
    if _time_ok(4.0):
        try:
            res = minimize(
                lambda p: _neg_ring_score(p, ring_counts),
                cur_params,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True}
            )
            cand_pts = _ring_params_to_points(res.x, ring_counts)
            cand_score = _score(cand_pts)
            if cand_score > cur_score:
                cur_score = cand_score
                cur_params = res.x.copy()
        except Exception:
            pass
    return _ring_params_to_points(cur_params, ring_counts), cur_score

def construct():
    rng = np.random.RandomState(42)

    # ---- 1. Diverse initializations ----
    inits = []
    angles13 = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles13), np.sin(angles13)]))
    angles12 = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    inits.append(np.vstack([np.column_stack([np.cos(angles12), np.sin(angles12)]), [[0.0, 0.0]]]))

    ring_structures = [
        [13], [1, 12], [1, 6, 6], [7, 6], [2, 11], [3, 10],
        [4, 9], [5, 8], [6, 7], [1, 3, 9], [1, 4, 8], [1, 5, 7], [1, 2, 10],
    ]
    for rc in ring_structures:
        n_rings = len(rc)
        base_r = [1.0 / (i + 1) for i in range(n_rings)]
        for ph in [0.0, np.pi / 6.0]:
            x0 = np.zeros(2 * n_rings)
            for i in range(n_rings):
                x0[2 * i] = base_r[i]
                x0[2 * i + 1] = ph if i > 0 else 0.0
            inits.append(_ring_params_to_points(x0, rc))

    for _ in range(4):
        r = np.sqrt(rng.rand(N))
        th = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(th), r * np.sin(th)]))

    for jitter in [0.02, 0.06]:
        rr = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([rr * np.cos(angles13), rr * np.sin(angles13)]))

    # ---- 2. Score initializations, keep best ----
    best_score = 0.0
    best_pts = inits[0].copy()
    scored = []
    for init_pts in inits:
        if not _time_ok(30.0):
            break
        s = _score(init_pts)
        scored.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
    scored.sort(key=lambda x: -x[0])

    # ---- 3. Newton equi-area refinement on top candidates ----
    top = []
    for s, p in scored[:6]:
        if not any(np.allclose(p, q, atol=1e-5) for q in top):
            top.append(p)
    if not any(np.allclose(best_pts, q, atol=1e-5) for q in top):
        top.append(best_pts)

    for init_pts in top:
        if not _time_ok(8.0):
            break
        cur_pts = init_pts.copy()
        cur_score = _score(cur_pts)
        
        # Try Newton equi-area with different k_active values
        for k_act in [3, 5, 8]:
            if not _time_ok(3.0):
                break
            try:
                new_pts = _newton_equiarea(cur_pts, k_active=k_act, max_iter=40)
                new_score = _score(new_pts)
                if new_score > cur_score:
                    cur_pts, cur_score = new_pts, new_score
            except Exception:
                pass
        
        # Smooth Powell polish
        if _time_ok(4.0):
            try:
                res = minimize(
                    lambda p: _neg_smooth_score(p, alpha=60.0),
                    cur_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-11, 'fatol': 1e-13}
                )
                cand = res.x.reshape(-1, 2)
                cs = _score(cand)
                if cs > cur_score:
                    cur_pts, cur_score = cand, cs
            except Exception:
                pass
        
        # Exact NM polish
        if _time_ok(2.0):
            try:
                res = minimize(_neg_score, cur_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-13})
                cand = res.x.reshape(-1, 2)
                cs = _score(cand)
                if cs > cur_score:
                    cur_pts, cur_score = cand, cs
            except Exception:
                pass
        
        if cur_score > best_score:
            best_score, best_pts = cur_score, cur_pts.copy()

    # ---- 4. Symmetric ring optimization with annealing ----
    for rc in ring_structures:
        if not _time_ok(8.0):
            break
        n_rings = len(rc)
        base_r = [1.0 / (i + 1) for i in range(n_rings)]
        x0_list = []
        x0_list.append(np.array([base_r[i] if i % 2 == 0 else base_r[i]
                                 for i in range(n_rings)] + [0.0] * n_rings))
        for _ in range(2):
            x0 = np.zeros(2 * n_rings)
            x0[0::2] = np.abs(rng.uniform(0.2, 1.2, size=n_rings))
            x0[1::2] = rng.uniform(-np.pi / 6, np.pi / 6, size=n_rings)
            x0_list.append(x0)
        for x0 in x0_list:
            if not _time_ok(5.0):
                break
            cand_pts, cand_score = _anneal_ring(rc, x0)
            if cand_score > best_score:
                best_score, best_pts = cand_score, cand_pts.copy()
            # Also try Newton on the ring result
            if _time_ok(2.0):
                try:
                    new_pts = _newton_equiarea(cand_pts, k_active=5, max_iter=30)
                    new_score = _score(new_pts)
                    if new_score > best_score:
                        best_score, best_pts = new_score, new_pts.copy()
                except Exception:
                    pass

    # ---- 5. Final high-precision polish on the best ----
    if _time_ok(4.0):
        try:
            # Newton first
            new_pts = _newton_equiarea(best_pts, k_active=5, max_iter=50)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_pts, best_score = new_pts, new_score
        except Exception:
            pass
        if _time_ok(2.0):
            try:
                res = minimize(_neg_score, best_pts.flatten(), method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-13, 'fatol': 1e-15})
                cand = res.x.reshape(-1, 2)
                cs = _score(cand)
                if cs > best_score:
                    best_pts = cand
            except Exception:
                pass

    return best_pts
# EVOLVE-BLOCK-END
