import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triple indices for vectorized triangle area computation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)


def _triangle_areas_vectorized(points):
    """Compute areas of all C(N,3) triangles in vectorized form."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    """Compute convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        if np.allclose(points, points[0], atol=1e-12):
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    """Score = min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(points):
    """Negative score for minimization."""
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _soft_min_value(values, temperature):
    """Smooth approximation of min using log-sum-exp."""
    if temperature <= 1e-12:
        return float(np.min(values))
    min_val = np.min(values)
    shifted = values - min_val
    log_sum = np.log(np.sum(np.exp(-shifted / temperature)))
    return min_val - temperature * log_sum


def _soft_score(points, temperature):
    """Score using soft-min instead of hard min."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    if np.any(areas < 1e-15):
        return 0.0
    soft_min = _soft_min_value(areas, temperature)
    if soft_min < 1e-15:
        return 0.0
    return soft_min / hull_area


def _neg_soft_score(points, temperature):
    """Negative soft score for minimization."""
    s = _soft_score(points, temperature)
    if s <= 0:
        return 1e10
    return -s


# ---------- Radial coordinate helpers ----------

def _radial_to_cartesian(params):
    """Convert [r1,...,rN, theta1,...,thetaN] to (N,2) Cartesian points."""
    r = np.clip(params[:N], 1e-6, 1.0)
    theta = params[N:]
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _cartesian_to_radial(points):
    """Convert (N,2) Cartesian to [r1,...,rN, theta1,...,thetaN]."""
    r = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    return np.concatenate([r, theta])


# ---------- Penalty-based formulation: maximize t s.t. area_i >= t ----------

def _penalty_objective_radial(x, mu):
    """
    Penalty objective in radial coords: minimize -t/hull_area + mu * sum(max(0, t - area_i)^2)
    x = [t, r1,...,rN, theta1,...,thetaN]
    """
    t = x[0]
    r = np.clip(x[1:1+N], 1e-6, 1.0)
    theta = x[1+N:1+2*N]
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    areas = _triangle_areas_vectorized(pts)
    violations = np.maximum(0, t - areas)
    penalty = np.sum(violations ** 2)
    hull_a = _hull_area(pts)
    if hull_a < 1e-12:
        return 1e10
    return -t / hull_a + mu * penalty


def _penalty_optimize_radial(points, deadline, mu_start=10.0, mu_end=1e8, n_mu_steps=15,
                              max_iter_per_mu=120):
    """
    Penalty-based optimization in radial coordinates.
    """
    best = points.copy()
    best_s = _score(best)
    
    areas = _triangle_areas_vectorized(best)
    t_init = float(np.min(areas)) * 0.85
    
    r0 = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    theta0 = np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)
    r0 = np.clip(r0, 1e-6, 1.0)
    
    x0 = np.zeros(1 + 2 * N)
    x0[0] = t_init
    x0[1:1+N] = r0
    x0[1+N:] = theta0
    
    mus = np.geomspace(mu_start, mu_end, n_mu_steps)
    
    for mu in mus:
        if time.time() > deadline - 8:
            break
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu),
                x0,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_mu, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    # Additional polish pass with higher mu
    if time.time() < deadline - 5:
        try:
            res = minimize(
                lambda x: _penalty_objective_radial(x, mu_end),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13},
            )
            x0 = res.x
            r = np.clip(x0[1:1+N], 1e-6, 1.0)
            theta = x0[1+N:]
            cand_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            s = _score(cand_pts)
            if s > best_s:
                best_s = s
                best = cand_pts.copy()
        except Exception:
            pass
    
    return best, best_s


# ---------- Seed generators ----------

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_two_ring(n_out, r_inner, angle_offset):
    """Two-ring configuration."""
    n_in = N - n_out
    a_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    a_in = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_in, endpoint=False)
    outer = np.column_stack([np.cos(a_out), np.sin(a_out)])
    inner = np.column_stack([r_inner * np.cos(a_in), r_inner * np.sin(a_in)])
    return np.vstack([outer, inner])


def _seed_three_ring(n1, n2, n3, r2, r3, off2, off3):
    """Three-ring configuration."""
    a1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
    a2 = np.linspace(off2, 2 * np.pi + off2, n2, endpoint=False)
    a3 = np.linspace(off3, 2 * np.pi + off3, n3, endpoint=False)
    pts = np.vstack([
        np.column_stack([np.cos(a1), np.cos(a1)]),  # placeholder fix below
    ])
    # Correct implementation
    pts_list = []
    pts_list.append(np.column_stack([np.cos(a1), np.sin(a1)]))
    pts_list.append(np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]))
    pts_list.append(np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]))
    return np.vstack(pts_list)


def _seed_quasi_regular(seed=42):
    """Quasi-regular: regular polygon with small random perturbations."""
    rng = np.random.RandomState(seed)
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    r = 1.0 + 0.03 * rng.randn(N)
    r = np.clip(r, 0.5, 1.5)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def _seed_centered_rings():
    """Center + inner ring + outer ring."""
    pts = np.zeros((N, 2))
    # 1 center
    # 6 inner at r=0.5
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[1 + i] = [0.5 * np.cos(a), 0.5 * np.sin(a)]
    # 6 outer at r=1.0
    for i in range(6):
        a = 2 * np.pi * i / 6
        pts[7 + i] = [1.0 * np.cos(a), 1.0 * np.sin(a)]
    return pts


def _seed_golden_angle(seed=0):
    """Points on a circle with golden angle spacing."""
    rng = np.random.RandomState(seed)
    ang = 2 * np.pi * rng.permutation(N) / N
    r = np.ones(N) * (1.0 + 0.02 * rng.randn(N))
    r = np.clip(r, 0.7, 1.3)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


# ---------- Optimization routines ----------

def _annealed_soft_minimize(points, deadline, max_temp=0.15, min_temp=1e-7, n_steps=14):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B then Powell at each temperature level.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 8:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 6:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective
        if time.time() < deadline - 4:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 120, 'xatol': 1e-11, 'fatol': 1e-11},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
    
    return best, best_s


def _polish_true_objective(points, deadline):
    """Polish using the true (non-smooth) objective with Powell then Nelder-Mead."""
    best = points.copy()
    best_s = _score(best)
    
    # Powell
    if time.time() < deadline - 6:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Powell',
                options={'maxiter': 400, 'xatol': 1e-12, 'fatol': 1e-12},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Nelder-Mead fine polish
    if time.time() < deadline - 3:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    # Final high-precision Nelder-Mead
    if time.time() < deadline - 1:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True},
            )
            cand = res.x.reshape(N, 2)
            s = _score(cand)
            if s > best_s:
                best_s = s
                best = cand.copy()
        except Exception:
            pass
    
    return best, best_s


def _local_search(points, deadline, n_iter=500, sigma_start=0.02, sigma_end=0.0003):
    """
    Coordinate-wise local search with adaptive step size.
    Tries moving each point in random directions, keeping improvements.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 5:
            break
        
        # Adaptive sigma
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        # Pick a random point and a random direction
        idx = rng.randint(N)
        direction = rng.randn(2)
        direction /= (np.linalg.norm(direction) + 1e-15)
        
        # Try several step sizes
        for scale in [1.0, 0.5, 0.2, 0.1, 2.0, 0.05]:
            step = sigma * scale
            cand = best.copy()
            cand[idx] += step * direction
            
            s = _score(cand)
            if s > best_s + 1e-15:
                best_s = s
                best = cand.copy()
                break
    
    return best, best_s


# ---------- Support-line polygon representation (structural change) ----------
# Points are vertices of a convex polygon defined by N support lines with
# fixed, evenly-spaced outward normal angles. We optimize only the N offsets.

_FIXED_NORMALS = np.column_stack([
    np.cos(2 * np.pi * np.arange(N) / N),
    np.sin(2 * np.pi * np.arange(N) / N),
])


def _polygon_from_offsets(offsets, big=50.0):
    """
    Given N offsets for N fixed outward normals, build the intersection polygon.
    Halfspace i: n_i . x <= offsets[i].
    Returns (M, 2) vertex array or None.
    """
    offsets = np.asarray(offsets, dtype=float)
    A = np.column_stack([_FIXED_NORMALS, -offsets])
    b = np.ones(N)
    A_box = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0]
    ])
    b_box = np.array([big, big, big, big])
    A_full = np.vstack([A, A_box])
    b_full = np.concatenate([b, b_box])
    try:
        from scipy.spatial import HalfspaceIntersection
        hsi = HalfspaceIntersection(A_full, b_full)
        verts = hsi.intersections
        mask = np.all(np.abs(verts) < big * 0.9, axis=1)
        verts = verts[mask]
        if len(verts) < 3:
            return None
        return verts
    except Exception:
        return None


def _select_n_vertices(verts, n=N):
    """Select n vertices from a convex polygon, evenly spaced by index."""
    m = len(verts)
    if m == n:
        return verts
    if m < n:
        # Pad by inserting midpoints
        out = list(verts)
        idx = 0
        while len(out) < n:
            a = out[idx % len(out)]
            b = out[(idx + 1) % len(out)]
            insert_pos = (idx + 1) % len(out)
            if insert_pos <= idx:
                insert_pos = len(out)
            out.insert(insert_pos, 0.5 * (a + b))
            idx += 1
        return np.array(out[:n])
    # m > n: take evenly spaced indices
    idx = np.linspace(0, m - 1, n, dtype=int)
    return verts[idx]


def _score_from_offsets(offsets):
    """Score for a given set of support-line offsets."""
    verts = _polygon_from_offsets(offsets)
    if verts is None:
        return 0.0
    pts = _select_n_vertices(verts)
    return _score(pts)


def _points_from_offsets(offsets):
    verts = _polygon_from_offsets(offsets)
    if verts is None:
        return None
    return _select_n_vertices(verts)


def _offsets_from_points(points):
    """
    Recover approximate support-line offsets from N points.
    For each fixed normal direction, the offset is the max projection of any point.
    This gives a polygon that contains all points.
    """
    projections = points @ _FIXED_NORMALS.T
    return np.max(projections, axis=0)


def _simulated_annealing_offsets(offsets_init, deadline, n_restarts=2, n_steps=3000):
    """
    Simulated annealing on the N offset parameters.
    Objective: maximize min triangle area / hull area.
    """
    rng = np.random.RandomState(2024)
    best_offsets = offsets_init.copy()
    best_s = _score_from_offsets(best_offsets)
    if best_s <= 0:
        best_s = 1e-15

    T_start = 0.5
    T_end = 1e-5

    for restart in range(n_restarts):
        cur_offsets = offsets_init.copy() + rng.randn(N) * 0.05 * (restart + 1)
        cur_s = _score_from_offsets(cur_offsets)
        if cur_s <= 0:
            cur_s = 1e-15
        cur_f = -cur_s

        for step in range(n_steps):
            if time.time() > deadline - 10:
                break

            alpha = step / max(n_steps - 1, 1)
            T = T_start * (T_end / T_start) ** alpha

            # Propose a perturbation on a random subset of offsets
            n_pert = rng.randint(1, 4)
            idx = rng.choice(N, size=n_pert, replace=False)
            scale = 0.1 * (1.0 - 0.5 * alpha)
            cur_offsets = cur_offsets.copy()
            cur_offsets[idx] += rng.randn(n_pert) * scale

            new_s = _score_from_offsets(cur_offsets)
            if new_s <= 0:
                new_s = 1e-15
            new_f = -new_s

            delta_f = new_f - cur_f
            if delta_f < 0 or rng.random() < np.exp(-delta_f / max(T, 1e-12)):
                cur_f = new_f
                cur_s = new_s
                if cur_s > best_s:
                    best_s = cur_s
                    best_offsets = cur_offsets.copy()

    return best_offsets, best_s


def _support_line_optimize(points, deadline):
    """
    Optimize support line offsets starting from initial points.
    """
    offsets_init = _offsets_from_points(points)
    best_offsets, best_s = _simulated_annealing_offsets(offsets_init, deadline)
    
    # Convert back to points
    pts = _points_from_offsets(best_offsets)
    if pts is None:
        return points, _score(points)
    
    s = _score(pts)
    if s > best_s:
        best_s = s
    
    return pts, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Generate diverse initializations
    inits = []
    inits.append(_seed_regular())
    inits.append(_seed_centered_rings())
    inits.append(_seed_two_ring(7, 0.5, 0.0))
    inits.append(_seed_two_ring(6, 0.4, np.pi / 6))
    inits.append(_seed_two_ring(8, 0.6, np.pi / 4))
    inits.append(_seed_two_ring(7, 0.55, np.pi / 12))
    inits.append(_seed_two_ring(9, 0.45, np.pi / 8))
    inits.append(_seed_three_ring(5, 4, 4, 0.5, 0.8, 0.0, np.pi / 5))
    inits.append(_seed_three_ring(6, 3, 4, 0.4, 0.7, np.pi / 6, np.pi / 3))
    inits.append(_seed_three_ring(7, 3, 3, 0.55, 0.75, np.pi / 7, np.pi / 4))
    inits.append(_seed_quasi_regular())
    inits.append(_seed_quasi_regular(seed=99))
    inits.append(_seed_golden_angle(0))
    inits.append(_seed_golden_angle(1))
    inits.append(_seed_golden_angle(7))

    # Add some random inits
    rng = np.random.RandomState(42)
    for _ in range(4):
        inits.append(rng.random((N, 2)))

    best_points = None
    best_score = -1

    # Phase 1: Annealed soft-min optimization on each seed
    for i, init in enumerate(inits):
        if time.time() >= deadline - 30:
            break
        
        try:
            pts, s = _annealed_soft_minimize(init, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 1.5: Penalty-based optimization in radial coordinates
    if best_points is not None and time.time() < deadline - 20:
        try:
            pts, s = _penalty_optimize_radial(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 1.6: Support-line optimization (structural change)
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _support_line_optimize(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Phase 2: Polish the best result with true objective
    if best_points is not None:
        if time.time() < deadline - 12:
            try:
                pts, s = _polish_true_objective(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Phase 3: Local search refinement
    if best_points is not None:
        if time.time() < deadline - 6:
            try:
                pts, s = _local_search(best_points, deadline)
                if s > best_score:
                    best_score = s
                    best_points = pts.copy()
            except Exception:
                pass
    
    # Final polish if time remains
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
            )
            pts = res.x.reshape(N, 2)
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _seed_regular()
    
    return best_points
