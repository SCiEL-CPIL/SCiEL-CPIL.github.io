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


def _epigraph_objective(params, temperature=0.01):
    """
    Epigraph formulation: params = [x1,y1,...,x13,y13, t]
    Minimize: -t + penalty * max(0, t*H - min_area)^2 / H
    where H = hull area, min_area = min triangle area.
    This jointly optimizes points and the target minimum area t.
    """
    pts = params[:2*N].reshape(N, 2)
    t = params[2*N]
    hull_area = _hull_area(pts)
    if hull_area < 1e-12:
        return 1e10
    areas = _triangle_areas_vectorized(pts)
    min_area = float(np.min(areas))
    # We want t * hull_area <= min_area, i.e., t <= min_area / hull_area
    # Penalty for violation
    violation = max(0.0, t * hull_area - min_area)
    # Also penalize if t is too small (encourage larger t)
    # Objective: minimize -t + violation_penalty
    penalty_weight = 100.0
    obj = -t + penalty_weight * violation * violation / (hull_area * hull_area + 1e-12)
    return obj


# ---------- Support-line polygon representation ----------
# Points are vertices of a convex polygon defined by N support lines with
# fixed, evenly-spaced outward normal angles. We optimize only the N offsets.

_FIXED_NORMALS = np.column_stack([
    np.cos(2 * np.pi * np.arange(N) / N),
    np.sin(2 * np.pi * np.arange(N) / N),
])


def _polygon_from_offsets_fast(offsets, big=50.0):
    """
    Fast polygon intersection using Sutherland-Hodgman clipping.
    Halfspace i: n_i . x <= offsets[i].
    Returns (M, 2) vertex array or None.
    """
    offsets = np.asarray(offsets, dtype=float)
    normals = _FIXED_NORMALS
    
    # Start with a large square
    verts = np.array([[-big, -big], [big, -big], [big, big], [-big, big]], dtype=float)
    
    for i in range(N):
        n = normals[i]
        c = offsets[i]
        new_verts = []
        m = len(verts)
        for j in range(m):
            v0 = verts[j]
            v1 = verts[(j + 1) % m]
            d0 = n[0] * v0[0] + n[1] * v0[1] - c
            d1 = n[0] * v1[0] + n[1] * v1[1] - c
            
            if d0 <= 0:
                new_verts.append(v0)
            if d0 * d1 < 0:
                # Edge crosses the line
                t = d0 / (d0 - d1)
                ix = v0 + t * (v1 - v0)
                new_verts.append(ix)
        
        if len(new_verts) < 3:
            return None
        verts = np.array(new_verts)
    
    if len(verts) < 3:
        return None
    return verts


def _select_n_vertices(verts, n=N):
    """Select n vertices from a convex polygon, evenly spaced by arc length."""
    m = len(verts)
    if m == n:
        return verts
    if m < n:
        # Pad by inserting midpoints on the longest edges
        out = list(verts)
        while len(out) < n:
            # Find longest edge
            max_len = 0
            max_idx = 0
            for i in range(len(out)):
                a = out[i]
                b = out[(i + 1) % len(out)]
                L = np.linalg.norm(b - a)
                if L > max_len:
                    max_len = L
                    max_idx = i
            a = out[max_idx]
            b = out[(max_idx + 1) % len(out)]
            mid = 0.5 * (a + b)
            out.insert(max_idx + 1, mid)
        return np.array(out[:n])
    # m > n: take evenly spaced by cumulative arc length
    cum = np.zeros(m + 1)
    for i in range(m):
        cum[i+1] = cum[i] + np.linalg.norm(verts[(i+1)%m] - verts[i])
    total = cum[-1]
    targets = np.linspace(0, total, n, endpoint=False)
    result = []
    for t in targets:
        idx = np.searchsorted(cum, t, side='right') - 1
        idx = min(idx, m - 1)
        if cum[idx+1] - cum[idx] > 1e-15:
            frac = (t - cum[idx]) / (cum[idx+1] - cum[idx])
        else:
            frac = 0.0
        p = verts[idx] + frac * (verts[(idx+1)%m] - verts[idx])
        result.append(p)
    return np.array(result)


def _score_from_offsets(offsets):
    """Score for a given set of support-line offsets."""
    verts = _polygon_from_offsets_fast(offsets)
    if verts is None:
        return 0.0
    pts = _select_n_vertices(verts)
    return _score(pts)


def _points_from_offsets(offsets):
    verts = _polygon_from_offsets_fast(offsets)
    if verts is None:
        return None
    return _select_n_vertices(verts)


def _offsets_from_points(points):
    """
    Recover approximate support-line offsets from N points.
    For each fixed normal direction, the offset is the max projection of any point.
    This gives a polygon that contains all points.
    """
    offsets = points @ _FIXED_NORMALS.T
    return offsets


def _simulated_annealing_offsets(offsets_init, deadline, n_restarts=2):
    """
    Simulated annealing on the N offset parameters.
    Uses the fast polygon intersection for speed.
    """
    rng = np.random.RandomState(2024)
    best_offsets = offsets_init.copy()
    best_s = _score_from_offsets(best_offsets)
    if best_s <= 0:
        best_s = 1e-15

    T_start = 0.3
    T_end = 1e-6
    n_steps = 8000

    for restart in range(n_restarts):
        cur_offsets = offsets_init.copy() + rng.randn(N) * 0.03 * (restart + 1)
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
            n_pert = rng.randint(1, 3)
            idx = rng.choice(N, size=n_pert, replace=False)
            scale = 0.05 * (1.0 - 0.7 * alpha)
            cur_offsets = cur_offsets.copy()
            cur_offsets[idx] += rng.randn(n_pert) * scale

            new_s = _score_from_offsets(cur_offsets)
            if new_s <= 0:
                new_s = 1e-15
            new_f = -new_s

            delta_f = new_f - cur_f
            if delta_f < 0 or (T > 1e-8 and rng.random() < np.exp(-delta_f / T)):
                cur_f = new_f
                cur_s = new_s
                if cur_s > best_s:
                    best_s = cur_s
                    best_offsets = cur_offsets.copy()

    return best_offsets, best_s


# ---------- Seed generators ----------

def _seed_regular():
    """Regular 13-gon."""
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _seed_regular_offsets():
    """Offsets for a regular 13-gon inscribed in unit circle."""
    # For a regular N-gon with circumradius R, the inradius (apothem) is R*cos(pi/N)
    # The support line offset for each normal equals the inradius.
    return np.full(N, np.cos(np.pi / N))


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
        np.column_stack([np.cos(a1), np.sin(a1)]),
        np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)]),
        np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)]),
    ])
    return pts


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

def _annealed_soft_minimize(points, deadline, max_temp=0.1, min_temp=1e-6, n_steps=8):
    """
    Anneal the soft-min temperature and optimize at each step.
    Uses L-BFGS-B on soft objective, then Powell for refinement.
    """
    best = points.copy()
    best_s = _score(best)
    
    temps = np.exp(np.linspace(np.log(max_temp), np.log(min_temp), n_steps))
    
    for temp in temps:
        if time.time() >= deadline - 6:
            break
        
        # L-BFGS-B on soft objective
        if time.time() < deadline - 4:
            try:
                def obj(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj, best.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12},
                )
                cand = res.x.reshape(N, 2)
                s = _score(cand)
                if s > best_s:
                    best_s = s
                    best = cand.copy()
            except Exception:
                pass
        
        # Powell on soft objective for fine-tuning
        if time.time() < deadline - 2:
            try:
                def obj2(x):
                    pts = x.reshape(N, 2)
                    return _neg_soft_score(pts, temp)
                
                res = minimize(
                    obj2, best.flatten(),
                    method='Powell',
                    options={'maxiter': 80, 'xatol': 1e-10, 'fatol': 1e-10},
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


def _local_search(points, deadline, n_iter=2000, sigma_start=0.015, sigma_end=0.0002):
    """
    Local search with single-point and pairwise perturbations.
    """
    best = points.copy()
    best_s = _score(best)
    rng = np.random.RandomState(123)
    
    for it in range(n_iter):
        if time.time() >= deadline - 4:
            break
        
        # Adaptive sigma
        progress = it / max(n_iter - 1, 1)
        sigma = sigma_start * (sigma_end / sigma_start) ** progress
        
        if rng.random() < 0.7:
            # Single point perturbation
            idx = rng.randint(N)
            direction = rng.randn(2)
            direction /= (np.linalg.norm(direction) + 1e-15)
            for scale in [1.0, 0.5, 0.2, 0.1]:
                step = sigma * scale
                cand = best.copy()
                cand[idx] += step * direction
                s = _score(cand)
                if s > best_s + 1e-14:
                    best_s = s
                    best = cand.copy()
                    break
        else:
            # Pairwise perturbation: move two points in opposite-ish directions
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            d1 = rng.randn(2)
            d1 /= (np.linalg.norm(d1) + 1e-15)
            d2 = -0.5 * d1 + 0.5 * rng.randn(2)
            d2 /= (np.linalg.norm(d2) + 1e-15)
            for scale in [1.0, 0.5, 0.2]:
                step = sigma * scale
                cand = best.copy()
                cand[idx1] += step * d1
                cand[idx2] += step * d2
                s = _score(cand)
                if s > best_s + 1e-14:
                    best_s = s
                    best = cand.copy()
                    break
    
    return best, best_s


def _epigraph_optimize(points, deadline, t_init=None):
    """
    Optimize using epigraph formulation with L-BFGS-B.
    params = [x1,y1,...,x13,y13, t]
    """
    best = points.copy()
    best_s = _score(best)
    if t_init is None:
        t_init = best_s * 0.95
    
    params0 = np.concatenate([best.flatten(), [t_init]])
    
    try:
        res = minimize(
            _epigraph_objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12},
        )
        cand_pts = res.x[:2*N].reshape(N, 2)
        cand_t = res.x[2*N]
        s = _score(cand_pts)
        if s > best_s:
            best_s = s
            best = cand_pts.copy()
    except Exception:
        pass
    
    return best, best_s


def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    best_points = None
    best_score = -1

    # ---------- Phase A: Support-line SA (fast polygon intersection) ----------
    reg_offsets = _seed_regular_offsets()

    offset_inits = [reg_offsets]
    rng = np.random.RandomState(7)
    for _ in range(2):
        offset_inits.append(reg_offsets + rng.randn(N) * 0.04)
    # Non-uniform pattern
    alt = np.full(N, np.cos(np.pi / N))
    alt[::2] *= 1.08
    alt[1::2] *= 0.92
    offset_inits.append(alt)

    for oi, off_init in enumerate(offset_inits):
        if time.time() >= deadline - 60:
            break
        try:
            off_opt, s = _simulated_annealing_offsets(off_init, deadline)
            if s > best_score:
                best_score = s
                pts = _points_from_offsets(off_opt)
                if pts is not None:
                    best_points = pts
        except Exception:
            pass

    # ---------- Phase B: Coordinate-based soft-min optimization ----------
    inits = []
    if best_points is not None:
        inits.append(best_points.copy())
    inits.append(_seed_regular())
    inits.append(_seed_two_ring(7, 0.5, 0.0))
    inits.append(_seed_quasi_regular())

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

    # ---------- Phase C: Epigraph optimization ----------
    if best_points is not None and time.time() < deadline - 15:
        try:
            pts, s = _epigraph_optimize(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase D: Polish with true objective ----------
    if best_points is not None and time.time() < deadline - 12:
        try:
            pts, s = _polish_true_objective(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase E: Local search ----------
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts, s = _local_search(best_points, deadline)
            if s > best_score:
                best_score = s
                best_points = pts.copy()
        except Exception:
            pass

    # ---------- Phase F: Final Nelder-Mead polish ----------
    if best_points is not None and time.time() < deadline - 2:
        try:
            res = minimize(
                lambda x: _neg_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True},
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
