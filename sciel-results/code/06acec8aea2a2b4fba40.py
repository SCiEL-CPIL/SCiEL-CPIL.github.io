import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)
])

# Precompute angles for support-function parameterization
_THETA = np.linspace(0, 2 * np.pi, N, endpoint=False)
_COS_T = np.cos(_THETA)
_SIN_T = np.sin(_THETA)
# cos/sin of theta_i - theta_j for all pairs (needed for vertex computation)
_COS_DIFF = np.cos(_THETA[:, None] - _THETA[None, :])
_SIN_DIFF = np.sin(_THETA[:, None] - _THETA[None, :])

def _support_to_points(h):
    """Convert support distances to vertex coordinates.
    
    For a convex polygon with support function h(θ_i) at angles θ_i,
    vertex i is at the intersection of support lines i-1 and i.
    h_i = x_i * cos(θ_i) + y_i * sin(θ_i)
    h_{i-1} = x_i * cos(θ_{i-1}) + y_i * sin(θ_{i-1})
    
    Solving: x_i = (h_i * sin(θ_{i-1}) - h_{i-1} * sin(θ_i)) / sin(θ_{i-1} - θ_i)
             y_i = (h_{i-1} * cos(θ_{i-1}) - h_i * cos(θ_i)) / sin(θ_{i-1} - θ_i)
    """
    h_prev = np.roll(h, 1)  # h_{i-1}
    # sin(θ_{i-1} - θ_i) = -sin(θ_i - θ_{i-1}) = -sin(2π/N)
    sin_diff = np.sin(2 * np.pi / N)
    # x_i = (h_i * sin(θ_{i-1}) - h_{i-1} * sin(θ_i)) / sin(θ_{i-1} - θ_i)
    #     = (h_i * sin(θ_{i-1}) - h_{i-1} * sin(θ_i)) / (-sin_diff)
    #     = (h_{i-1} * sin(θ_i) - h_i * sin(θ_{i-1})) / sin_diff
    x = (h_prev * _SIN_T - h * np.roll(_SIN_T, 1)) / sin_diff
    # y_i = (h_{i-1} * cos(θ_{i-1}) - h_i * cos(θ_i)) / sin(θ_{i-1} - θ_i)
    #     = (h_i * cos(θ_i) - h_{i-1} * cos(θ_{i-1})) / sin_diff
    y = (h * _COS_T - h_prev * np.roll(_COS_T, 1)) / sin_diff
    return np.column_stack((x, y))

def _support_hull_area(h):
    """Compute hull area from support function using shoelace on vertices."""
    pts = _support_to_points(h)
    x = pts[:, 0]
    y = pts[:, 1]
    # Shoelace formula
    cross = x * np.roll(y, -1) - y * np.roll(x, -1)
    area = 0.5 * np.sum(cross)
    return abs(area)

def _compute_areas(points):
    """Compute all triangle areas efficiently using vectorized cross products."""
    p = points
    C = p[:, 0:1] * p[:, 1:2].T - p[:, 1:2] * p[:, 0:1].T
    i_idx = TRIPLE_INDICES[:, 0]
    j_idx = TRIPLE_INDICES[:, 1]
    k_idx = TRIPLE_INDICES[:, 2]
    vals = C[j_idx, k_idx] + C[i_idx, j_idx] - C[i_idx, k_idx]
    return 0.5 * np.abs(vals)

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _calculate_score(points):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _soft_min_score(points, temperature=0.015):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    max_shift = np.max(shifted)
    s = np.sum(np.exp((shifted - max_shift) / temperature))
    soft_min = m + max_shift - temperature * np.log(s)
    return soft_min / hull_area

def _two_temp_soft_min(points, temp1=0.008, temp2=0.04):
    if points.shape != (N, 2):
        return 0.0
    try:
        hull_area = _hull_area(points)
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0
    areas = _compute_areas(points)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    max_shift = np.max(shifted)
    s1 = np.sum(np.exp((shifted - max_shift) / temp1))
    soft1 = m + max_shift - temp1 * np.log(s1)
    s2 = np.sum(np.exp((shifted - max_shift) / temp2))
    soft2 = m + max_shift - temp2 * np.log(s2)
    return (soft1 * soft2) ** 0.5 / hull_area

def _support_score(h):
    """Score from support distances: min triangle area / hull area."""
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return 0.0
    if np.any(h < 1e-8):
        return 0.0
    pts = _support_to_points(h)
    hull_area = _support_hull_area(h)
    if hull_area < 1e-9:
        return 0.0
    areas = _compute_areas(pts)
    return np.min(areas) / hull_area

def _support_soft_min(h, temperature=0.015):
    """Soft-min score from support distances."""
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return 0.0
    if np.any(h < 1e-8):
        return 0.0
    pts = _support_to_points(h)
    hull_area = _support_hull_area(h)
    if hull_area < 1e-9:
        return 0.0
    areas = _compute_areas(pts)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    max_shift = np.max(shifted)
    s = np.sum(np.exp((shifted - max_shift) / temperature))
    soft_min = m + max_shift - temperature * np.log(s)
    return soft_min / hull_area

def _support_two_temp(h, temp1=0.008, temp2=0.04):
    """Two-temperature soft-min from support distances."""
    h = np.asarray(h, dtype=float)
    if h.shape != (N,):
        return 0.0
    if np.any(h < 1e-8):
        return 0.0
    pts = _support_to_points(h)
    hull_area = _support_hull_area(h)
    if hull_area < 1e-9:
        return 0.0
    areas = _compute_areas(pts)
    m = np.max(areas)
    if m < 1e-12:
        return 0.0
    shifted = areas - m
    max_shift = np.max(shifted)
    s1 = np.sum(np.exp((shifted - max_shift) / temp1))
    soft1 = m + max_shift - temp1 * np.log(s1)
    s2 = np.sum(np.exp((shifted - max_shift) / temp2))
    soft2 = m + max_shift - temp2 * np.log(s2)
    return (soft1 * soft2) ** 0.5 / hull_area

def _optimize_support(time_limit=30.0):
    """Optimize directly in support-distance space (14-dim, smooth)."""
    start_time = time.time()
    best_score = 0.0
    best_h = None

    def neg_soft(h, temp=0.015):
        return -_support_soft_min(h, temperature=temp)

    def neg_two_temp(h):
        return -_support_two_temp(h)

    def neg_hard(h):
        return -_support_score(h)

    # Regular polygon as initial guess
    h0 = np.ones(N)
    s0 = _support_score(h0)
    if s0 > best_score:
        best_score, best_h = s0, h0.copy()

    rng = np.random.default_rng(42)

    # Multi-start with L-BFGS-B on soft surrogate
    n_starts = 12
    for trial in range(n_starts):
        if time.time() - start_time > time_limit:
            break
        if trial == 0:
            h_init = h0.copy()
        else:
            # Perturbed regular polygon
            h_init = h0 + rng.uniform(-0.1, 0.1, size=N)
            h_init = np.maximum(h_init, 0.3)

        # Stage 1: L-BFGS-B on two-temp soft (loose)
        try:
            res = minimize(neg_two_temp, h_init, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
            h1 = res.x
            h1 = np.maximum(h1, 0.1)
            s1 = _support_score(h1)
            if s1 > best_score:
                best_score, best_h = s1, h1.copy()
        except Exception:
            pass

        # Stage 2: Annealed soft-min L-BFGS-B
        if time.time() - start_time < time_limit * 0.8:
            for temp in [0.03, 0.015, 0.008, 0.004, 0.002]:
                if time.time() - start_time > time_limit * 0.8:
                    break
                try:
                    res = minimize(lambda h, t=temp: neg_soft(h, t), h1, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-9})
                    h2 = res.x
                    h2 = np.maximum(h2, 0.1)
                    s2 = _support_score(h2)
                    if s2 > best_score:
                        best_score, best_h = s2, h2.copy()
                    h1 = h2
                except Exception:
                    pass

        # Stage 3: Hard objective L-BFGS-B
        if time.time() - start_time < time_limit * 0.6:
            try:
                res = minimize(neg_hard, h1, method='L-BFGS-B',
                               options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10})
                h3 = res.x
                h3 = np.maximum(h3, 0.1)
                s3 = _support_score(h3)
                if s3 > best_score:
                    best_score, best_h = s3, h3.copy()
            except Exception:
                pass

        # Stage 4: Nelder-Mead on hard
        if time.time() - start_time < time_limit * 0.4:
            try:
                res = minimize(neg_hard, h1, method='Nelder-Mead',
                               options={'xatol': 1e-10, 'fatol': 1e-11, 'maxiter': 500, 'disp': False})
                h4 = res.x
                h4 = np.maximum(h4, 0.1)
                s4 = _support_score(h4)
                if s4 > best_score:
                    best_score, best_h = s4, h4.copy()
            except Exception:
                pass

    # Final intensive polish on best
    if best_h is not None and time.time() < start_time + time_limit - 2:
        remaining = time_limit - (time.time() - start_time) - 1
        if remaining > 2:
            try:
                res = minimize(neg_hard, best_h, method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12})
                hf = res.x
                hf = np.maximum(hf, 0.1)
                sf = _support_score(hf)
                if sf > best_score:
                    best_score, best_h = sf, hf.copy()
            except Exception:
                pass

    if best_h is None:
        best_h = h0
    return best_score, best_h

def _polish_points(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    def neg_two_temp(x):
        return -_two_temp_soft_min(x.reshape(N, 2))

    def neg_soft(x, temp=0.015):
        return -_soft_min_score(x.reshape(N, 2), temperature=temp)

    def neg_hard(x):
        return -_calculate_score(x.reshape(N, 2))

    # Stage 1: L-BFGS-B on two-temperature smooth surrogate (loose)
    try:
        res = minimize(neg_two_temp, p.flatten(), method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-13, 'gtol': 1e-10})
        np2 = res.x.reshape(N, 2)
        s2 = _calculate_score(np2)
        if s2 > s:
            p, s = np2, s2
    except Exception:
        pass

    # Stage 2: L-BFGS-B with annealed soft-min (progressively tighter)
    if time.time() - start_time < time_limit * 0.7:
        for temp in [0.03, 0.015, 0.008, 0.004]:
            if time.time() - start_time > time_limit * 0.7:
                break
            try:
                res = minimize(lambda x: neg_soft(x, temp), p.flatten(), method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-13, 'gtol': 1e-10})
                np2 = res.x.reshape(N, 2)
                s2 = _calculate_score(np2)
                if s2 > s:
                    p, s = np2, s2
            except Exception:
                pass

    # Stage 3: L-BFGS-B on the true (piecewise) objective
    if time.time() - start_time < time_limit * 0.5:
        try:
            res = minimize(neg_hard, p.flatten(), method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11})
            np2 = res.x.reshape(N, 2)
            s2 = _calculate_score(np2)
            if s2 > s:
                p, s = np2, s2
        except Exception:
            pass

    # Stage 4: Coordinate descent polish with annealing
    if time.time() - start_time < time_limit * 0.4:
        cd_limit = time_limit - (time.time() - start_time) - 0.5
        if cd_limit > 1.0:
            try:
                p, s = _coordinate_descent_polish(p, time_limit=cd_limit)
            except Exception:
                pass

    # Stage 5: Final Nelder-Mead on hard objective
    if time.time() - start_time < time_limit * 0.3:
        nm_limit = time_limit - (time.time() - start_time) - 0.5
        if nm_limit > 1.0:
            try:
                res = minimize(neg_hard, p.flatten(), method='Nelder-Mead',
                               options={'xatol': 1e-10, 'fatol': 1e-11, 'maxiter': 800, 'disp': False})
                np2 = res.x.reshape(N, 2)
                s2 = _calculate_score(np2)
                if s2 > s:
                    p, s = np2, s2
            except Exception:
                pass

    return p, s

def _coordinate_descent_polish(points, time_limit=5.0):
    start_time = time.time()
    p = points.copy()
    s = _calculate_score(p)

    for iteration in range(6):
        if time.time() - start_time > time_limit:
            break
        temp = 0.05 * (0.5 ** iteration)
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            old_pt = p[i].copy()

            def obj_2d(x):
                p[i] = x
                return -_soft_min_score(p, temperature=temp)

            try:
                res = minimize(obj_2d, old_pt, method='Nelder-Mead',
                               options={'xatol': 1e-9, 'fatol': 1e-9, 'maxiter': 400, 'disp': False})
                p[i] = res.x
                s_new = _calculate_score(p)
                if s_new <= s - 1e-12:
                    p[i] = old_pt
                else:
                    s = s_new
            except Exception:
                p[i] = old_pt
        s = _calculate_score(p)
    return p, s

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    best_score = 0.0
    best_points = None

    def try_update(score, points):
        nonlocal best_score, best_points
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()

    # 1. Support-function optimization (primary: smooth 14-dim, no hull calls)
    if time.time() < deadline - 30:
        s_sup, h_sup = _optimize_support(time_limit=50.0)
        if h_sup is not None:
            pts_sup = _support_to_points(h_sup)
            s_check = _calculate_score(pts_sup)
            try_update(s_check, pts_sup)
            # Polish in point space too
            if time.time() < deadline - 15:
                polished, ps = _polish_points(pts_sup, time_limit=15.0)
                try_update(ps, polished)

    # 2. Regular 14-gon baseline + polish
    if time.time() < deadline - 15:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts_reg = np.column_stack((np.cos(angles), np.sin(angles)))
        s_reg = _calculate_score(pts_reg)
        try_update(s_reg, pts_reg)
        if time.time() < deadline - 10:
            polished, ps = _polish_points(pts_reg, time_limit=10.0)
            try_update(ps, polished)

    # 3. Jittered starts in support space
    if time.time() < deadline - 20:
        rng = np.random.default_rng(123)
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            h_jit = np.ones(N) + rng.uniform(-0.15, 0.15, size=N)
            h_jit = np.maximum(h_jit, 0.4)
            pts_jit = _support_to_points(h_jit)
            s_jit = _calculate_score(pts_jit)
            try_update(s_jit, pts_jit)
            if time.time() < deadline - 12:
                polished, ps = _polish_points(pts_jit, time_limit=8.0)
                try_update(ps, polished)

    # 4. Final intensive polish on best
    if best_points is not None and time.time() < deadline - 8:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            polished, ps = _polish_points(best_points, time_limit=remaining)
            try_update(ps, polished)

    # 5. Last-resort support polish
    if best_points is not None and time.time() < deadline - 3:
        # Convert best points back to approximate support and re-optimize
        remaining = deadline - time.time() - 1.5
        if remaining > 3:
            try:
                # Estimate support distances from best points
                x = best_points[:, 0]
                y = best_points[:, 1]
                h_est = x * _COS_T + y * _SIN_T
                h_est = np.maximum(h_est, 0.1)
                s_est = _support_score(h_est)
                # Quick L-BFGS on support
                res = minimize(lambda h: -_support_score(h), h_est, method='L-BFGS-B',
                               options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11})
                h_final = np.maximum(res.x, 0.1)
                pts_final = _support_to_points(h_final)
                s_final = _calculate_score(pts_final)
                try_update(s_final, pts_final)
            except Exception:
                pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
