"""Heilbronn convex regions n=14: Fourier-harmonic convex hull with optimized point placement.

Key structural change: represent the convex boundary as a smooth closed curve
r(θ) = R0 + Σ a_k cos(kθ) + b_k sin(kθ), k=1..K, with R0 large enough to
guarantee strict convexity. The 14 points are placed at optimized arc-length
parameters along this curve. This reduces search dimensionality from 28 to
~12 (harmonics) + 14 (point positions) = 26, but the harmonic constraint
ensures convexity and smoothness, making the landscape better conditioned.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
K_HARM = 3  # number of Fourier harmonics (k=1..K_HARM), giving 2*K_HARM + 1 = 7 shape params
M_SAMPLES = 256  # samples for arc-length computation


def _all_triangle_areas(points):
    """Compute all C(n,3) triangle areas vectorized."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    # Use broadcasting for all triples
    i_idx, j_idx, k_idx = np.indices((n, n, n))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    xi, yi = x[i_idx[mask]], y[i_idx[mask]]
    xj, yj = x[j_idx[mask]], y[j_idx[mask]]
    xk, yk = x[k_idx[mask]], y[k_idx[mask]]
    areas = 0.5 * np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    return areas


def _min_triangle_area(points):
    """Exact minimum triangle area over all C(n,3) triples."""
    return np.min(_all_triangle_areas(points))


def _fourier_curve(coeffs, theta):
    """Evaluate the Fourier-harmonic curve r(θ) at angles theta.
    coeffs = [R0, a1, b1, a2, b2, ..., aK, bK]
    Returns (x, y) arrays."""
    R0 = coeffs[0]
    r = np.full_like(theta, R0, dtype=float)
    for k in range(1, K_HARM + 1):
        ak = coeffs[2*k - 1]
        bk = coeffs[2*k]
        r += ak * np.cos(k * theta) + bk * np.sin(k * theta)
    # Ensure positive radius
    r = np.maximum(r, 0.05)
    return r * np.cos(theta), r * np.sin(theta)


def _arc_length_params(coeffs, n_points):
    """Given Fourier coefficients, compute n_points equally-spaced (by arc length)
    parameter values on the curve. Returns theta array of length n_points."""
    # Sample the curve densely
    theta_s = np.linspace(0, 2 * np.pi, M_SAMPLES, endpoint=False)
    r_s = np.empty(M_SAMPLES)
    for i, t in enumerate(theta_s):
        r_s[i] = coeffs[0]
        for k in range(1, K_HARM + 1):
            r_s[i] += coeffs[2*k-1] * np.cos(k*t) + coeffs[2*k] * np.sin(k*t)
        r_s[i] = max(r_s[i], 0.05)
    
    x_s = r_s * np.cos(theta_s)
    y_s = r_s * np.sin(theta_s)
    
    # Compute cumulative arc length
    dx = np.diff(x_s, prepend=x_s[-1])
    dy = np.diff(y_s, prepend=y_s[-1])
    seg_len = np.sqrt(dx**2 + dy**2)
    cum_len = np.cumsum(seg_len)
    cum_len -= cum_len[0]
    total_len = cum_len[-1] + seg_len[-1]
    if total_len < 1e-12:
        return np.linspace(0, 2*np.pi, n_points, endpoint=False)
    
    # Desired arc lengths for n equally-spaced points
    target_lens = np.linspace(0, total_len, n_points, endpoint=False)
    
    # Interpolate to find theta values
    # cum_len is monotonically increasing over [0, total_len]
    theta_vals = np.interp(target_lens, cum_len, np.concatenate([theta_s, [theta_s[0] + 2*np.pi]]))
    theta_vals = theta_vals % (2 * np.pi)
    return theta_vals


def _hull_area(points):
    try:
        return ConvexHull(points).volume
    except Exception:
        return 0.0


def _fourier_hull_area(coeffs):
    """Compute hull area directly from Fourier curve via Green's theorem."""
    theta_s = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    r_s = np.full(4096, coeffs[0])
    for k in range(1, K_HARM + 1):
        r_s += coeffs[2*k-1] * np.cos(k*theta_s) + coeffs[2*k] * np.sin(k*theta_s)
    r_s = np.maximum(r_s, 0.05)
    x_s = r_s * np.cos(theta_s)
    y_s = r_s * np.sin(theta_s)
    # Shoelace formula
    area = 0.5 * np.abs(np.sum(x_s * np.roll(y_s, -1) - np.roll(x_s, -1) * y_s))
    return area


def _score(points):
    """min triangle area / convex hull area."""
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2) or np.any(~np.isfinite(points)):
        return 0.0
    ha = _hull_area(points)
    if ha < 1e-12:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0.0:
        return 0.0
    return ma / ha


def _fourier_points(coeffs, point_params):
    """Place points on the Fourier curve at given parameter values.
    coeffs: shape (2*K_HARM + 1,)
    point_params: shape (N,) - angles on [0, 2π)
    Returns (N, 2) array."""
    theta = np.asarray(point_params, dtype=float) % (2 * np.pi)
    x, y = _fourier_curve(coeffs, theta)
    return np.column_stack([x, y])


def _fourier_score_fast(coeffs, point_params):
    """Fast score computation for Fourier representation.
    Uses consecutive triples for min area (valid for convex position) and
    direct hull area from curve."""
    pts = _fourier_points(coeffs, point_params)
    if np.any(~np.isfinite(pts)):
        return 0.0
    # Check if points are in convex position and sorted by angle
    # Sort by angle
    angles = np.arctan2(pts[:, 1], pts[:, 0])
    order = np.argsort(angles)
    pts_sorted = pts[order]
    
    # Min triangle area: check all triples for accuracy but use consecutive for speed
    # For now, use all triples (N=14 gives C(14,3)=364, fast enough)
    ma = _min_triangle_area(pts_sorted)
    if ma <= 0:
        return 0.0
    
    # Hull area from the curve
    ha = _fourier_hull_area(coeffs)
    if ha < 1e-12:
        return 0.0
    return ma / ha


def _regular_polygon(k, r=1.0, phase=0.0):
    a = phase + 2 * np.pi * np.arange(k) / k
    return r * np.column_stack([np.cos(a), np.sin(a)])


def _regular_fourier_coeffs(R0=1.0):
    """Fourier coefficients for a circle of radius R0."""
    coeffs = np.zeros(2 * K_HARM + 1)
    coeffs[0] = R0
    return coeffs


def _regular_point_params(n=N):
    """Equally-spaced point parameters for a regular n-gon."""
    return 2 * np.pi * np.arange(n) / n


def _pts_from_angles(theta, r=1.0):
    """Points on unit circle from angles (not necessarily sorted)."""
    th = np.asarray(theta, dtype=float)
    return r * np.column_stack([np.cos(th), np.sin(th)])


def _fourier_objective(flat):
    """Objective: minimize negative score.
    flat = [coeffs (2*K_HARM+1), point_params (N)]
    Total dimension: 2*K_HARM + 1 + N = 7 + 14 = 21"""
    dim_shape = 2 * K_HARM + 1
    coeffs = flat[:dim_shape]
    pparams = flat[dim_shape:]
    
    # Ensure R0 is sufficiently large for convexity
    # Convexity condition for r(θ) = R0 + Σ(a_k cos kθ + b_k sin kθ):
    # r''(θ) + r(θ) > 0 for all θ
    # r'' = -Σ k²(a_k cos kθ + b_k sin kθ)
    # So we need R0 + Σ(a_k cos kθ + b_k sin kθ) - Σ k²(a_k cos kθ + b_k sin kθ) > 0
    # Worst case: R0 - Σ k² * |a_k| - Σ k² * |b_k| > 0
    # So require R0 > Σ k² * (|a_k| + |b_k|)
    harm_magnitude = 0.0
    for k in range(1, K_HARM + 1):
        ak = abs(coeffs[2*k - 1])
        bk = abs(coeffs[2*k])
        harm_magnitude += k * k * (ak + bk)
    
    min_R0 = harm_magnitude + 0.1  # safety margin
    if coeffs[0] < min_R0:
        # Penalize non-convex configurations heavily
        penalty = (min_R0 - coeffs[0]) ** 2 * 100
        return 1.0 + penalty
    
    s = _fourier_score_fast(coeffs, pparams)
    if s < 1e-12:
        return 1.0
    return -s


def _consecutive_min_triangle_area(theta):
    """Fast O(n) min triangle area assuming points on circle, sorted by angle.
    Only checks consecutive triples which is valid for convex position."""
    th = np.sort(theta)
    n = len(th)
    min_area = float('inf')
    for i in range(n):
        i1, i2, i3 = i % n, (i+1) % n, (i+2) % n
        d1 = th[i2] - th[i1]
        d2 = th[i3] - th[i2]
        # Area of triangle with vertices at angles separated by d1, d2 on unit circle
        # = 0.5 * |sin(d1) * sin(d2) * sin(2*pi - d1 - d2)| but simpler to compute directly
        x1, y1 = np.cos(th[i1]), np.sin(th[i1])
        x2, y2 = np.cos(th[i2]), np.sin(th[i2])
        x3, y3 = np.cos(th[i3]), np.sin(th[i3])
        a = 0.5 * abs((x2-x1)*(y3-y1) - (y2-y1)*(x3-x1))
        if a < min_area:
            min_area = a
    return min_area


def _fast_ratio_from_angles(theta):
    """Fast ratio using consecutive triples only (valid for convex position on circle)."""
    th = np.sort(theta)
    min_area = _consecutive_min_triangle_area(th)
    # Hull area for points on unit circle in convex position = area of polygon
    # which can be computed as sum of cross products
    x = np.cos(th)
    y = np.sin(th)
    # Shoelace formula
    ha = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if ha < 1e-12:
        return 0.0
    return min_area / ha


def _objective_from_angles(theta_flat):
    """Objective for minimization: minimize -ratio (maximize ratio).
    Uses fast consecutive-triple approximation for speed."""
    th = np.asarray(theta_flat, dtype=float)
    if th.shape != (N,) or np.any(~np.isfinite(th)):
        return 1e10
    ratio = _fast_ratio_from_angles(th)
    if ratio < 1e-12:
        return 1e10
    return -ratio


def _objective_exact_from_flat(flat):
    """Objective using exact min over all triples (slower but accurate)."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s


def _nm_optimize(x0, deadline, obj, maxiter=5000):
    """Nelder-Mead with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-12,
                                   'fatol': 1e-14, 'adaptive': True,
                                   'maxfev': maxiter * 20})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _powell_optimize(x0, deadline, obj, maxiter=1000):
    """Powell with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='Powell',
                          options={'maxiter': maxiter, 'xtol': 1e-14,
                                   'ftol': 1e-16})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _lbfgs_optimize(x0, deadline, obj, maxiter=2000):
    """L-BFGS-B with time guard."""
    if time.time() > deadline:
        return None
    try:
        result = minimize(obj, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14,
                                   'gtol': 1e-12, 'maxfun': maxiter * 50})
        if time.time() > deadline:
            return None
        return result
    except Exception:
        return None


def _consider(pts, best):
    """Update best if pts gives a better score."""
    if pts is None or pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return
    s = _score(pts)
    if s > best['score']:
        best['score'] = s
        best['pts'] = pts.copy()


def _consider_fourier(flat, best):
    """Update best from Fourier representation."""
    dim_shape = 2 * K_HARM + 1
    coeffs = flat[:dim_shape]
    pparams = flat[dim_shape:]
    pts = _fourier_points(coeffs, pparams)
    _consider(pts, best)


def _normalize_unit_area(points):
    """Scale so convex hull area = 1."""
    try:
        ha = ConvexHull(points).volume
        if ha > 1e-10:
            return points * np.sqrt(1.0 / ha)
    except Exception:
        pass
    xs, ys = points[:, 0], points[:, 1]
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        return points / np.sqrt(w * h)
    return points


def _generate_fourier_starts(deadline):
    """Generate diverse starting configurations in Fourier space.
    Returns list of flat arrays of shape (2*K_HARM + 1 + N,)."""
    starts = []
    dim_shape = 2 * K_HARM + 1
    base_pparams = 2 * np.pi * np.arange(N) / N
    
    # 1. Circle (R0=1, all harmonics zero)
    coeffs0 = _regular_fourier_coeffs(1.0)
    starts.append(np.concatenate([coeffs0, base_pparams]))
    
    # 2. Circle with different R0
    for R0 in [0.8, 1.2, 1.5]:
        if time.time() > deadline - 40:
            break
        coeffs0 = _regular_fourier_coeffs(R0)
        starts.append(np.concatenate([coeffs0, base_pparams]))
    
    # 3. Ellipse-like (first harmonic)
    rng = np.random.default_rng(42)
    for trial in range(15):
        if time.time() > deadline - 40:
            break
        coeffs = np.zeros(dim_shape)
        coeffs[0] = 1.0
        # Small first harmonic (ellipse)
        coeffs[1] = rng.uniform(-0.15, 0.15)  # a1
        coeffs[2] = rng.uniform(-0.15, 0.15)  # b1
        # Very small higher harmonics
        for k in range(2, K_HARM + 1):
            coeffs[2*k-1] = rng.normal(0, 0.03)
            coeffs[2*k] = rng.normal(0, 0.03)
        # Slightly perturbed point positions
        pparams = base_pparams + rng.normal(0, 0.05, N)
        starts.append(np.concatenate([coeffs, pparams]))
    
    # 4. More diverse harmonic perturbations
    for trial in range(20):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(100 + trial)
        coeffs = np.zeros(dim_shape)
        coeffs[0] = 1.0
        for k in range(1, K_HARM + 1):
            coeffs[2*k-1] = rng.normal(0, 0.08 / k)
            coeffs[2*k] = rng.normal(0, 0.08 / k)
        pparams = base_pparams + rng.normal(0, 0.1, N)
        starts.append(np.concatenate([coeffs, pparams]))
    
    # 5. Random point positions on circle
    for trial in range(15):
        if time.time() > deadline - 40:
            break
        rng = np.random.default_rng(200 + trial)
        coeffs = _regular_fourier_coeffs(1.0)
        pparams = np.sort(rng.uniform(0, 2*np.pi, N))
        starts.append(np.concatenate([coeffs, pparams]))
    
    # 6. Alternating gap patterns
    for ratio in [1.3, 1.7, 2.2]:
        if time.time() > deadline - 40:
            break
        gaps = np.ones(N)
        gaps[::2] *= ratio
        gaps /= np.sum(gaps) * 2 * np.pi
        pparams = np.cumsum(gaps)
        pparams[0] = 0
        coeffs = _regular_fourier_coeffs(1.0)
        starts.append(np.concatenate([coeffs, pparams]))
    
    return starts


def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin
    best = {'score': 0.0, 'pts': None}
    
    # Evaluate regular polygon first
    reg_pts = _regular_polygon(N)
    _consider(reg_pts, best)
    
    # Phase 1: Fourier-harmonic optimization (new structural approach)
    starts = _generate_fourier_starts(deadline)
    
    for idx, flat0 in enumerate(starts):
        if time.time() > deadline - 20:
            break
        
        # Method 1: Nelder-Mead on Fourier parameters
        if time.time() < deadline - 25:
            r = _nm_optimize(flat0.copy(), deadline, _fourier_objective, maxiter=2000)
            if r is not None and np.isfinite(r.fun):
                _consider_fourier(r.x, best)
        
        # Method 2: Powell on Fourier parameters
        if time.time() < deadline - 25:
            r = _powell_optimize(flat0.copy(), deadline, _fourier_objective, maxiter=300)
            if r is not None and np.isfinite(r.fun):
                _consider_fourier(r.x, best)
        
        # Method 3: L-BFGS on Fourier parameters
        if time.time() < deadline - 25:
            r = _lbfgs_optimize(flat0.copy(), deadline, _fourier_objective, maxiter=800)
            if r is not None and np.isfinite(r.fun):
                _consider_fourier(r.x, best)
    
    # Phase 2: Refine best in direct 2D space
    if best['pts'] is not None and time.time() < deadline - 10:
        flat0 = best['pts'].flatten()
        r = _nm_optimize(flat0.copy(), deadline, _fourier_objective_2d, maxiter=3000)
        if r is not None and np.isfinite(r.fun):
            _consider(r.x.reshape(N, 2), best)
    
    # Phase 3: Final polish with Powell in 2D
    if best['pts'] is not None and time.time() < deadline - 5:
        flat0 = best['pts'].flatten()
        r = _powell_optimize(flat0.copy(), deadline, _fourier_objective_2d, maxiter=500)
        if r is not None and np.isfinite(r.fun):
            _consider(r.x.reshape(N, 2), best)
    
    # Fallback: ensure we return something valid
    if best['pts'] is None or best['score'] < 1e-6:
        best['pts'] = _regular_polygon(N)
        best['score'] = _score(best['pts'])
    
    return _normalize_unit_area(best['pts'])


def _fourier_objective_2d(flat):
    """Objective for direct 2D point optimization."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    s = _score(pts)
    if s < 1e-12:
        return 1e10
    return -s
