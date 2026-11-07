"""Second autocorrelation inequality - optimized combined approach."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

# DCT-based parameterization for global shape optimization
def _h_from_dct_lowdim(params, K=10):
    """Reconstruct h from first K DCT-II coefficients.
    h[n] = sum_{k=0}^{K-1} c_k * cos(pi*(n+0.5)*k/N)
    Then clip to non-negative and normalize.
    """
    x = np.arange(N)
    h = np.zeros(N)
    for k in range(K):
        c_k = params[k]
        if abs(c_k) > 1e-12:
            h += c_k * np.cos(np.pi * (x + 0.5) * k / N)
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        h = np.ones(N) / N
    return h

def _neg_value_dct(params):
    """Negative value for DCT coefficient optimization."""
    K = 10
    h = _h_from_dct_lowdim(params, K)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

# DCT-based parameterization
# Use DCT-II forward to get coefficients, then inverse DCT-II to reconstruct
# h = IDCT(a) where a is the coefficient vector
# We use only the first K coefficients (low-dimensional) for optimization

def _h_from_dct(coeffs, K=12):
    """Reconstruct h from DCT coefficients.
    coeffs: array of length K (first K DCT coefficients)
    h = IDCT_II(coeffs_padded) where coeffs_padded has zeros after K
    Then normalize and ensure non-negativity.
    """
    # Full DCT-II coefficient vector would be length N
    # We optimize only first K, rest are zero
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    
    # Inverse DCT-II: h[n] = sum_k c_k * cos(pi*(n+0.5)*k/N)
    # Using scipy's idct with type=2
    h = np.zeros(N)
    for k in range(K):
        if full_coeffs[k] != 0:
            h += full_coeffs[k] * np.cos(np.pi * (np.arange(N) + 0.5) * k / N)
    
    # Ensure non-negativity
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        h = np.ones(N) / N
    return h

def _neg_value_dct(params):
    """Optimize first K DCT coefficients."""
    K = 12
    h = _h_from_dct(params[:K], K)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

def _h_from_logparams(params):
    """params: raw values, h = exp(params) normalized to sum=1 (positivity guaranteed)"""
    e = np.exp(params - np.max(params))
    h = e / e.sum()
    return h

def _neg_value_log(params):
    h = _h_from_logparams(params)
    return -_compute_value(h)

def _neg_value_gauss(params):
    """Single Gaussian: [amplitude, width, center, floor]"""
    a, b, c, d = params
    x = np.linspace(-0.25, 0.25, N)
    h = a * np.exp(-b * (x - c)**2) + d
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

def _neg_value_bimodal(params):
    """Two Gaussians: [a1, b1, c1, a2, b2, c2]"""
    a1, b1, c1, a2, b2, c2 = params
    x = np.linspace(-0.25, 0.25, N)
    h = a1 * np.exp(-b1 * (x - c1)**2) + a2 * np.exp(-b2 * (x - c2)**2)
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

def _neg_value_poly(params):
    """Polynomial: [a0, a1, a2, a3, a4]"""
    x = np.linspace(-0.25, 0.25, N)
    h = params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

def _h_from_tent(params):
    """Piecewise linear tent basis.
    params: [knot_x_0, ..., knot_x_{K-1}, amp_0, ..., amp_{K-1}]
    K knots with amplitudes. h(x) = sum_j amp_j * max(0, 1 - |x - knot_x_j| / bw_j)
    where bw_j is distance to nearest neighbor knot (or fixed half-width at boundaries).
    """
    K = 6  # number of tent centers
    knot_xs = params[:K]
    amps = params[K:]
    x = np.linspace(-0.25, 0.25, N)
    h = np.zeros(N)
    # Sort knots for bandwidth computation
    sorted_knots = np.sort(knot_xs)
    for j in range(K):
        # Bandwidth: half distance to nearest neighbor
        if j == 0:
            if K > 1:
                bw = max(0.01, (sorted_knots[1] - sorted_knots[0]) / 2.0)
            else:
                bw = 0.25
        elif j == K - 1:
            if K > 1:
                bw = max(0.01, (sorted_knots[-1] - sorted_knots[-2]) / 2.0)
            else:
                bw = 0.25
        else:
            left_bw = (sorted_knots[j] - sorted_knots[j-1]) / 2.0
            right_bw = (sorted_knots[j+1] - sorted_knots[j]) / 2.0
            bw = max(0.01, min(left_bw, right_bw))
        # Tent: triangular bump centered at knot_xs[j] with base width 2*bw
        t = 1.0 - np.abs(x - knot_xs[j]) / bw
        t = np.maximum(t, 0.0)
        h += amps[j] * t
    h = np.maximum(h, 0.0)
    return h

def _neg_value_tent(params):
    h = _h_from_tent(params)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

def _h_from_spline(params):
    """Cubic spline through K control points.
    params: [cp_x_0, ..., cp_x_{K-1}, cp_y_0, ..., cp_y_{K-1}]
    """
    from scipy.interpolate import CubicSpline
    K = 7
    cp_x = params[:K]
    cp_y = params[K:]
    # Ensure sorted x
    order = np.argsort(cp_x)
    cp_x_sorted = cp_x[order]
    cp_y_sorted = cp_y[order]
    # Remove duplicates
    mask = np.concatenate(([True], np.diff(cp_x_sorted) > 1e-8))
    cp_x_sorted = cp_x_sorted[mask]
    cp_y_sorted = cp_y_sorted[mask]
    if len(cp_x_sorted) < 2:
        return np.ones(N) * 0.01
    try:
        cs = CubicSpline(cp_x_sorted, cp_y_sorted, bc_type='clamped')
        x = np.linspace(-0.25, 0.25, N)
        h = cs(x)
        h = np.maximum(h, 0.0)
        return h
    except Exception:
        return np.ones(N) * 0.01

def _neg_value_spline(params):
    h = _h_from_spline(params)
    if h.sum() < 1e-10:
        return 0.0
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    def try_optimize_log(params0, maxiter, xatol, fatol):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            res = minimize(_neg_value_log, params0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol,
                                   'adaptive': True})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_logparams(res.x)
        except Exception:
            pass

    def try_bfgs_log(params0, maxiter):
        nonlocal best_h, best_val
        if time.time() > deadline - 3:
            return
        try:
            res = minimize(_neg_value_log, params0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
            if res.fun < -best_val:
                best_val = -res.fun
                best_h = _h_from_logparams(res.x)
        except Exception:
            pass

    def try_parametric(neg_func, x0, bounds, maxiter_nm=1500, maxiter_lbfgs=400):
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            res_nm = minimize(neg_func, x0, method='Nelder-Mead',
                             options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            best_x = res_nm.x
            best_f = res_nm.fun
            
            if time.time() < deadline - 5:
                try:
                    res_lbfgs = minimize(neg_func, best_x, method='L-BFGS-B',
                                        bounds=bounds,
                                        options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10})
                    if res_lbfgs.fun < best_f:
                        best_x = res_lbfgs.x
                        best_f = res_lbfgs.fun
                except Exception:
                    pass
            
            if best_f < -best_val:
                best_val = -best_f
                # Reconstruct h from the parametric form
                if neg_func is _neg_value_gauss:
                    a, b, c, d = best_x
                    x = np.linspace(-0.25, 0.25, N)
                    best_h = np.maximum(a * np.exp(-b * (x - c)**2) + d, 0.0)
                elif neg_func is _neg_value_bimodal:
                    a1, b1, c1, a2, b2, c2 = best_x
                    x = np.linspace(-0.25, 0.25, N)
                    best_h = np.maximum(a1 * np.exp(-b1 * (x - c1)**2) + a2 * np.exp(-b2 * (x - c2)**2), 0.0)
                elif neg_func is _neg_value_poly:
                    x = np.linspace(-0.25, 0.25, N)
                    best_h = np.maximum(best_x[0] + best_x[1]*x + best_x[2]*x**2 + best_x[3]*x**3 + best_x[4]*x**4, 0.0)
                elif neg_func is _neg_value_tent:
                    best_h = _h_from_tent(best_x)
                elif neg_func is _neg_value_spline:
                    best_h = _h_from_spline(best_x)
        except Exception:
            pass

    rng = np.random.default_rng(50)
    x = np.linspace(-1, 1, N)
    x_half = np.linspace(-0.5, 0.5, N)
    x_quarter = np.linspace(-0.25, 0.25, N)

    # Diverse initial shapes for log-parametric optimization
    candidates = []
    candidates.append(np.ones(N))
    # Gaussians with various widths
    for a in [0.5, 1, 2, 4, 8, 16, 32, 64]:
        candidates.append(np.exp(-a * x_quarter**2))
    # Polynomials
    for k in [2, 4, 6, 8]:
        candidates.append(np.maximum(0, 1 - np.abs(x_quarter)**k))
    # Cosines
    for freq in [1, 2, 3, 4]:
        candidates.append(np.maximum(0, np.cos(freq * np.pi * x_half)))
    # Sinc-like
    candidates.append(np.sinc(x) ** 2)
    # Bimodal
    for d in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        candidates.append(np.exp(-8*(x_quarter-d)**2) + np.exp(-8*(x_quarter+d)**2))
    # Plateau
    plateau = np.zeros(N)
    plateau[10:40] = 1.0
    candidates.append(plateau)
    # Triangle
    candidates.append(np.maximum(0, 1 - 2 * np.abs(x_half)))
    # Asymmetric
    candidates.append(np.exp(-x_quarter**2) + 0.3 * np.exp(-8*(x_quarter-0.15)**2))
    candidates.append(np.exp(-x_quarter**2) + 0.3 * np.exp(-8*(x_quarter+0.15)**2))
    # Random
    for _ in range(15):
        candidates.append(np.abs(rng.standard_normal(N)) + 0.05)

    # Phase 1: Nelder-Mead from each candidate (log-parametric)
    for h0 in candidates:
        if time.time() > deadline - 30:
            break
        h0 = np.asarray(h0, dtype=float)
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize_log(params0, 2000, 1e-7, 1e-9)

    # Phase 2: Parametric optimization with different forms
    # Single Gaussian
    bounds_gauss = [(0.01, 10.0), (1.0, 200.0), (-0.25, 0.25), (-5.0, 5.0)]
    gauss_starts = [
        [1.0, 10.0, 0.0, 0.0],
        [1.0, 20.0, 0.0, 0.0],
        [1.0, 40.0, 0.0, 0.0],
        [1.0, 80.0, 0.0, 0.0],
        [1.0, 160.0, 0.0, 0.0],
        [2.0, 15.0, 0.0, 0.0],
        [1.0, 30.0, 0.05, 0.0],
        [1.0, 30.0, -0.05, 0.0],
    ]
    for x0 in gauss_starts:
        if time.time() > deadline - 20:
            break
        try_parametric(_neg_value_gauss, x0, bounds_gauss, 1000, 300)

    # Bimodal
    bounds_bimodal = [(0.01, 10.0), (1.0, 200.0), (-0.25, 0.25),
                      (0.01, 10.0), (1.0, 200.0), (-0.25, 0.25)]
    bimodal_starts = [
        [1.0, 15.0, -0.1, 1.0, 15.0, 0.1],
        [1.0, 20.0, -0.15, 1.0, 20.0, 0.15],
        [1.0, 30.0, -0.12, 1.0, 30.0, 0.12],
        [1.0, 50.0, -0.1, 1.0, 50.0, 0.1],
        [1.0, 80.0, -0.08, 1.0, 80.0, 0.08],
        [2.0, 25.0, -0.1, 2.0, 25.0, 0.1],
    ]
    for x0 in bimodal_starts:
        if time.time() > deadline - 15:
            break
        try_parametric(_neg_value_bimodal, x0, bounds_bimodal, 800, 250)

    # Polynomial
    bounds_poly = [(-5.0, 5.0)] * 5
    poly_starts = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 5.0, 0.0, 0.0],
        [1.0, 0.0, 10.0, 0.0, 0.0],
        [1.0, 0.0, 20.0, 0.0, 0.0],
        [0.5, 0.0, 8.0, 0.0, 2.0],
        [1.0, 0.0, 15.0, 0.0, -3.0],
    ]
    for x0 in poly_starts:
        if time.time() > deadline - 10:
            break
        try_parametric(_neg_value_poly, x0, bounds_poly, 600, 200)

    # Phase 2b: Tent (piecewise linear) optimization
    # 6 tent centers with amplitudes: 12 params total
    tent_bounds = [(-0.24, 0.24)] * 6 + [(0.01, 10.0)] * 6
    tent_starts = [
        # Symmetric single peak
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        # Two peaks
        [-0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 1.0, 1.5, 1.5, 1.5, 1.5, 1.0],
        # Three peaks
        [-0.15, 0.0, 0.15, 0.05, -0.05, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0],
        # Sharp central peak
        [-0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.5, 0.8, 3.0, 0.8, 0.5, 0.3],
        # Broad plateau-like
        [-0.2, -0.1, 0.0, 0.1, 0.2, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
        # Asymmetric
        [-0.15, -0.08, -0.02, 0.05, 0.12, 0.2, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5],
    ]
    for x0 in tent_starts:
        if time.time() > deadline - 15:
            break
        try_parametric(_neg_value_tent, x0, tent_bounds, 1200, 300)

    # Phase 2c: Spline optimization
    # 7 control points: x positions and y values = 14 params
    spline_bounds = [(-0.24, 0.24)] * 7 + [(0.0, 10.0)] * 7
    spline_starts = [
        # Single smooth peak
        [-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.15, 0.5, 1.0, 2.0, 1.0, 0.5, 0.3, 0.3],
        # Two smooth peaks
        [-0.2, -0.1, -0.02, 0.02, 0.1, 0.2, -0.15, 0.5, 1.5, 0.5, 0.5, 1.5, 0.5, 0.3],
        # Broad hump
        [-0.2, -0.1, 0.0, 0.1, 0.2, -0.15, 0.15, 0.5, 1.0, 1.5, 1.0, 0.5, 0.3, 0.3],
        # Sharp peak
        [-0.15, -0.05, 0.0, 0.05, 0.15, -0.1, 0.1, 0.3, 0.8, 3.0, 0.8, 0.3, 0.2, 0.2],
    ]
    for x0 in spline_starts:
        if time.time() > deadline - 12:
            break
        try_parametric(_neg_value_spline, x0, spline_bounds, 1000, 250)

    # Phase 2d: DCT coefficient optimization (low-dimensional global representation)
    # Optimize first 10 DCT-II coefficients
    dct_bounds = [(-5.0, 5.0)] * 10
    dct_starts = [
        # Uniform (all zeros except DC)
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Slight curvature
        [1.0, 0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # More curvature
        [1.0, 0.0, -1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Asymmetric
        [1.0, 0.5, -0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        # Bimodal-like
        [0.5, 0.0, -1.0, 0.0, 1.0, 0.0, -0.5, 0.0, 0.0, 0.0],
        # Random
        [rng.normal(1.0, 0.5) for _ in range(10)],
        [rng.normal(1.0, 1.0) for _ in range(10)],
    ]
    for x0 in dct_starts:
        if time.time() > deadline - 10:
            break
        try_parametric(_neg_value_dct, x0, dct_bounds, 800, 200)

    # Phase 3: Polish best with L-BFGS-B
    if time.time() < deadline - 10:
        try_bfgs_log(np.log(best_h), 600)

    # Phase 4: More random restarts if time permits
    for i in range(20):
        if time.time() > deadline - 8:
            break
        if i % 3 == 0:
            h0 = np.abs(rng.standard_normal(N)) + 0.01
        elif i % 3 == 1:
            # Perturb best solution
            noise = rng.standard_normal(N) * 0.1
            h0 = np.exp(np.log(best_h) + noise)
        else:
            # Tent-based random restart
            K = 6
            knot_xs = rng.uniform(-0.22, 0.22, K)
            amps = np.abs(rng.standard_normal(K)) + 0.1
            params0_tent = np.concatenate([knot_xs, amps])
            h0 = _h_from_tent(params0_tent)
            if h0.sum() < 1e-10:
                h0 = np.abs(rng.standard_normal(N)) + 0.01
        
        h0 = np.maximum(h0, 1e-6)
        h0 = h0 / h0.sum()
        params0 = np.log(h0)
        try_optimize_log(params0, 1000, 1e-7, 1e-9)

    # Phase 5: Final polish with tighter tolerances
    if time.time() < deadline - 5:
        try_bfgs_log(np.log(best_h), 400)
        
    # Phase 5b: Tent/spline polish from best solution's shape
    if time.time() < deadline - 4:
        # Extract approximate tent params from best_h
        x_grid = np.linspace(-0.25, 0.25, N)
        # Use 6 evenly spaced knots with local max amplitudes
        K = 6
        knot_xs = np.linspace(-0.22, 0.22, K)
        amps = np.zeros(K)
        for j in range(K):
            # Local max near knot
            idx = np.argmin(np.abs(x_grid - knot_xs[j]))
            lo = max(0, idx - 3)
            hi = min(N, idx + 4)
            amps[j] = max(0.01, np.max(best_h[lo:hi]))
        tent_x0 = np.concatenate([knot_xs, amps])
        try_parametric(_neg_value_tent, tent_x0, tent_bounds, 800, 200)
    
    if time.time() < deadline - 3:
        # Spline from best
        K_sp = 7
        cp_x = np.linspace(-0.23, 0.23, K_sp)
        idxs = (np.linspace(0, N-1, K_sp)).astype(int)
        cp_y = best_h[idxs]
        spline_x0 = np.concatenate([cp_x, cp_y])
        try_parametric(_neg_value_spline, spline_x0, spline_bounds, 600, 150)

    # Phase 5c: DCT polish from best solution
    if time.time() < deadline - 2:
        # Compute DCT coefficients of best_h and use as starting point
        x = np.arange(N)
        dct_coeffs = np.zeros(10)
        for k in range(10):
            dct_coeffs[k] = np.sum(best_h * np.cos(np.pi * (x + 0.5) * k / N))
        # Normalize to reasonable scale
        dct_coeffs = dct_coeffs / max(1.0, np.max(np.abs(dct_coeffs))) * 1.5
        try_parametric(_neg_value_dct, dct_coeffs, dct_bounds, 500, 150)

    # Phase 6: One last Nelder-Mead on the best with very tight tolerance
    if time.time() < deadline - 2:
        try_optimize_log(np.log(best_h), 600, 1e-9, 1e-12)

    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    if best_h.sum() < 1e-10:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / best_h.sum()
    return list(best_h)
# EVOLVE-BLOCK-END
