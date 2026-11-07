"""Second autocorrelation inequality - combined tent-basis + full-dim polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _compute_value(h):
    """Compute the objective value for given heights."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or not np.all(np.isfinite(h)):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 * norm_inf < 1e-300:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def _h_from_tent(params):
    """Piecewise linear tent basis with K variable knots.
    params: [knot_x_0, ..., knot_x_{K-1}, amp_0, ..., amp_{K-1}]
    K knots with amplitudes. h(x) = sum_j amp_j * max(0, 1 - |x - knot_x_j| / bw_j)
    where bw_j is distance to nearest neighbor knot (or fixed half-width at boundaries).
    Ensures strict non-negativity.
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
    if h.sum() < 1e-10:
        h = np.ones(N) / N
    return h

def _neg_value_tent(params):
    """Negative value for tent parameter optimization."""
    h = _h_from_tent(params)
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

def _h_from_gauss(params):
    """Single Gaussian: [amplitude, width, center, floor]"""
    a, b, c, d = params
    x = np.linspace(-0.25, 0.25, N)
    h = a * np.exp(-b * (x - c)**2) + d
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        h = np.ones(N) / N
    return h

def _neg_value_gauss(params):
    h = _h_from_gauss(params)
    return -_compute_value(h)

def _h_from_bimodal(params):
    """Two Gaussians: [a1, b1, c1, a2, b2, c2]"""
    a1, b1, c1, a2, b2, c2 = params
    x = np.linspace(-0.25, 0.25, N)
    h = a1 * np.exp(-b1 * (x - c1)**2) + a2 * np.exp(-b2 * (x - c2)**2)
    h = np.maximum(h, 0.0)
    if h.sum() < 1e-10:
        h = np.ones(N) / N
    return h

def _neg_value_bimodal(params):
    h = _h_from_bimodal(params)
    return -_compute_value(h)

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _compute_value(best_h)

    rng = np.random.default_rng(42)

    def _try_optimize_tent(x0, bounds, maxiter_nm=2000, maxiter_lbfgs=500):
        """Optimize tent parameters with Nelder-Mead then L-BFGS-B polish."""
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        try:
            res_nm = minimize(_neg_value_tent, x0, method='Nelder-Mead',
                             options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            cur_params = res_nm.x
            cur_val = -res_nm.fun
            
            if time.time() < deadline - 15:
                try:
                    res_lbfgs = minimize(_neg_value_tent, cur_params, method='L-BFGS-B',
                                        bounds=bounds,
                                        options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10,
                                                 'maxfun': 5000})
                    if -res_lbfgs.fun > cur_val:
                        cur_params = res_lbfgs.x
                        cur_val = -res_lbfgs.fun
                except Exception:
                    pass
            
            if cur_val > best_val:
                best_val = cur_val
                best_h = _h_from_tent(cur_params)
        except Exception:
            pass

    def _try_optimize_log(params0, maxiter_nm=1500, maxiter_lbfgs=500):
        """Optimize log-params with Nelder-Mead then L-BFGS-B polish."""
        nonlocal best_h, best_val
        if time.time() > deadline - 10:
            return
        try:
            res_nm = minimize(_neg_value_log, params0, method='Nelder-Mead',
                             options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            cur_params = res_nm.x
            cur_val = -res_nm.fun
            
            if time.time() < deadline - 15:
                try:
                    res_lbfgs = minimize(_neg_value_log, cur_params, method='L-BFGS-B',
                                        options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10,
                                                 'maxfun': 5000})
                    if -res_lbfgs.fun > cur_val:
                        cur_params = res_lbfgs.x
                        cur_val = -res_lbfgs.fun
                except Exception:
                    pass
            
            if cur_val > best_val:
                best_val = cur_val
                best_h = _h_from_logparams(cur_params)
        except Exception:
            pass

    def _try_optimize_gauss(x0, bounds, maxiter_nm=1000, maxiter_lbfgs=300):
        """Optimize Gaussian parameters."""
        nonlocal best_h, best_val
        if time.time() > deadline - 5:
            return
        try:
            res_nm = minimize(_neg_value_gauss, x0, method='Nelder-Mead',
                             options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            cur_params = res_nm.x
            cur_val = -res_nm.fun
            
            if time.time() < deadline - 10:
                try:
                    res_lbfgs = minimize(_neg_value_gauss, cur_params, method='L-BFGS-B',
                                        bounds=bounds,
                                        options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10,
                                                 'maxfun': 3000})
                    if -res_lbfgs.fun > cur_val:
                        cur_params = res_lbfgs.x
                        cur_val = -res_lbfgs.fun
                except Exception:
                    pass
            
            if cur_val > best_val:
                best_val = cur_val
                best_h = _h_from_gauss(cur_params)
        except Exception:
            pass

    def _try_optimize_bimodal(x0, bounds, maxiter_nm=1500, maxiter_lbfgs=400):
        """Optimize bimodal Gaussian parameters."""
        nonlocal best_h, best_val
        if time.time() > deadline - 8:
            return
        try:
            res_nm = minimize(_neg_value_bimodal, x0, method='Nelder-Mead',
                             options={'maxiter': maxiter_nm, 'xatol': 1e-8, 'fatol': 1e-10,
                                      'adaptive': True})
            cur_params = res_nm.x
            cur_val = -res_nm.fun
            
            if time.time() < deadline - 12:
                try:
                    res_lbfgs = minimize(_neg_value_bimodal, cur_params, method='L-BFGS-B',
                                        bounds=bounds,
                                        options={'maxiter': maxiter_lbfgs, 'ftol': 1e-12, 'gtol': 1e-10,
                                                 'maxfun': 4000})
                    if -res_lbfgs.fun > cur_val:
                        cur_params = res_lbfgs.x
                        cur_val = -res_lbfgs.fun
                except Exception:
                    pass
            
            if cur_val > best_val:
                best_val = cur_val
                best_h = _h_from_bimodal(cur_params)
        except Exception:
            pass

    # Tent parameter bounds: knots in [-0.25, 0.25], amps in [0, 1]
    K = 6
    tent_bounds = [(-0.25, 0.25)] * K + [(0, 1)] * K

    # Gaussian bounds: amplitude [0,1], width [1,20], center [-0.25,0.25], floor [0,0.1]
    gauss_bounds = [(0, 1), (1, 20), (-0.25, 0.25), (0, 0.1)]

    # Bimodal bounds
    bimodal_bounds = [(0, 1), (1, 20), (-0.25, 0.25), (0, 1), (1, 20), (-0.25, 0.25)]

    # Phase 1: Tent-based optimization (low-dimensional, can model sharp structures)
    # Multiple starting points for tent
    tent_starts = []
    # Symmetric 2-knot
    tent_starts.append([-0.1, 0.1, 1.0, 1.0] + [0.0] * (2*K - 4))
    # Symmetric 3-knot
    tent_starts.append([-0.15, 0.0, 0.15, 0.8, 1.0, 0.8] + [0.0] * (2*K - 6))
    # Asymmetric 2-knot
    tent_starts.append([-0.15, 0.05, 1.0, 0.7] + [0.0] * (2*K - 4))
    # 4-knot spread
    tent_starts.append([-0.2, -0.07, 0.07, 0.2, 0.9, 0.9, 0.9, 0.9] + [0.0] * (2*K - 8))
    # Random tent starts
    for _ in range(5):
        knots = np.sort(rng.uniform(-0.22, 0.22, K))
        amps = rng.uniform(0.3, 1.0, K)
        tent_starts.append(np.concatenate([knots, amps]))

    for ts in tent_starts:
        if time.time() > deadline - 30:
            break
        _try_optimize_tent(ts, tent_bounds)

    # Phase 2: Parametric families (Gaussian, bimodal)
    gauss_starts = [
        [1.0, 5.0, 0.0, 0.0],      # centered, medium width
        [1.0, 10.0, 0.0, 0.0],     # centered, narrow
        [1.0, 3.0, 0.0, 0.05],     # centered, wide, with floor
        [0.8, 8.0, 0.05, 0.0],     # slightly off-center
        [0.8, 8.0, -0.05, 0.0],    # slightly off-center other way
    ]
    for gs in gauss_starts:
        if time.time() > deadline - 25:
            break
        _try_optimize_gauss(gs, gauss_bounds)

    bimodal_starts = [
        [0.8, 10.0, -0.1, 0.8, 10.0, 0.1],   # symmetric bimodal
        [1.0, 8.0, -0.12, 0.6, 12.0, 0.08],  # asymmetric bimodal
        [0.9, 12.0, -0.15, 0.9, 12.0, 0.15], # wider bimodal
        [1.0, 6.0, -0.08, 1.0, 6.0, 0.08],   # narrow bimodal
    ]
    for bs in bimodal_starts:
        if time.time() > deadline - 20:
            break
        _try_optimize_bimodal(bs, bimodal_bounds)

    # Phase 3: Full-dimensional log-param polish from best solution
    if time.time() < deadline - 20:
        best_params = np.log(best_h + 1e-15)
        _try_optimize_log(best_params, maxiter_nm=1000, maxiter_lbfgs=800)

    # Phase 4: Additional random restarts in full-dim if time permits
    extra = 0
    while time.time() < deadline - 15 and extra < 10:
        h0 = np.abs(rng.standard_normal(N)) + 0.01
        h0 = h0 / h0.sum()
        params0 = np.log(h0 + 1e-15)
        _try_optimize_log(params0, maxiter_nm=800, maxiter_lbfgs=400)
        extra += 1

    # Phase 5: Final high-precision polish on the best solution
    if time.time() < deadline - 5:
        best_params = np.log(best_h + 1e-15)
        try:
            res_final = minimize(_neg_value_log, best_params, method='L-BFGS-B',
                                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12,
                                         'maxfun': 10000})
            if -res_final.fun > best_val:
                best_val = -res_final.fun
                best_h = _h_from_logparams(res_final.x)
        except Exception:
            pass

    # Ensure valid output
    best_h = np.asarray(best_h, dtype=float)
    best_h = np.maximum(best_h, 0.0)
    if best_h.sum() < 1e-15:
        best_h = np.ones(N) / N
    else:
        best_h = best_h / best_h.sum()

    return list(best_h)
# EVOLVE-BLOCK-END
