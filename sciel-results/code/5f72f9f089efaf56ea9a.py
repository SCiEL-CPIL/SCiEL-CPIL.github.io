"""
Model-and-optimize approach for the second autocorrelation inequality.

Key structural facts:
  * h >= 0, so g = conv(h,h) >= 0.
  * norm_1 = sum(g)/(m+1) = sum(h)^2/(m+1).
  * norm_inf = max(g).
  * The objective is scale-invariant: value depends only on shape of h.
  * We want to maximize l2 / (norm_1 * norm_inf).

Strategy:
  1. Analytical plateau family gives a strong baseline (value ~1.35+ for k~15).
  2. Parameterize h as a smooth profile (e.g., cosine bump with adjustable
     width and peak, or a sum of Gaussians) and fit parameters directly.
  3. Use scipy.optimize with the exact objective to polish.
  4. Multi-start with diverse parameter initializations.
"""
import numpy as np
import time
import scipy.optimize as opt

N = 50
M = 2 * N - 1          # len(g) = 99
W = 1.0 / (M + 1)      # 1/100
TIME_LIMIT = 150.0
RNG = np.random.RandomState(50)


def _build_y(g):
    y = np.zeros(M + 2)
    y[1:-1] = g
    return y


def compute_value(h):
    """Exact evaluator metric for given heights."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)) or np.any(h < -1e-12):
        return 0.0
    h = np.maximum(0.0, h)
    g = np.convolve(h, h)
    m = len(g)
    y = _build_y(g)
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _neg_value(h):
    """Negative exact value for minimization."""
    h = np.maximum(0.0, h)
    return -compute_value(h)


def _plateau_profile(h):
    """
    Parameterized smooth profile: h_i = A * max(0, cos(pi * (x_i - center) / width))^p
    Parameters: [A, center, width, p]
    """
    A, center, width, p = h
    A = max(0.0, A)
    center = center  # in [0, N-1]
    width = max(1e-6, width)  # half-width in index units
    p = max(0.5, p)
    
    x = np.arange(N, dtype=float)
    t = np.abs(x - center) / width
    val = np.where(t <= 1.0, A * np.cos(0.5 * np.pi * t) ** p, 0.0)
    return val


def _bump_profile(params):
    """
    Sum of two Gaussian bumps.
    params: [A1, c1, s1, A2, c2, s2]
    """
    A1, c1, s1, A2, c2, s2 = params
    A1 = max(0.0, A1)
    A2 = max(0.0, A2)
    s1 = max(0.5, s1)
    s2 = max(0.5, s2)
    x = np.arange(N, dtype=float)
    h = A1 * np.exp(-0.5 * ((x - c1) / s1) ** 2) + A2 * np.exp(-0.5 * ((x - c2) / s2) ** 2)
    return h


def _flat_top_profile(params):
    """
    Flat-top with cosine taper edges.
    params: [A, center, flat_halfwidth, taper_width]
    """
    A, center, flat_hw, taper_w = params
    A = max(0.0, A)
    flat_hw = max(0.5, flat_hw)
    taper_w = max(0.5, taper_w)
    x = np.arange(N, dtype=float)
    dist = np.abs(x - center)
    # Inside flat region
    mask_flat = dist <= flat_hw
    # Taper region
    mask_taper = (dist > flat_hw) & (dist <= flat_hw + taper_w)
    h = np.zeros(N)
    h[mask_flat] = A
    t = (dist[mask_taper] - flat_hw) / taper_w
    h[mask_taper] = A * 0.5 * (1 + np.cos(np.pi * t))
    return h


def _optimize_flat_top(p0, bounds, deadline):
    """Optimize the flat-top parameterization."""
    def obj(params):
        if time.time() > deadline:
            return 1e10
        h = _flat_top_profile(params)
        return -compute_value(h)
    
    try:
        res = opt.minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x, res.fun
    except Exception:
        return p0, 1e10


def _dct_basis(N, n_modes):
    """Return DCT-II basis matrix of shape (N, n_modes)."""
    k = np.arange(N)
    n = np.arange(n_modes)
    # DCT-II: cos(pi * (2k+1) * n / (2N))
    basis = np.cos(np.pi * np.outer(2 * k + 1, n) / (2 * N))
    return basis


def _dct_profile(coeffs, n_modes):
    """
    Reconstruct h from DCT coefficients.
    coeffs: [a0, a1, ..., a_{n_modes-1}] where h_k = sum_j a_j * basis[j,k]
    We enforce nonnegativity via softplus or clipping.
    """
    basis = _dct_basis(N, n_modes)
    h = basis @ coeffs
    return h


def _optimize_bump(p0, bounds, deadline):
    """Optimize the two-bump parameterization."""
    def obj(params):
        if time.time() > deadline:
            return 1e10
        h = _bump_profile(params)
        return -compute_value(h)
    
    try:
        res = opt.minimize(obj, p0, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        return res.x, res.fun
    except Exception:
        return p0, 1e10


def construct():
    start_time = time.time()
    deadline = start_time + (TIME_LIMIT - 10.0)  # safety margin

    best_val = 0.0
    best_h = np.ones(N)

    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, h)
        if not np.all(np.isfinite(h)):
            return
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()

    # --- Phase 0: Analytical plateau baseline ---
    # For a plateau of width k: value(k) = (m+1)*(4k^2+1)/(15*k)
    # This is maximized at large k but the plateau must fit in N=50.
    # Actually for uniform h (all ones), g is triangular, value is known.
    # Let's just evaluate several plateaus.
    for k in range(5, N + 1):
        h = np.zeros(N)
        center = N // 2
        start = max(0, center - k // 2)
        end = min(N, start + k)
        h[start:end] = 1.0
        track(h)

    # Evaluate all-ones
    track(np.ones(N))

    # --- Phase 1: Optimize flat-top parameterization ---
    ft_bounds = [
        (0.1, 10.0),       # A
        (N * 0.1, N * 0.9),  # center
        (1.0, N * 0.4),    # flat_halfwidth
        (0.5, N * 0.3),    # taper_width
    ]

    ft_starts = [
        [1.0, N / 2.0, 10.0, 5.0],
        [1.0, N / 2.0, 15.0, 8.0],
        [1.0, N / 2.0, 20.0, 10.0],
        [1.5, N / 2.0, 8.0, 4.0],
        [0.5, N / 2.0, 25.0, 12.0],
    ]
    
    for s in ft_starts:
        if time.time() > deadline - 20:
            break
        p0 = np.array(s, dtype=float)
        res_x, res_f = _optimize_flat_top(p0, ft_bounds, deadline)
        if time.time() > deadline - 5:
            break
        h = _flat_top_profile(res_x)
        track(h)

    # --- Phase 2: Optimize two-bump parameterization ---
    bump_bounds = [
        (0.1, 5.0),        # A1
        (N * 0.1, N * 0.9),  # c1
        (1.0, N * 0.4),    # s1
        (0.1, 5.0),        # A2
        (N * 0.1, N * 0.9),  # c2
        (1.0, N * 0.4),    # s2
    ]

    bump_starts = [
        [1.0, N * 0.3, 5.0, 1.0, N * 0.7, 5.0],
        [1.0, N * 0.25, 7.0, 1.0, N * 0.75, 7.0],
        [1.5, N * 0.3, 4.0, 1.5, N * 0.7, 4.0],
        [1.0, N * 0.4, 10.0, 1.0, N * 0.6, 10.0],
    ]

    for s in bump_starts:
        if time.time() > deadline - 20:
            break
        p0 = np.array(s, dtype=float)
        res_x, res_f = _optimize_bump(p0, bump_bounds, deadline)
        if time.time() > deadline - 5:
            break
        h = _bump_profile(res_x)
        track(h)

    # --- Phase 3: DCT-based low-dimensional optimization ---
    n_modes_list = [5, 8, 12]
    
    for n_modes in n_modes_list:
        if time.time() > deadline - 20:
            break
        
        basis = _dct_basis(N, n_modes)
        
        # Initial coefficients: project best_h onto DCT basis
        init_coeffs = basis.T @ best_h
        # Add some random starts
        dct_starts = [
            init_coeffs.copy(),
            init_coeffs + RNG.normal(0, 0.1, n_modes),
            init_coeffs * 0.5 + RNG.normal(0, 0.5, n_modes),
            np.zeros(n_modes),
            np.ones(n_modes) * 0.5,
        ]
        
        for start in dct_starts:
            if time.time() > deadline - 10:
                break
            
            def dct_obj(coeffs, n_m=n_modes, basis_mat=basis):
                h = basis_mat @ coeffs
                h = np.maximum(0.0, h)
                return -compute_value(h)
            
            try:
                res = opt.minimize(
                    dct_obj,
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                h_dct = basis @ res.x
                h_dct = np.maximum(0.0, h_dct)
                track(h_dct)
            except Exception:
                pass

    # --- Phase 4: Nelder-Mead polish on exact objective ---
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        scale = 0.03 * (1.0 + 0.5 * RNG.rand())
        pert = best_h + RNG.normal(0.0, scale, N)
        pert = np.maximum(0.0, pert)
        try:
            res = opt.minimize(
                _neg_value,
                pert,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            track(res.x)
        except Exception:
            pass

    # --- Phase 5: Final DCT polish with more modes ---
    n_modes_final = 15
    basis_final = _dct_basis(N, n_modes_final)
    init_coeffs_final = basis_final.T @ best_h
    
    def dct_obj_final(coeffs):
        h = basis_final @ coeffs
        h = np.maximum(0.0, h)
        return -compute_value(h)
    
    try:
        res_final = opt.minimize(
            dct_obj_final,
            init_coeffs_final,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
        )
        h_final = basis_final @ res_final.x
        h_final = np.maximum(0.0, h_final)
        track(h_final)
    except Exception:
        pass

    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
