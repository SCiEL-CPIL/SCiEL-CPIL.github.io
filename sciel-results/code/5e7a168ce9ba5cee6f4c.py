<<<<<<< SEARCH
def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Strategy 1: Boxcar representation with Differential Evolution
    # M boxcars, each with (amp, pos, width)
    M = 6
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 3.0))   # amplitude
        bounds.append((0.0, 1.0))   # position
        bounds.append((0.01, 0.5))  # width

    if time.time() < deadline - 40.0:
        try:
            de_result = differential_evolution(
                smooth_objective_boxcar,
                bounds,
                args=(0.01,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            if de_result.success and np.isfinite(de_result.fun):
                h_candidate = boxcar_to_heights(de_result.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
=======
def tent_to_heights(params):
    """
    Symmetric triangular (tent) pulse centered at 0.
    params: [base_half_width, apex_height]
    base_half_width in [0.05, 0.5] (fraction of half-domain = 0.25)
    apex_height in [0.5, 5.0]
    """
    half_w = params[0]
    apex = params[1]
    # base_half_width is fraction of the full domain half-width (0.25)
    # so actual half-width in units of [-0.25, 0.25] is half_w * 0.25
    actual_half_w = half_w * 0.25
    
    # Bin centers in [-0.25, 0.25]
    bin_centers = -0.25 + (np.arange(N) + 0.5) * W
    
    # Triangular: h(x) = apex * max(0, 1 - |x| / actual_half_w)
    heights = apex * np.maximum(0.0, 1.0 - np.abs(bin_centers) / actual_half_w)
    return np.clip(heights, 0.0, None)


def smooth_objective_tent(params, tau=0.01):
    """
    Smooth surrogate for tent pulse using log-sum-exp.
    """
    h = tent_to_heights(params)
    s = h.sum()
    if s < 1e-15:
        return 1e10

    f_fine = h[CELL]
    F = np.fft.rfft(f_fine, N_FFT)
    conv = np.fft.irfft(F * F, N_FFT)[:2 * FINE_N - 1]
    conv *= DX

    m = np.max(conv)
    if m < 1e-15:
        return 1e10
    lse = m + tau * np.log(np.sum(np.exp((conv - m) / tau)))

    integral_sq = (s * W) ** 2
    if integral_sq < 1e-15:
        return 1e10

    return lse / integral_sq


def construct():
    """
    Construct a nonnegative step function with 600 equal intervals on [-1/4, 1/4]
    to minimize the normalized maximum of its autoconvolution.
    """
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial best: uniform
    best_h = np.ones(N, dtype=np.float64)
    best_val = ratio_from_h(best_h)

    np.random.seed(42)

    # Strategy 0: Single symmetric tent pulse with Differential Evolution
    # params: [base_half_width, apex_height]
    tent_bounds = [(0.05, 0.5), (0.5, 5.0)]
    if time.time() < deadline - 50.0:
        try:
            de_tent = differential_evolution(
                smooth_objective_tent,
                tent_bounds,
                args=(0.01,),
                seed=42,
                maxiter=150,
                popsize=20,
                tol=1e-8,
                polish=True
            )
            if de_tent.success and np.isfinite(de_tent.fun):
                h_candidate = tent_to_heights(de_tent.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Also try a few hand-crafted tent candidates
    for hw in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
        for amp in [1.0, 1.5, 2.0, 2.5, 3.0]:
            h_candidate = tent_to_heights([hw, amp])
            val = ratio_from_h(h_candidate)
            if val < best_val and np.isfinite(val):
                best_val = val
                best_h = h_candidate.copy()

    # Strategy 1: Boxcar representation with Differential Evolution
    # M boxcars, each with (amp, pos, width)
    M = 6
    bounds = []
    for _ in range(M):
        bounds.append((0.0, 3.0))   # amplitude
        bounds.append((0.0, 1.0))   # position
        bounds.append((0.01, 0.5))  # width

    if time.time() < deadline - 40.0:
        try:
            de_result = differential_evolution(
                smooth_objective_boxcar,
                bounds,
                args=(0.01,),
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-6,
                polish=True
            )
            if de_result.success and np.isfinite(de_result.fun):
                h_candidate = boxcar_to_heights(de_result.x)
                val = ratio_from_h(h_candidate)
                if val < best_val and np.isfinite(val):
                    best_val = val
                    best_h = h_candidate.copy()
        except Exception:
            pass
>>>>>>> REPLACE

