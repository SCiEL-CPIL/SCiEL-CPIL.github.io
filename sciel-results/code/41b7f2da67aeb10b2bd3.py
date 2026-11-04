import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _autoconvolution(h):
    """Compute the full autoconvolution of h (length 2*N-1)."""
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    H = np.fft.rfft(h, n=n_fft)
    conv = np.fft.irfft(H * H, n=n_fft)
    return conv[:2 * N - 1]

def _compute_ratio(h):
    """Compute ||f||_2^2 / (||f||_1 * ||f*f||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l2_sq = float(np.sum(h * h))
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    conv = _autoconvolution(h)
    conv_inf = float(np.max(np.abs(conv)))
    if conv_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * conv_inf)

def _shape_score(h):
    """Scale-invariant shape score: for unit-L1 shape, score = l2_sq / l_inf."""
    h = np.asarray(h, dtype=np.float64)
    l1 = float(np.sum(h))
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    l2_sq = float(np.sum(h_norm * h_norm))
    conv = _autoconvolution(h_norm)
    l_inf = float(np.max(np.abs(conv)))
    if l_inf < 1e-15:
        return 0.0
    return l2_sq / l_inf

def _neg_shape_score(params):
    """Negative shape score for minimization. params are log of positive values."""
    h = np.exp(params)
    return -_shape_score(h)

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_h = np.ones(N)
    best_score = _compute_ratio(best_h)

    # --- Phase 1: Parametric shape search (fast, no iterative optimization) ---
    def eval_shape(h):
        s = _shape_score(h)
        return s

    best_shape = None
    best_shape_score = 0.0

    x = np.linspace(-1.0, 1.0, N)

    # Family 1: Gaussian windows
    for sigma in np.linspace(0.05, 2.5, 50):
        for center in np.linspace(-0.6, 0.6, 20):
            h = np.exp(-0.5 * ((x - center) / sigma) ** 2)
            s = eval_shape(h)
            if s > best_shape_score:
                best_shape_score = s
                best_shape = h.copy()

    # Family 2: Lorentzian (Cauchy) windows
    for gamma in np.linspace(0.05, 2.5, 40):
        for center in np.linspace(-0.6, 0.6, 15):
            h = 1.0 / (1.0 + ((x - center) / gamma) ** 2)
            s = eval_shape(h)
            if s > best_shape_score:
                best_shape_score = s
                best_shape = h.copy()

    # Family 3: Raised-cosine (Hann) windows
    for support_frac in np.linspace(0.1, 1.0, 25):
        n_support = max(2, int(N * support_frac))
        for center_idx in range(0, max(1, N - n_support + 1)):
            h = np.zeros(N)
            idx = center_idx
            t = np.linspace(0, 1, n_support)
            h[idx:idx + n_support] = 0.5 * (1 + np.cos(2 * np.pi * (t - 0.5)))
            s = eval_shape(h)
            if s > best_shape_score:
                best_shape_score = s
                best_shape = h.copy()

    # Family 4: Super-Gaussian h = (1 - |x-c|^p)^q
    for p in [2.0, 3.0, 4.0, 6.0, 8.0, 12.0]:
        for q in [1.0, 1.5, 2.0, 3.0, 4.0]:
            for support_frac in np.linspace(0.2, 1.0, 12):
                n_support = max(2, int(N * support_frac))
                for center_idx in range(0, max(1, N - n_support + 1), max(1, (N - n_support) // 6)):
                    h = np.zeros(N)
                    idx = center_idx
                    t = np.linspace(-1, 1, n_support)
                    h[idx:idx + n_support] = np.maximum(0, 1 - np.abs(t) ** p) ** q
                    s = eval_shape(h)
                    if s > best_shape_score:
                        best_shape_score = s
                        best_shape = h.copy()

    # Family 5: Box with tapered edges
    for box_frac in np.linspace(0.05, 0.8, 15):
        for taper_frac in np.linspace(0.0, 0.4, 10):
            total_frac = box_frac + 2 * taper_frac
            if total_frac > 1.0:
                continue
            n_total = max(3, int(N * total_frac))
            n_box = max(1, int(N * box_frac))
            n_taper = (n_total - n_box) // 2
            for center_idx in range(0, max(1, N - n_total + 1), max(1, (N - n_total) // 8)):
                h = np.zeros(N)
                idx = center_idx
                if n_taper > 0:
                    h[idx:idx + n_taper] = np.linspace(0, 1, n_taper)
                    idx += n_taper
                h[idx:idx + n_box] = 1.0
                idx += n_box
                if n_taper > 0:
                    h[idx:idx + n_taper] = np.linspace(1, 0, n_taper)
                s = eval_shape(h)
                if s > best_shape_score:
                    best_shape_score = s
                    best_shape = h.copy()

    # Family 6: Two-bump (double Gaussian)
    for sigma in np.linspace(0.05, 0.8, 15):
        for separation in np.linspace(0.2, 1.5, 15):
            h = (np.exp(-0.5 * ((x - separation / 2) / sigma) ** 2) +
                 np.exp(-0.5 * ((x + separation / 2) / sigma) ** 2))
            s = eval_shape(h)
            if s > best_shape_score:
                best_shape_score = s
                best_shape = h.copy()

    # Family 7: Sparse spectral approach - h = |IDFT(sparse spectrum)|^2
    # Try K=2,3,4 active frequency bins
    for K in [2, 3, 4]:
        # Try all combinations of K bins from the available frequency bins
        n_freq = N // 2 + 1  # number of rfft bins
        # Focus on low-frequency bins for smooth shapes
        candidate_bins = list(range(min(n_freq, 10)))
        from itertools import combinations
        for bins in combinations(candidate_bins, K):
            # Try various magnitude distributions
            for mag_pattern in [np.ones(K), np.linspace(0.5, 1.0, K), np.exp(np.linspace(0, 1, K))]:
                spec = np.zeros(n_freq, dtype=np.complex128)
                for i, b in enumerate(bins):
                    spec[b] = mag_pattern[i]
                # h = |ifft(spec)|^2
                h_complex = np.fft.irfft(spec, n=N)
                h = np.real(h_complex) ** 2 + np.imag(h_complex) ** 2
                # Ensure nonnegative (should be by construction)
                h = np.maximum(h, 0)
                s = eval_shape(h)
                if s > best_shape_score:
                    best_shape_score = s
                    best_shape = h.copy()

    # Spike baselines
    for spike_pos in range(N):
        h = np.zeros(N)
        h[spike_pos] = 1.0
        s = eval_shape(h)
        if s > best_shape_score:
            best_shape_score = s
            best_shape = h.copy()

    # Uniform baseline
    h_uniform = np.ones(N)
    s = eval_shape(h_uniform)
    if s > best_shape_score:
        best_shape_score = s
        best_shape = h_uniform.copy()

    if best_shape is None:
        best_shape = np.zeros(N)
        best_shape[N // 2] = 1.0

    # Update best_h with the best shape found
    best_h = best_shape.copy()
    best_score = _compute_ratio(best_h)

    # --- Phase 2: Local refinement using complex parameterization h = |g|^2 ---
    # This removes non-negativity constraints and allows smoother optimization
    if time.time() < deadline - 20:
        # Initialize g from best_shape: g = sqrt(best_shape) * exp(i*phase)
        # Try several phase initializations
        n_refine = 5
        for r in range(n_refine):
            if time.time() > deadline - 10:
                break

            if r == 0:
                # Real positive initialization
                g0 = np.sqrt(np.maximum(best_h, 1e-10)).astype(np.complex128)
            elif r == 1:
                # Random phase
                g0 = np.sqrt(np.maximum(best_h, 1e-10)).astype(np.complex128)
                g0 *= np.exp(1j * np.random.randn(N))
            elif r == 2:
                # Perturbed
                g0 = np.sqrt(np.maximum(best_h, 1e-10)).astype(np.complex128)
                g0 *= (1 + 0.1 * np.random.randn(N) + 0.1j * np.random.randn(N))
            else:
                # Sparse spectral initialization
                spec = np.fft.rfft(g0, n=N)
                # Zero out most high-frequency components
                for i in range(len(spec)):
                    if i > 5:
                        spec[i] *= 0.1
                g0 = np.fft.irfft(spec, n=N)
                g0 = np.maximum(np.abs(g0), 1e-5) + 0j

            # Flatten complex params to real vector
            def flatten_g(g):
                return np.concatenate([np.real(g), np.imag(g)])

            def unflatten_g(v):
                n = len(v) // 2
                return v[:n].astype(np.complex128) + 1j * v[n:].astype(np.complex128)

            def neg_score_complex(params):
                g = unflatten_g(params)
                h = np.real(g) ** 2 + np.imag(g) ** 2
                h = np.maximum(h, 0)
                return -_shape_score(h)

            try:
                result = minimize(
                    neg_score_complex,
                    flatten_g(g0),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                g_opt = unflatten_g(result.x)
                h_opt = np.real(g_opt) ** 2 + np.imag(g_opt) ** 2
                h_opt = np.maximum(h_opt, 0)
                score = _shape_score(h_opt)
                if score > best_shape_score:
                    best_shape_score = score
                    best_shape = h_opt.copy()
            except Exception:
                pass

        # Update best_h
        best_h = best_shape.copy()
        best_score = _compute_ratio(best_h)

    # --- Phase 3: Scale to maximize raw score ---
    # score(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For h = c * shape:
    #   score = (c^2 * ||shape||_2^2) / (c * ||shape||_1 * c^2 * ||shape*shape||_inf)
    #         = ||shape||_2^2 / (c * ||shape||_1 * ||shape*shape||_inf)
    #         = shape_score / c
    # So to maximize raw score, make c as small as possible.
    # Use a very small scale to avoid floating point issues.
    scale = 1e-8
    result = best_h * scale
    result = np.maximum(result, 0.0)

    return [float(x) for x in result]
