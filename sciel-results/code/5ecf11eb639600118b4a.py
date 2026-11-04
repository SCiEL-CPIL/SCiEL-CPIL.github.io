"""Second autocorrelation inequality: maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf).

Strategy:
1. Use the complex parameterization h = |g|^2 to naturally enforce non-negativity.
2. Optimize the complex vector g using L-BFGS-B on the flattened real/imag parts.
3. Use a smooth-max approximation for the supremum norm in the objective to allow
   gradient-based optimization.
4. Combine diverse initializations (Gaussian, Lorentzian, random, structured) with
   local refinement.
5. Exploit scale invariance: the ratio is invariant under scaling h -> c*h, so we
   can normalize during optimization and return the shape.
6. Final high-precision polish pass with exact peak computation.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
_DEADLINE = 115.0  # seconds, safety margin before 120s kill


def _autoconv_peak(h):
    """Compute the supremum norm of the autoconvolution of h."""
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)


def _ratio(h):
    """Compute the ratio: squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
    h = np.asarray(h, dtype=float)
    if np.any(h < 0) or np.sum(h) < 1e-15:
        return 0.0
    l2sq = np.sum(h ** 2)
    l1 = np.sum(h)
    peak = _autoconv_peak(h)
    if peak < 1e-30:
        return 0.0
    return l2sq / (l1 * peak)


def _smooth_max(x, sigma=1.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    if x_max == 0:
        return 0.0
    x_shifted = x - x_max
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_shifted / sigma))) * sigma
    return log_sum_exp


def _objective_complex(g_flat, sigma=0.01):
    """
    Objective to minimize: negative of the ratio.
    g_flat is a real vector of length 2*N representing complex g.
    h = |g|^2.
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0] ** 2 + g[:, 1] ** 2

    # Normalize h to have L1 norm 1 to ensure scale invariance and stability
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1

    l2sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    peak = _smooth_max(conv, sigma=sigma)

    if peak < 1e-30:
        return 0.0

    return -l2sq / peak  # Since l1 is 1, ratio is l2sq / peak


def _objective_exact(g_flat):
    """
    Objective to minimize using exact peak (for polish).
    """
    g = g_flat.reshape(N, 2)
    h = g[:, 0] ** 2 + g[:, 1] ** 2

    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h = h / l1

    l2sq = np.sum(h ** 2)
    conv = np.convolve(h, h, mode='full')
    peak = np.max(conv)

    if peak < 1e-30:
        return 0.0

    return -l2sq / peak


def _optimize_complex(g0, time_limit=10.0, sigma=0.01):
    """Run L-BFGS-B optimization on complex parameters."""
    start_time = time.time()

    try:
        result = minimize(
            lambda x: _objective_complex(x, sigma=sigma),
            g0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 5000}
        )

        elapsed = time.time() - start_time
        if elapsed > time_limit:
            g = g0.reshape(N, 2)
            h = g[:, 0] ** 2 + g[:, 1] ** 2
            return h

        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0] ** 2 + g_opt[:, 1] ** 2
        return h_opt

    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0] ** 2 + g[:, 1] ** 2
        return h


def _polish_exact(g0, time_limit=5.0):
    """High-precision polish using exact peak computation."""
    start_time = time.time()
    try:
        result = minimize(
            _objective_exact,
            g0,
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 10000}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            g = g0.reshape(N, 2)
            h = g[:, 0] ** 2 + g[:, 1] ** 2
            return h

        g_opt = result.x.reshape(N, 2)
        h_opt = g_opt[:, 0] ** 2 + g_opt[:, 1] ** 2
        return h_opt
    except Exception:
        g = g0.reshape(N, 2)
        h = g[:, 0] ** 2 + g[:, 1] ** 2
        return h


def _h_to_g_flat(h):
    """Convert h (nonnegative) to g_flat (real vector of length 2N)."""
    g_real = np.sqrt(np.maximum(h, 0))
    g_imag = np.zeros(N)
    return np.concatenate([g_real, g_imag])


def _gaussian_profile(center, width, N, grid_start, grid_end):
    """Generate a Gaussian profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return np.exp(-0.5 * ((x - center) / width) ** 2)


def _lorentzian_profile(center, width, N, grid_start, grid_end):
    """Generate a Lorentzian (Cauchy) profile on the grid."""
    x = np.linspace(grid_start, grid_end, N)
    return 1.0 / (1.0 + ((x - center) / width) ** 2)


def _raised_cosine(t):
    """Raised cosine window on t in [0, 1]."""
    return 0.5 * (1 + np.cos(np.pi * t))


def construct():
    start_time = time.time()

    grid_start = -0.25
    grid_end = 0.25
    x = np.linspace(grid_start, grid_end, N)

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # Generate diverse initializations
    initializations_h = []

    # 1. Uniform
    initializations_h.append(np.ones(N))

    # 2. Gaussian-like with various widths
    for width in [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]:
        initializations_h.append(_gaussian_profile(0.0, width, N, grid_start, grid_end))

    # 3. Lorentzian-like
    for width in [0.05, 0.1, 0.15]:
        initializations_h.append(_lorentzian_profile(0.0, width, N, grid_start, grid_end))

    # 4. Bimodal
    for sep in [0.1, 0.2, 0.3, 0.4]:
        for w in [0.05, 0.08, 0.12]:
            h_bi = _gaussian_profile(-sep, w, N, grid_start, grid_end) + \
                   _gaussian_profile(sep, w, N, grid_start, grid_end)
            initializations_h.append(h_bi)

    # 5. Raised cosine
    t = (x - grid_start) / (grid_end - grid_start)
    initializations_h.append(_raised_cosine(t))

    # 6. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(2 * np.pi * t))

    # 7. Triangle / tent
    initializations_h.append(1 - np.abs(x) / 0.25)

    # 8. Peak in center
    for w in [3, 5, 8]:
        h_center = np.zeros(N)
        h_center[N // 2 - w:N // 2 + w + 1] = 1.0
        initializations_h.append(h_center)

    # 9. Peak at edges
    for w in [3, 5]:
        h_edge = np.zeros(N)
        h_edge[0:w] = 1.0
        h_edge[-w:] = 1.0
        initializations_h.append(h_edge)

    # 10. Random smooth
    for seed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)

    # 11. DFT-based random
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H) ** 2
        h_dft = h_dft / np.max(h_dft)
        initializations_h.append(h_dft)

    # 12. Linear ramps
    initializations_h.append(np.linspace(0.1, 1.0, N))
    initializations_h.append(np.linspace(1.0, 0.1, N))

    # 13. Sine-based
    initializations_h.append(np.sin(np.pi * (x - grid_start) / (grid_end - grid_start)) ** 2)
    initializations_h.append(np.cos(np.pi * x / 0.5) ** 2)

    # 14. Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    initializations_h.append(h_alt)

    # 15. Sparse peaks
    for k in range(2, 8):
        h_sparse = np.zeros(N)
        indices = np.random.RandomState(k).choice(N, k, replace=False)
        h_sparse[indices] = 1.0
        initializations_h.append(h_sparse)

    # Optimization loop
    time_per_opt = 6.0  # seconds per optimization attempt, increased slightly to allow more iterations

    # Sort initializations by initial score to prioritize promising starts
    init_scores = []
    for h0 in initializations_h:
        s = _ratio(h0)
        init_scores.append(s)
    
    # Sort indices by score descending
    sorted_indices = np.argsort(init_scores)[::-1]
    
    for idx in sorted_indices:
        h0 = initializations_h[idx]
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE:
            break

        # Convert h0 to g0
        g0_flat = _h_to_g_flat(h0)

        # Try with different smoothness parameters
        for sigma in [0.01, 0.05, 0.1]:
            elapsed = time.time() - start_time
            if elapsed + time_per_opt > _DEADLINE:
                break
            
            h_opt = _optimize_complex(g0_flat, time_per_opt, sigma=sigma)

            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt

    # Final polish: run a few more iterations from the best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 15:
        # Try small perturbations around the best
        for pert_seed in [100, 200, 300, 400, 500, 600, 700, 800]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE:
                break
            rng_local = np.random.RandomState(pert_seed)
            # Perturb in g-space
            g_best_real = np.sqrt(np.maximum(best_h, 0))
            g_best_imag = np.zeros(N)

            pert_real = g_best_real + rng_local.randn(N) * 0.05
            pert_imag = rng_local.randn(N) * 0.05

            g_pert_flat = np.concatenate([pert_real, pert_imag])

            h_opt = _optimize_complex(g_pert_flat, 5.0, sigma=0.01)
            score = _ratio(h_opt)
            if score > best_score:
                best_score = score
                best_h = h_opt

    # High-precision polish pass
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        g_best_flat = _h_to_g_flat(best_h)
        h_polished = _polish_exact(g_best_flat, time_limit=min(10.0, _DEADLINE - elapsed - 2.0))
        score = _ratio(h_polished)
        if score > best_score:
            best_score = score
            best_h = h_polished

    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return list(best_h)
