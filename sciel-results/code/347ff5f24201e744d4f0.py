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
7. Differential Evolution for global exploration of sparse Sidon-like structures.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 50
_DEADLINE = 145.0  # seconds, safety margin before 150s kill


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


def _polish_exact(g0, time_limit=5.0):
    """
    Final high-precision polish using exact peak computation (no smoothing).
    This removes the bias from the smooth-max approximation.
    """
    start_time = time.time()
    
    def _obj_exact(g_flat):
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

    try:
        result = minimize(
            _obj_exact,
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


def _sparse_sidon_objective(params):
    """
    Objective for Differential Evolution on sparse Sidon-like structures.
    params: [amplitude_1, ..., amplitude_K, phase_1, ..., phase_K, index_1, ..., index_K]
    K is fixed (e.g., 5).
    """
    K = 5
    amps = np.abs(params[:K])
    phases = params[K:2*K]
    indices = np.round(params[2*K:]).astype(int)
    
    # Ensure indices are within [0, N-1] and unique (Sidon-like)
    indices = np.clip(indices, 0, N-1)
    
    # Check for duplicates and penalize heavily
    if len(set(indices)) < K:
        return 1e10
    
    # Check Sidon property: pairwise sums must be distinct
    used_sums = set()
    valid = True
    for i in range(K):
        for j in range(i+1, K):
            s = indices[i] + indices[j]
            if s in used_sums:
                valid = False
                break
            used_sums.add(s)
    if not valid:
        return 1e10
    
    h = np.zeros(N)
    for k in range(K):
        h[indices[k]] = amps[k] * np.cos(phases[k])
        # Since we want non-negative h, we take absolute value or ensure positive amplitude
        # Actually, h must be non-negative. Let's just use amplitude directly.
        # The phase doesn't make sense for real non-negative h.
        # Let's re-parametrize: just amplitudes and indices.
    
    # Re-do with simpler param: [amp_1, ..., amp_K, index_1, ..., index_K]
    # This function is a placeholder for the DE call which will use a different signature
    
    return 0.0


def _de_sparse_search(time_limit=20.0):
    """
    Use Differential Evolution to find sparse Sidon-like sets of peaks.
    Parameters: K amplitudes and K indices.
    """
    K = 5
    # Bounds: amplitudes [0, 1], indices [0, N-1]
    bounds = [(0, 1)] * K + [(0, N-1)] * K
    
    def _obj(params):
        amps = params[:K]
        indices = np.round(params[K:]).astype(int)
        indices = np.clip(indices, 0, N-1)
        
        # Check for unique indices
        if len(set(indices)) < K:
            return 1e10
            
        # Check Sidon property
        used_sums = set()
        for i in range(K):
            for j in range(i+1, K):
                s = indices[i] + indices[j]
                if s in used_sums:
                    return 1e10
                used_sums.add(s)
        
        h = np.zeros(N)
        for k in range(K):
            h[indices[k]] = amps[k]
            
        if np.sum(h) < 1e-15:
            return 1e10
            
        return -_ratio(h)
        
    try:
        result = differential_evolution(
            _obj,
            bounds,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            seed=42,
            polish=False  # We will polish separately
        )
        
        if result.success:
            amps = result.x[:K]
            indices = np.round(result.x[K:]).astype(int)
            indices = np.clip(indices, 0, N-1)
            
            h = np.zeros(N)
            for k in range(K):
                h[indices[k]] = amps[k]
                
            # Smooth it slightly to make it a better starting point for L-BFGS-B
            kernel = np.ones(3) / 3
            h_smooth = np.convolve(h, kernel, mode='same')
            return h_smooth
    except Exception:
        pass
    return None


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
    for width in [0.05, 0.1, 0.15, 0.2, 0.3]:
        initializations_h.append(_gaussian_profile(0.0, width, N, grid_start, grid_end))

    # 3. Lorentzian-like
    for width in [0.05, 0.1, 0.15]:
        initializations_h.append(_lorentzian_profile(0.0, width, N, grid_start, grid_end))

    # 4. Bimodal
    for sep in [0.2, 0.3, 0.4]:
        h_bi = _gaussian_profile(-sep, 0.08, N, grid_start, grid_end) + \
               _gaussian_profile(sep, 0.08, N, grid_start, grid_end)
        initializations_h.append(h_bi)

    # 5. Raised cosine
    t = (x - grid_start) / (grid_end - grid_start)
    initializations_h.append(_raised_cosine(t))

    # 6. Cosine window
    initializations_h.append(0.5 + 0.5 * np.cos(2 * np.pi * t))

    # 7. Triangle / tent
    initializations_h.append(1 - np.abs(x) / 0.25)

    # 8. Peak in center
    h_center = np.zeros(N)
    h_center[N // 2 - 3:N // 2 + 4] = 1.0
    initializations_h.append(h_center)

    # 9. Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:4] = 1.0
    h_edge[-4:] = 1.0
    initializations_h.append(h_edge)

    # 10. Random smooth
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        initializations_h.append(h_smooth)

    # 11. DFT-based random
    for seed in [10, 20, 30, 40]:
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

    # 14. Narrow Gaussian
    initializations_h.append(_gaussian_profile(0.0, 0.03, N, grid_start, grid_end))

    # 15. Sparse Sidon-like from DE
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 30:
        h_de = _de_sparse_search(20.0)
        if h_de is not None:
            initializations_h.append(h_de)

    time_per_opt = 8.0  # seconds per optimization attempt

    for i, h0 in enumerate(initializations_h):
        elapsed = time.time() - start_time
        if elapsed + time_per_opt > _DEADLINE - 15:
            break

        # Convert h0 to g0
        g0_flat = _h_to_g_flat(h0)

        # Two-stage smoothing: start with larger sigma, refine with smaller
        h_opt = _optimize_complex(g0_flat, time_per_opt / 2, sigma=0.1)
        g_mid = _h_to_g_flat(h_opt)
        h_opt = _optimize_complex(g_mid, time_per_opt / 2, sigma=0.01)

        score = _ratio(h_opt)
        if score > best_score:
            best_score = score
            best_h = h_opt

    # Final polish: run a few more iterations from the best solution
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 10:
        # Try small perturbations around the best
        for pert_seed in [100, 200, 300, 400, 500]:
            elapsed = time.time() - start_time
            if elapsed + 5 > _DEADLINE - 10:
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

    # High-precision polish with exact peak
    elapsed = time.time() - start_time
    if elapsed < _DEADLINE - 5:
        g_best = _h_to_g_flat(best_h)
        h_polished = _polish_exact(g_best, time_limit=5.0)
        score = _ratio(h_polished)
        if score > best_score:
            best_score = score
            best_h = h_polished

    # Ensure final result is valid
    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return list(best_h)
