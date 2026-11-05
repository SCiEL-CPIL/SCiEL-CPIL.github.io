"""
Combined approach: Spectral null design + smooth gradient optimization.

Strategy:
1. Generate diverse initial points, including spectral-null designs (DC suppression)
2. Use smooth log-sum-exp surrogate with analytic gradient for BFGS optimization
3. Refine with Nelder-Mead on the exact objective
4. Keep Chebyshev basis as a secondary structural family
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d

N = 400
_DEADLINE = 143.0  # safety margin under 150s limit

# Precompute FFT sizes
_FFT_SIZE = 1 << (2 * N - 1).bit_length()
_FFT_SIZE = max(_FFT_SIZE, 1024)

# Cross-correlation FFT size for gradient computation
_FFT_CC_SIZE = 1 << (N + 2 * N - 1).bit_length()


def _fast_autoconv(h):
    """Fast autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    C = np.fft.irfft(H * H, _FFT_SIZE)
    return C[:2 * N - 1]


def _fast_value(h):
    """Compute objective using FFT-based autoconvolution."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.all(np.isfinite(h)):
        return np.inf
    c = _fast_autoconv(h)
    return abs(2.0 * N * np.max(c) / (s * s))


def _fast_cross_corr(weights, h):
    """
    Compute grad_lse[i] = 2 * sum_{m=0}^{N-1} weights[i+m] * h[m] for i=0..N-1
    using FFT-based cross-correlation.
    """
    h_rev = h[::-1]
    H_rev = np.fft.rfft(h_rev, n=_FFT_CC_SIZE)
    W = np.fft.rfft(weights, n=_FFT_CC_SIZE)
    conv_result = np.fft.irfft(H_rev * W, n=_FFT_CC_SIZE)
    grad_lse = 2.0 * conv_result[N - 1:2 * N - 1]
    return grad_lse


def _smooth_obj_and_grad(x, beta=150.0):
    """Compute both smooth objective and gradient in one pass."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)

    conv = _fast_autoconv(h)
    max_c = np.max(conv)
    shifted = conv - max_c
    max_shifted = np.max(shifted)
    if max_shifted > 500:
        return 1e10, np.zeros_like(h)

    exp_terms = np.exp(beta * (shifted - max_shifted))
    sum_weights = np.sum(exp_terms)
    if sum_weights < 1e-300:
        return 1e10, np.zeros_like(h)

    lse = max_c + max_shifted + np.log(sum_weights) / beta
    f_val = 2.0 * N * lse / (s * s)

    if not np.isfinite(f_val):
        return 1e10, np.zeros_like(h)

    weights = exp_terms / sum_weights
    grad_lse = _fast_cross_corr(weights, h)

    df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)

    if f_val >= 0:
        grad = df_dh
    else:
        grad = -df_dh

    return f_val, grad


def _smooth_objective(x, beta=150.0):
    """Smooth surrogate using log-sum-exp of conv entries."""
    f, _ = _smooth_obj_and_grad(x, beta)
    return f


def _smooth_grad(x, beta=150.0):
    """Analytic gradient of the smooth surrogate."""
    _, grad = _smooth_obj_and_grad(x, beta)
    return grad


def _cheb_basis(k, n_samples=N):
    """Generate Chebyshev polynomial basis vectors on [-1, 1]."""
    x = np.linspace(-1, 1, n_samples)
    basis = np.zeros((k, n_samples))
    basis[0] = 1.0
    if k > 1:
        basis[1] = x
    for j in range(2, k):
        basis[j] = 2.0 * x * basis[j - 1] - basis[j - 2]
    return basis


_CHEB_BASIS_CACHE = {}


def _get_cheb_basis(K):
    if K not in _CHEB_BASIS_CACHE:
        _CHEB_BASIS_CACHE[K] = _cheb_basis(K, N)
    return _CHEB_BASIS_CACHE[K]


def _make_h_from_cheb(coeffs, K):
    B = _get_cheb_basis(K)
    return B.T @ coeffs


def _obj_cheb(coeffs, K):
    h = _make_h_from_cheb(coeffs, K)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e12
    c = _fast_autoconv(h)
    max_c = np.max(c)
    val = 2.0 * N * max_c / (s * s)
    return abs(val)


def _generate_initial_points(rng):
    """Generate diverse initial points for optimization."""
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # Random uniform
    for _ in range(8):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal: mostly zero with random spikes
    for _ in range(8):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines
    t = np.linspace(0, 1, N, endpoint=False)
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))

    # Parameterized bump profiles
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [25, 60, 120, 200]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            h_double = h_bump - np.exp(-((i_arr - (N - 1 - center)) ** 2) / (2 * width ** 2))
            initial_points.append(h_double)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7]:
        for _ in range(2):
            freqs = rng.randint(1, 40, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.5, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.cos(2 * np.pi * f * t + p)
            initial_points.append(h_fourier)

    # Structured patterns: blocks of +1 and -1
    for block_size in [20, 50, 100]:
        n_blocks = N // block_size
        for _ in range(2):
            h_blocks = np.zeros(N)
            for b in range(n_blocks):
                sign = rng.choice([-1, 1])
                h_blocks[b * block_size:(b + 1) * block_size] = sign
            initial_points.append(h_blocks)

    # Smooth random via Gaussian filter
    for _ in range(8):
        h_smooth = rng.uniform(-1, 1, N)
        h_smooth = gaussian_filter1d(h_smooth, sigma=rng.uniform(2, 20))
        initial_points.append(h_smooth)

    # Spectral null initial points: design h such that DC component is small
    ones = np.ones(N)
    ones_norm_sq = np.dot(ones, ones)

    for _ in range(12):
        v = rng.normal(0, 1, N)
        dc_component = np.dot(v, ones) / ones_norm_sq * ones
        v_zero_dc = v - dc_component
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(v_zero_dc) / N
        h_spectral = v_zero_dc + dc_offset * ones
        initial_points.append(h_spectral)

    # More structured spectral nulls: use random phases in Fourier domain
    for _ in range(8):
        spectrum = rng.normal(0, 1, N // 2 + 1) + 1j * rng.normal(0, 1, N // 2 + 1)
        spectrum[0] = 0.0  # Null at DC
        spectrum_full = np.zeros(N, dtype=complex)
        spectrum_full[:N // 2 + 1] = spectrum
        if N % 2 == 0:
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2][::-1])
        else:
            spectrum_full[N // 2 + 1:] = np.conj(spectrum[1:N // 2 + 1][::-1])
        h_fourier_null = np.real(np.fft.ifft(spectrum_full))
        dc_offset = rng.uniform(0.1, 0.5) * np.linalg.norm(h_fourier_null) / N
        h_fourier_null += dc_offset
        initial_points.append(h_fourier_null)

    # Band-limited signal: zero in central frequency band, unity elsewhere
    # This directly implements the "spectral null" strategy
    for center_freq in [5, 10, 15, 20, 30]:
        for bandwidth in [3, 5, 10]:
            freqs = np.fft.rfftfreq(N) * N
            mask = np.ones(len(freqs))
            mask[np.abs(freqs - center_freq) < bandwidth] = 0.0
            # Random phases
            phases = rng.uniform(0, 2 * np.pi, len(freqs))
            spectrum = mask * np.exp(1j * phases)
            spectrum[0] = 0.0  # Ensure DC null
            spectrum_full = np.zeros(N, dtype=complex)
            spectrum_full[:len(spectrum)] = spectrum
            spectrum_full[len(spectrum):] = np.conj(spectrum[1:len(spectrum)][::-1])
            h_band = np.real(np.fft.ifft(spectrum_full))
            # Add small DC offset
            dc_offset = 0.1 * np.linalg.norm(h_band) / N
            h_band += dc_offset
            initial_points.append(h_band)

    return initial_points


def construct():
    """Construct step heights minimizing the objective value."""
    start = time.time()

    best_h = np.ones(N)
    best_val = 2.0

    def _update(h):
        nonlocal best_h, best_val
        if not np.all(np.isfinite(h)):
            return
        v = _fast_value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    rng = np.random.RandomState(42)

    # Generate and sort initial points
    initial_points = _generate_initial_points(rng)

    # Sort by initial value (best first)
    def sort_key(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = _fast_autoconv(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key)

    # Phase 1: Smooth gradient-based optimization from best initial points
    max_phase1 = min(30, len(initial_points))
    for i in range(max_phase1):
        if time.time() - start > _DEADLINE - 20:
            break

        h0 = initial_points[i]
        if not np.all(np.isfinite(h0)):
            continue

        # BFGS with smooth surrogate
        try:
            res = minimize(
                _smooth_objective,
                h0,
                jac=_smooth_grad,
                args=(150.0,),
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                _update(res.x)

                # Refine with Nelder-Mead on exact objective
                if time.time() - start < _DEADLINE - 10:
                    try:
                        r2 = minimize(
                            _fast_value, res.x,
                            method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                        )
                        if r2.x is not None and np.all(np.isfinite(r2.x)):
                            _update(r2.x)
                    except Exception:
                        pass
        except Exception:
            pass

    # Phase 2: Chebyshev basis optimization for structural diversity
    if time.time() - start < _DEADLINE - 15:
        for K in [5, 8, 12]:
            if time.time() - start > _DEADLINE - 10:
                break

            inits = [
                np.zeros(K),
                rng.uniform(-0.5, 0.5, K),
                rng.normal(0, 0.3, K),
            ]
            for init in inits:
                init = init.copy()
                init[0] = 1.0

                bounds = [(0.1, 10.0)] + [(-5.0, 5.0)] * (K - 1)
                remaining = _DEADLINE - (time.time() - start)
                if remaining < 5:
                    break

                try:
                    # Use scipy's differential evolution for Chebyshev coefficients
                    from scipy.optimize import differential_evolution
                    res = differential_evolution(
                        _obj_cheb, bounds, args=(K,),
                        x0=init,
                        maxiter=max(30, int(remaining * 1.5)),
                        popsize=max(10, min(30, 30 - K)),
                        tol=0.005, mutation=(0.5, 1.0), recombination=0.7,
                        seed=42 + K, polish=False,
                        init='sobol' if K <= 20 else 'latinhypercube'
                    )
                    if res.x is not None:
                        h = _make_h_from_cheb(res.x, K)
                        _update(h)

                        # Local polish
                        if time.time() - start < _DEADLINE - 5:
                            try:
                                r2 = minimize(
                                    _obj_cheb, res.x, args=(K,),
                                    method='Nelder-Mead',
                                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                                )
                                if r2.x is not None:
                                    h2 = _make_h_from_cheb(r2.x, K)
                                    _update(h2)
                            except Exception:
                                pass
                except Exception:
                    pass

    return best_h
