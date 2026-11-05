"""
Autocorrelation inequality: minimize |2*N*max(h*h)/sum(h)^2|

Strategy:
  - Direct time-domain optimization with FFT-accelerated autoconvolution.
  - Smooth log-sum-exp surrogate for the signed max.
  - Analytic gradient via FFT-based correlation (O(N log N) per step).
  - Diverse spectral initial points (DC-biased cosines, spectral-null modes).
  - Multi-beta L-BFGS-B refinement, then a short coordinate-descent polish.
  - Explicit deadline guard to stay well under the 150 s kill.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400


def construct():
    """
    Construct a sequence of N step heights minimizing:
        value = abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h)**2)
    """
    start_time = time.time()
    DEADLINE = 138.0  # safety margin before 150 s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    # Pre-compute FFT size for full autoconvolution (length 2N-1)
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2
    fft_size = max(fft_size, 1024)

    # Pre-compute FFT size for the gradient correlation (length up to 3N-2)
    fft_size_corr = 1
    while fft_size_corr < (2 * N - 1) + N - 1:
        fft_size_corr *= 2

    def fast_convolve(h):
        """Full autoconvolution via FFT."""
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = fast_convolve(h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    # Evaluate seed
    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # --- Smooth log-sum-exp surrogate ---
    def smooth_objective(h, beta=80.0):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10
        conv = fast_convolve(h)
        m = np.max(conv)
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -700, 700)
        lse = m + np.log(np.sum(np.exp(shifted))) / beta
        val = abs(2.0 * N * lse / (s * s))
        return val if np.isfinite(val) else 1e10

    def smooth_grad(h, beta=80.0):
        s = np.sum(h)
        if abs(s) < 1e-10:
            return np.zeros_like(h)

        conv = fast_convolve(h)
        m = np.max(conv)
        shifted = beta * (conv - m)
        shifted = np.clip(shifted, -700, 700)
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals)
        if sum_exp < 1e-300:
            weights = np.ones_like(conv) / len(conv)
        else:
            weights = exp_vals / sum_exp

        # d(soft_max)/dh_j = 2 * sum_k weights[j+k] * h[k]  for j=0..N-1
        # This is 2 * np.correlate(weights, h, mode='valid')
        d_soft = 2.0 * np.correlate(weights, h, mode='valid')

        lse = m + np.log(sum_exp) / beta
        inner = 2.0 * N * lse / (s * s)
        sign = 1.0 if inner >= 0 else -1.0

        grad = sign * 2.0 * N * (d_soft / (s * s) - 2.0 * lse / (s ** 3))
        return grad

    # --- Generate diverse initial points ---
    rng = np.random.RandomState(42)
    t = np.arange(N) / N
    initial_points = []

    # Seed
    initial_points.append(np.ones(N).astype(float))

    # DC-biased cosine modes (strong structure for autoconvolution)
    for k in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100]:
        for amp in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * t)
            s = np.sum(h)
            if abs(s) > 1e-12:
                initial_points.append(h)

    # Spectral-null: DC-removed random
    for _ in range(25):
        h = rng.normal(0, 1, N)
        h -= np.mean(h)
        if abs(np.sum(h)) < 0.1:
            h += 0.05
        initial_points.append(h)

    # Fourier-domain nulls: zero out low frequencies
    for _ in range(15):
        H = np.fft.rfft(rng.normal(0, 1, N))
        H[0:5] = 0
        h = np.fft.irfft(H, n=N)
        if abs(np.sum(h)) < 0.1:
            h += 0.05
        initial_points.append(h)

    # Random uniform
    for _ in range(15):
        initial_points.append(rng.uniform(-1, 1, N))

    # Bimodal spikes
    for _ in range(15):
        h = np.zeros(N)
        k = rng.randint(50, 300)
        idx = rng.choice(N, size=k, replace=False)
        h[idx] = rng.choice([-1, 1], size=k)
        initial_points.append(h)

    # Alternating signs
    h_alt = np.ones(N)
    h_alt[1::2] = -1
    initial_points.append(h_alt)

    # Low-frequency cosines and sines
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        initial_points.append(np.cos(2 * np.pi * freq * t))
        initial_points.append(np.sin(2 * np.pi * freq * t))

    # Gaussian bumps
    i_arr = np.arange(N)
    for center in [N // 4, N // 2, 3 * N // 4]:
        for width in [20, 40, 60, 100, 160]:
            h_bump = np.exp(-((i_arr - center) ** 2) / (2 * width ** 2))
            initial_points.append(h_bump)
            initial_points.append(-h_bump)

    # Random Fourier synthesis
    for n_comp in [3, 5, 7, 10]:
        for _ in range(8):
            freqs = rng.randint(1, 60, n_comp)
            phases = rng.uniform(0, 2 * np.pi, n_comp)
            amps = rng.uniform(0.3, 2.0, n_comp)
            h_fourier = np.zeros(N)
            for f, p, a in zip(freqs, phases, amps):
                h_fourier += a * np.sin(2 * np.pi * f * t + p)
            if abs(np.sum(h_fourier)) < 0.1:
                h_fourier += 0.05
            initial_points.append(h_fourier)

    # Anti-symmetric smoothed step
    for width in [30, 60, 100]:
        h_anti = np.zeros(N)
        h_anti[:N // 2] = 1.0
        h_anti[N // 2:] = -1.0
        # Simple Gaussian smoothing
        kernel = np.exp(-((np.arange(-width, width + 1)) ** 2) / (2 * (width / 3) ** 2))
        kernel /= np.sum(kernel)
        h_smooth = np.convolve(h_anti, kernel, mode='same')
        initial_points.append(h_smooth)
        initial_points.append(-h_smooth)

    # --- L-BFGS-B optimization from each initial point ---
    for init_h in initial_points:
        if time.time() - start_time > DEADLINE:
            break

        s = np.sum(init_h)
        if abs(s) < 1e-10:
            init_h = init_h + 0.1
            s = np.sum(init_h)

        for beta in [50.0, 100.0, 200.0]:
            if time.time() - start_time > DEADLINE:
                break
            try:
                result = minimize(
                    lambda x, b=beta: smooth_objective(x, b),
                    init_h,
                    jac=lambda x, b=beta: smooth_grad(x, b),
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-6}
                )
                h_opt = result.x
                val_opt, valid = evaluate(h_opt)
                if valid and val_opt < best_value:
                    best_value = val_opt
                    best_h = h_opt.copy()
                    if best_value < 1.0:
                        break
            except Exception:
                pass

        # Exact evaluation of initial point
        val_init, valid = evaluate(init_h)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = init_h.copy()

    # --- Coordinate-descent polish ---
    if time.time() - start_time < DEADLINE - 5:
        h_curr = best_h.copy()
        lr = 0.1
        deltas = [lr, -lr, lr * 0.5, -lr * 0.5, lr * 0.25, -lr * 0.25, lr * 0.125, -lr * 0.125]
        improved = True
        polish_iter = 0
        while improved and time.time() - start_time < DEADLINE - 2 and polish_iter < 12:
            improved = False
            polish_iter += 1
            for coord in rng.permutation(N):
                if time.time() - start_time > DEADLINE - 2:
                    break
                for delta in deltas:
                    h_try = h_curr.copy()
                    h_try[coord] += delta
                    val_try, valid = evaluate(h_try)
                    if valid and val_try < best_value:
                        best_value = val_try
                        best_h = h_try
                        h_curr = h_try
                        improved = True
                        break
            if not improved:
                lr *= 0.6
                deltas = [lr, -lr, lr * 0.5, -lr * 0.5, lr * 0.25, -lr * 0.25, lr * 0.125, -lr * 0.125]
                if lr < 1e-5:
                    break

    return best_h
