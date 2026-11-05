"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Combines low-dimensional Fourier parameterization (inspiration 1) with
FFT-based gradient computation and multi-start L-BFGS-B (current program).
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
    DEADLINE = 143.0  # 7s safety margin before 150s kill

    best_value = np.inf
    best_h = np.ones(N).astype(float)

    def evaluate(h):
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf, False
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        val = abs(2.0 * N * max_conv / (s * s))
        valid = np.all(np.isfinite(h)) and abs(s) > 1e-12
        return val, valid

    val_seed, _ = evaluate(best_h)
    best_value = val_seed

    # Precompute FFT sizes
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    fft_size_cc = 1
    while fft_size_cc < 3 * N - 1:
        fft_size_cc *= 2

    def fast_convolve(h):
        H = np.fft.rfft(h, n=fft_size)
        conv = np.fft.irfft(H * H, n=fft_size)
        return conv[:2 * N - 1]

    def fast_cross_corr(weights, h):
        h_rev = h[::-1]
        H_rev = np.fft.rfft(h_rev, n=fft_size_cc)
        W = np.fft.rfft(weights, n=fft_size_cc)
        conv_result = np.fft.irfft(H_rev * W, n=fft_size_cc)
        grad_lse = 2.0 * conv_result[N-1:2*N-1]
        return grad_lse

    # Fourier basis parameterization
    # h[n] = a0 + sum_{k=1}^{K-1} (a_k * cos(2*pi*k*n/N) + b_k * sin(2*pi*k*n/N))
    K = 12  # Number of Fourier modes (DC + 11 harmonics, both cos and sin)
    dim = 2 * K  # Dimension of parameter space
    n_idx = np.arange(N)

    def reconstruct(x):
        """Reconstruct h from Fourier parameters x."""
        h = np.full(N, x[0])  # DC component
        for k in range(1, K):
            h += x[k] * np.cos(2 * np.pi * k * n_idx / N)
            h += x[K + k - 1] * np.sin(2 * np.pi * k * n_idx / N)
        return h

    def smooth_objective_and_grad_full(h, beta=120.0):
        """Smooth surrogate for |2N * max(conv) / sum(h)^2| with gradient."""
        s = np.sum(h)
        if abs(s) < 1e-10:
            return 1e10, np.zeros_like(h)
        conv = fast_convolve(h)
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
        f = 2.0 * N * lse / (s * s)
        if not np.isfinite(f):
            return 1e10, np.zeros_like(h)
        weights = exp_terms / sum_weights
        grad_lse = fast_cross_corr(weights, h)
        df_dh = 2.0 * N * grad_lse / (s * s) - 4.0 * N * lse / (s * s * s)
        sign_f = 1.0 if f >= 0 else -1.0
        grad = sign_f * df_dh
        return f, grad

    def fourier_objective(x, beta=120.0):
        """Objective in Fourier parameter space."""
        h = reconstruct(x)
        f, _ = smooth_objective_and_grad_full(h, beta)
        return f

    def fourier_gradient(x, beta=120.0):
        """Gradient in Fourier parameter space via chain rule."""
        h = reconstruct(x)
        f, grad_h = smooth_objective_and_grad_full(h, beta)
        if not np.isfinite(f):
            return np.zeros(dim)
        # grad_x[0] = sum(grad_h)  (DC term)
        grad_x = np.zeros(dim)
        grad_x[0] = np.sum(grad_h)
        for k in range(1, K):
            grad_x[k] = np.dot(grad_h, np.cos(2 * np.pi * k * n_idx / N))
            grad_x[K + k - 1] = np.dot(grad_h, np.sin(2 * np.pi * k * n_idx / N))
        return grad_x

    rng = np.random.RandomState(123)

    # Generate diverse initial points in Fourier parameter space
    initial_points = []

    # Seed: all ones -> a0 = 1, rest = 0
    x_seed = np.zeros(dim)
    x_seed[0] = 1.0
    initial_points.append(x_seed)

    # Random Fourier coefficients
    for _ in range(20):
        x_rand = np.zeros(dim)
        x_rand[0] = rng.uniform(0.5, 2.0)  # Bias toward positive DC
        x_rand[1:] = rng.normal(0, 0.5, dim - 1)
        initial_points.append(x_rand)

    # Low-frequency dominant patterns
    for freq in [1, 2, 3, 5, 7, 10]:
        x_low = np.zeros(dim)
        x_low[0] = 1.0
        # Add a strong component at the specified frequency
        if freq < K:
            x_low[freq] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
            x_low[K + freq - 1] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)
        initial_points.append(x_low)

    # Alternating-like pattern (high frequency)
    x_alt = np.zeros(dim)
    x_alt[0] = 0.5
    if K > N // 2:
        pass  # Can't represent perfectly
    else:
        x_alt[-1] = 0.5  # Highest frequency component
    initial_points.append(x_alt)

    # Bump-like patterns in Fourier space
    for center_freq in [1, 2, 3]:
        for width in [1, 2, 3]:
            x_bump = np.zeros(dim)
            x_bump[0] = 1.0
            for k in range(max(1, center_freq - width), min(K, center_freq + width + 1)):
                if k >= 1 and k < K:
                    x_bump[k] = 0.3 * np.exp(-((k - center_freq) ** 2) / (2 * width ** 2))
                    x_bump[K + k - 1] = 0.3 * np.exp(-((k - center_freq) ** 2) / (2 * width ** 2))
            initial_points.append(x_bump)

    # Sort by initial value
    def sort_key_fourier(x):
        h = reconstruct(x)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key_fourier)

    # Optimize from each initial point in Fourier space
    for x0 in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        try:
            result = minimize(
                fourier_objective,
                x0,
                jac=fourier_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8}
            )
            h_opt = reconstruct(result.x)
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        h_init = reconstruct(x0)
        val_init, valid = evaluate(h_init)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = h_init.copy()

    # Now refine in full h-space from the best Fourier solution
    if time.time() - start_time < DEADLINE - 10:
        try:
            result = minimize(
                lambda h: fourier_objective(_h_to_fourier(h)) if False else smooth_objective_and_grad_full(h)[0],
                best_h,
                jac=lambda h: smooth_objective_and_grad_full(h)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-9, 'ftol': 1e-12}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Additional multi-start in full h-space with diverse initializations
    if time.time() - start_time < DEADLINE - 10:
        h_inits = []
        # Random
        for _ in range(8):
            h_inits.append(rng.uniform(-1, 1, N))
        # Bimodal
        for _ in range(8):
            h = np.zeros(N)
            k = rng.randint(50, 300)
            idx = rng.choice(N, size=k, replace=False)
            h[idx] = rng.choice([-1, 1], size=k)
            h_inits.append(h)
        # Smooth random
        for _ in range(8):
            h = rng.uniform(-1, 1, N)
            h = np.convolve(h, np.ones(5)/5, mode='same')
            h_inits.append(h)
        # Cosine-based
        t = np.linspace(0, 1, N, endpoint=False)
        for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
            h_inits.append(np.cos(2 * np.pi * freq * t))

        for h0 in h_inits:
            if time.time() - start_time > DEADLINE - 10:
                break
            s = np.sum(h0)
            if abs(s) < 1e-10:
                h0 = h0 + 0.1
            try:
                result = minimize(
                    lambda h: smooth_objective_and_grad_full(h)[0],
                    h0,
                    jac=lambda h: smooth_objective_and_grad_full(h)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'gtol': 1e-7}
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
            val_init, valid = evaluate(h0)
            if valid and val_init < best_value:
                best_value = val_init
                best_h = h0.copy()

    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        try:
            result = minimize(
                lambda h: smooth_objective_and_grad_full(h)[0],
                best_h,
                jac=lambda h: smooth_objective_and_grad_full(h)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-9, 'ftol': 1e-12}
            )
            h_opt = result.x
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
        except Exception:
            pass

    # Perturbation phase
    if time.time() - start_time < DEADLINE - 3:
        scales = [0.1, 0.05, 0.02, 0.01, 0.005]
        for scale in scales:
            if time.time() - start_time > DEADLINE - 3:
                break
            for _ in range(3):
                if time.time() - start_time > DEADLINE - 3:
                    break
                h_pert = best_h + scale * rng.normal(0, 1, N)
                val_pert, valid = evaluate(h_pert)
                if valid and val_pert < best_value:
                    best_value = val_pert
                    best_h = h_pert.copy()
                else:
                    try:
                        result = minimize(
                            lambda h: smooth_objective_and_grad_full(h)[0],
                            h_pert,
                            jac=lambda h: smooth_objective_and_grad_full(h)[1],
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'maxfun': 300, 'gtol': 1e-8}
                        )
                        h_opt = result.x
                        val_opt, valid = evaluate(h_opt)
                        if valid and val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                    except Exception:
                        pass

    return best_h
