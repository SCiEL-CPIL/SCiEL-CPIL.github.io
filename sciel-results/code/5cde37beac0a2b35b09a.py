"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Structural approach: parameterize h as a smooth low-frequency function
with a small number of free parameters (Gaussian bumps + polynomial), then
optimize in that low-dimensional space.
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
    DEADLINE = 145.0  # 5s safety margin before 150s kill

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

    n_idx = np.arange(N)

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

    # Gaussian bump parameterization
    # h[n] = a0 + sum_{i=1}^{M} a_i * exp(-(n - c_i)^2 / (2 * w_i^2))
    M = 5  # Number of Gaussian bumps
    # Parameters: a0, then for each bump: (a_i, c_i, w_i)
    dim = 1 + 3 * M

    def reconstruct_gauss(x):
        """Reconstruct h from Gaussian bump parameters x."""
        h = np.full(N, x[0])  # DC component
        for i in range(M):
            a = x[1 + 3*i]
            c = x[2 + 3*i]
            w = x[3 + 3*i]
            # Ensure width is positive
            w = abs(w) + 0.1
            h += a * np.exp(-(n_idx - c)**2 / (2 * w**2))
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

    def gauss_objective(x, beta=120.0):
        """Objective in Gaussian parameter space."""
        h = reconstruct_gauss(x)
        f, _ = smooth_objective_and_grad_full(h, beta)
        return f

    def gauss_gradient(x, beta=120.0):
        """Gradient in Gaussian parameter space via chain rule."""
        h = reconstruct_gauss(x)
        f, grad_h = smooth_objective_and_grad_full(h, beta)
        if not np.isfinite(f):
            return np.zeros(dim)
        grad_x = np.zeros(dim)
        # DC term
        grad_x[0] = np.sum(grad_h)
        # Bump terms
        for i in range(M):
            a = x[1 + 3*i]
            c = x[2 + 3*i]
            w = abs(x[3 + 3*i]) + 0.1
            # d/d a_i
            bump = np.exp(-(n_idx - c)**2 / (2 * w**2))
            grad_x[1 + 3*i] = np.dot(grad_h, bump)
            # d/d c_i
            dbump_dc = bump * (n_idx - c) / w**2
            grad_x[2 + 3*i] = a * np.dot(grad_h, dbump_dc)
            # d/d w_i
            dbump_dw = bump * (n_idx - c)**2 / w**3
            grad_x[3 + 3*i] = a * np.dot(grad_h, dbump_dw) * (1 if x[3 + 3*i] >= 0 else -1)
        return grad_x

    rng = np.random.RandomState(42)

    # Generate diverse initial points in Gaussian parameter space
    initial_points = []

    # Seed: all ones -> a0 = 1, rest = 0
    x_seed = np.zeros(dim)
    x_seed[0] = 1.0
    initial_points.append(x_seed)

    # Random Gaussian coefficients
    for _ in range(25):
        x_rand = np.zeros(dim)
        x_rand[0] = rng.uniform(0.3, 2.0)  # Bias toward positive DC
        for i in range(M):
            x_rand[1 + 3*i] = rng.normal(0, 0.5)  # Amplitude
            x_rand[2 + 3*i] = rng.uniform(0, N)  # Center
            x_rand[3 + 3*i] = rng.uniform(10, 100)  # Width
        initial_points.append(x_rand)

    # Low-frequency dominant patterns (wide bumps)
    for center in [N//4, N//2, 3*N//4]:
        x_low = np.zeros(dim)
        x_low[0] = 1.0
        x_low[1] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.8)  # Amplitude
        x_low[2] = center  # Center
        x_low[3] = rng.uniform(50, 150)  # Wide width
        initial_points.append(x_low)

    # Alternating-like pattern (narrow bumps)
    x_alt = np.zeros(dim)
    x_alt[0] = 0.5
    for i in range(M):
        x_alt[1 + 3*i] = rng.choice([-1, 1]) * rng.uniform(0.2, 0.5)
        x_alt[2 + 3*i] = rng.uniform(0, N)
        x_alt[3 + 3*i] = rng.uniform(5, 20)  # Narrow width
    initial_points.append(x_alt)

    # Bump-like patterns with varying widths
    for center_freq in [N//4, N//2, 3*N//4]:
        for width in [10, 30, 60]:
            x_bump = np.zeros(dim)
            x_bump[0] = 1.0
            x_bump[1] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
            x_bump[2] = center_freq
            x_bump[3] = width
            initial_points.append(x_bump)

    # Sort by initial value
    def sort_key_gauss(x):
        h = reconstruct_gauss(x)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    initial_points.sort(key=sort_key_gauss)

    # Optimize from each initial point in Gaussian space
    for x0 in initial_points:
        if time.time() - start_time > DEADLINE:
            break
        try:
            result = minimize(
                gauss_objective,
                x0,
                jac=gauss_gradient,
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'gtol': 1e-8}
            )
            h_opt = reconstruct_gauss(result.x)
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        h_init = reconstruct_gauss(x0)
        val_init, valid = evaluate(h_init)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = h_init.copy()

    # Now refine in full h-space from the best Gaussian solution
    if time.time() - start_time < DEADLINE - 10:
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
