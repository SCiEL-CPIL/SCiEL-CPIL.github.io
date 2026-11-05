"""
Autocorrelation inequality optimizer.
Minimizes |2*N*max(conv(h,h))/sum(h)^2| over N=400 step heights.
Uses a structural approach: parameterize h as a smooth low-frequency function
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

    # Low-dimensional parameterization: h[n] = a0 + a1*n + a2*n^2 + sum of Gaussian bumps
    # Parameters: [a0, a1, a2, (center1, width1, amp1), (center2, width2, amp2), ...]
    
    def param_to_h(params, n_bumps=3):
        """Convert parameter vector to h sequence."""
        a0 = params[0]
        a1 = params[1]
        a2 = params[2]
        h = a0 + a1 * (n_idx / N - 0.5) + a2 * (n_idx / N - 0.5) ** 2
        for i in range(n_bumps):
            base = 3 + i * 3
            center = params[base]
            width = params[base + 1]
            amp = params[base + 2]
            h += amp * np.exp(-0.5 * ((n_idx / N - center) / width) ** 2)
        return h

    def fourier_param_to_h(params, n_modes=15):
        """Convert Fourier parameter vector to h sequence."""
        h = np.full(N, params[0])
        for k in range(1, n_modes):
            h += params[k] * np.cos(2 * np.pi * k * n_idx / N)
            h += params[n_modes + k - 1] * np.sin(2 * np.pi * k * n_idx / N)
        return h

    rng = np.random.RandomState(42)

    # Strategy 1: Optimize in low-dimensional Gaussian bump parameter space
    n_bumps = 3
    dim_bump = 3 + 3 * n_bumps  # a0, a1, a2, then 3 params per bump
    
    def bump_objective(params):
        h = param_to_h(params, n_bumps)
        f, _ = smooth_objective_and_grad_full(h)
        return f

    def bump_gradient(params):
        h = param_to_h(params, n_bumps)
        f, grad_h = smooth_objective_and_grad_full(h)
        if not np.isfinite(f):
            return np.zeros(dim_bump)
        # Compute gradient w.r.t. parameters
        grad_params = np.zeros(dim_bump)
        # a0: derivative is 1 for all n
        grad_params[0] = np.sum(grad_h)
        # a1: derivative is (n_idx/N - 0.5)
        grad_params[1] = np.dot(grad_h, n_idx / N - 0.5)
        # a2: derivative is (n_idx/N - 0.5)^2
        grad_params[2] = np.dot(grad_h, (n_idx / N - 0.5) ** 2)
        for i in range(n_bumps):
            base = 3 + i * 3
            center = params[base]
            width = params[base + 1]
            amp = params[base + 2]
            x = n_idx / N
            # d/dcenter: amp * exp(...) * (-(x-center)/width) * (1/N)
            d_center = amp * np.exp(-0.5 * ((x - center) / width) ** 2) * (-(x - center) / width) / N
            grad_params[base] = np.dot(grad_h, d_center)
            # d/dwidth: amp * exp(...) * (-(x-center)^2/width^2) * (1/N)
            d_width = amp * np.exp(-0.5 * ((x - center) / width) ** 2) * (-(x - center) ** 2 / width ** 2) / N
            grad_params[base + 1] = np.dot(grad_h, d_width)
            # d/damp: exp(...)
            d_amp = np.exp(-0.5 * ((x - center) / width) ** 2)
            grad_params[base + 2] = np.dot(grad_h, d_amp)
        return grad_params

    # Generate initial points for bump parameterization
    bump_inits = []
    # Seed-like: constant
    bump_inits.append(np.zeros(dim_bump))
    bump_inits[0][0] = 1.0
    # Random
    for _ in range(15):
        p = np.zeros(dim_bump)
        p[0] = rng.uniform(0.5, 2.0)
        p[1] = rng.uniform(-0.5, 0.5)
        p[2] = rng.uniform(-0.5, 0.5)
        for i in range(n_bumps):
            base = 3 + i * 3
            p[base] = rng.uniform(0.1, 0.9)  # center
            p[base + 1] = rng.uniform(0.05, 0.3)  # width
            p[base + 2] = rng.uniform(-1.0, 1.0)  # amplitude
        bump_inits.append(p)
    # Symmetric bumps
    for _ in range(5):
        p = np.zeros(dim_bump)
        p[0] = rng.uniform(0.5, 1.5)
        p[1] = 0.0
        p[2] = rng.uniform(-0.3, 0.3)
        for i in range(n_bumps):
            base = 3 + i * 3
            c = rng.uniform(0.15, 0.85)
            p[base] = c
            p[base + 1] = rng.uniform(0.08, 0.25)
            p[base + 2] = rng.uniform(-0.8, 0.8)
        bump_inits.append(p)

    # Sort by initial value
    def sort_key_bump(p):
        h = param_to_h(p, n_bumps)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    bump_inits.sort(key=sort_key_bump)

    # Optimize from each initial point
    for p0 in bump_inits:
        if time.time() - start_time > DEADLINE:
            break
        try:
            result = minimize(
                bump_objective,
                p0,
                jac=bump_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-8}
            )
            h_opt = param_to_h(result.x, n_bumps)
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        h_init = param_to_h(p0, n_bumps)
        val_init, valid = evaluate(h_init)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = h_init.copy()

    # Strategy 2: Optimize in low-dimensional Fourier parameter space
    n_modes = 12
    dim_fourier = 2 * n_modes

    def fourier_objective(params):
        h = fourier_param_to_h(params, n_modes)
        f, _ = smooth_objective_and_grad_full(h)
        return f

    def fourier_gradient(params):
        h = fourier_param_to_h(params, n_modes)
        f, grad_h = smooth_objective_and_grad_full(h)
        if not np.isfinite(f):
            return np.zeros(dim_fourier)
        grad_params = np.zeros(dim_fourier)
        grad_params[0] = np.sum(grad_h)
        for k in range(1, n_modes):
            grad_params[k] = np.dot(grad_h, np.cos(2 * np.pi * k * n_idx / N))
            grad_params[n_modes + k - 1] = np.dot(grad_h, np.sin(2 * np.pi * k * n_idx / N))
        return grad_params

    # Generate initial points for Fourier parameterization
    fourier_inits = []
    # Seed
    p_seed = np.zeros(dim_fourier)
    p_seed[0] = 1.0
    fourier_inits.append(p_seed)
    # Random
    for _ in range(15):
        p = np.zeros(dim_fourier)
        p[0] = rng.uniform(0.5, 2.0)
        p[1:] = rng.normal(0, 0.4, dim_fourier - 1)
        fourier_inits.append(p)
    # Low-frequency dominant
    for freq in [1, 2, 3, 5]:
        p = np.zeros(dim_fourier)
        p[0] = 1.0
        p[freq] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
        p[n_modes + freq - 1] = rng.choice([-1, 1]) * rng.uniform(0.3, 0.7)
        fourier_inits.append(p)

    def sort_key_fourier(p):
        h = fourier_param_to_h(p, n_modes)
        s = np.sum(h)
        if abs(s) < 1e-12:
            return np.inf
        conv = fast_convolve(h)
        return abs(2.0 * N * np.max(conv) / (s * s))

    fourier_inits.sort(key=sort_key_fourier)

    for p0 in fourier_inits:
        if time.time() - start_time > DEADLINE:
            break
        try:
            result = minimize(
                fourier_objective,
                p0,
                jac=fourier_gradient,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 500, 'gtol': 1e-8}
            )
            h_opt = fourier_param_to_h(result.x, n_modes)
            val_opt, valid = evaluate(h_opt)
            if valid and val_opt < best_value:
                best_value = val_opt
                best_h = h_opt.copy()
                if best_value < 1.0:
                    break
        except Exception:
            pass
        h_init = fourier_param_to_h(p0, n_modes)
        val_init, valid = evaluate(h_init)
        if valid and val_init < best_value:
            best_value = val_init
            best_h = h_init.copy()

    # Strategy 3: Full-space refinement from best solution
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

    # Strategy 4: Additional multi-start in full h-space
    if time.time() - start_time < DEADLINE - 10:
        h_inits = []
        # Random
        for _ in range(5):
            h_inits.append(rng.uniform(-1, 1, N))
        # Bimodal
        for _ in range(5):
            h = np.zeros(N)
            k = rng.randint(50, 300)
            idx = rng.choice(N, size=k, replace=False)
            h[idx] = rng.choice([-1, 1], size=k)
            h_inits.append(h)
        # Smooth random
        for _ in range(5):
            h = rng.uniform(-1, 1, N)
            h = np.convolve(h, np.ones(5)/5, mode='same')
            h_inits.append(h)
        # Cosine-based
        t = np.linspace(0, 1, N, endpoint=False)
        for freq in [1, 2, 3, 5, 7]:
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
                    options={'maxiter': 200, 'maxfun': 400, 'gtol': 1e-7}
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
        scales = [0.1, 0.05, 0.02, 0.01]
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
                            options={'maxiter': 100, 'maxfun': 200, 'gtol': 1e-8}
                        )
                        h_opt = result.x
                        val_opt, valid = evaluate(h_opt)
                        if valid and val_opt < best_value:
                            best_value = val_opt
                            best_h = h_opt.copy()
                    except Exception:
                        pass

    return best_h
