import numpy as np
import time
import random
from scipy.optimize import minimize, differential_evolution

N = 600
DEADLINE = 145

def construct():
    start_time = time.time()

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def compute_autoconv_max(h):
        """Compute max of autoconvolution of h (assumed sum=1)."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return np.max(conv)

    def compute_autoconv(h):
        """Compute full autoconvolution of h."""
        size = 2 * N
        h_pad = np.zeros(size)
        h_pad[:N] = h
        H = np.fft.fft(h_pad, n=size)
        conv = np.fft.ifft(H * H).real
        return conv

    # Parametric families for initialization
    i = np.arange(N, dtype=float)
    center = (N - 1) / 2.0

    def make_bimodal(sigma, alpha):
        """Bimodal: two Gaussians at positions alpha*N and (1-alpha)*N"""
        pos1 = alpha * N
        pos2 = (1 - alpha) * N
        g1 = np.exp(-((i - pos1) ** 2) / (2 * sigma ** 2))
        g2 = np.exp(-((i - pos2) ** 2) / (2 * sigma ** 2))
        return g1 + g2

    def make_poly(c, d, e):
        """Polynomial: c + d*(x-center)^2 + e*(x-center)^4, clipped to nonneg"""
        x = (i - center) / (N / 2)
        val = c + d * x**2 + e * x**4
        return np.maximum(val, 0.01)

    # Generate diverse initializations
    inits = []

    # Uniform
    inits.append(np.ones(N))

    # Bimodal with various parameters
    for sigma in [N/10, N/8, N/6, N/5]:
        for alpha in [0.1, 0.15, 0.2, 0.25, 0.3]:
            inits.append(make_bimodal(sigma, alpha))

    # Polynomial shapes
    for c, d, e in [(1, 1, 0), (1, 2, 0), (1, 1, 0.5), (1, 0.5, 1), (2, 1, 0), (1, 3, 0)]:
        inits.append(make_poly(c, d, e))

    # U-shaped
    u_shape = 1.0 + 2.0 * ((i - center) / (N/2))**2
    inits.append(u_shape)

    # W-shape (two bumps)
    w_shape = 1.0 + 1.5 * np.cos(4 * np.pi * i / N)
    inits.append(np.maximum(w_shape, 0.01))

    # Random smooth
    for _ in range(5):
        freqs = np.random.choice([1, 2, 3, 4, 5], size=3, replace=False)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        amps = np.random.uniform(0.5, 1.5, size=3)
        random_shape = np.ones(N)
        for f, p, a in zip(freqs, phases, amps):
            random_shape += a * np.cos(2 * np.pi * f * i / N + p)
        inits.append(np.maximum(random_shape, 0.01))

    best_h = None
    best_val = np.inf

    # Stage 1: Evaluate all initializations
    for init in inits:
        if time_left() < 10:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)

    # Stage 2: Cosine series parameterization with L-BFGS-B optimization
    if time_left() > 30:
        M = 20  # Number of cosine terms (increased from 12)
        H_freq = np.fft.rfft(best_h)
        coeffs = np.zeros(M + 1)
        coeffs[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                coeffs[m] = 2 * H_freq[m].real / N

        # Precompute cosine basis
        cos_basis = np.cos(2 * np.pi * np.outer(np.arange(M + 1), i) / N)

        def coeffs_to_h(c):
            """Convert cosine coefficients to heights."""
            return cos_basis @ c

        def objective_and_grad(c):
            """Compute smooth max autoconvolution and its gradient w.r.t. coefficients."""
            h = coeffs_to_h(c)
            h = np.maximum(h, 1e-10)
            s = np.sum(h)
            h_norm = h / s

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h_norm
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real

            m = np.max(conv)
            beta = 50.0
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            # Chain rule: grad_c = cos_basis^T @ grad_h / s
            grad_c = cos_basis.T @ grad_h / s

            val = m + np.log(np.sum(exp_conv)) / beta
            return val, grad_c

        # Use L-BFGS-B for optimization
        try:
            result = minimize(
                lambda c: objective_and_grad(c)[0],
                coeffs.copy(),
                jac=lambda c: objective_and_grad(c)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 200}
            )
            if result.success and result.fun < best_val:
                final_h = coeffs_to_h(result.x)
                final_h = np.maximum(final_h, 0)
                if np.sum(final_h) > 0:
                    final_h = final_h / np.sum(final_h)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < best_val:
                        best_val = final_val
                        best_h = final_h.copy()
        except Exception:
            pass

    # Stage 3: L-BFGS-B on full h with smooth peak objective
    if time_left() > 40:
        def smooth_peak_obj_and_grad(h):
            """Smooth max autoconvolution with gradient."""
            h = np.maximum(h, 1e-10)
            h = h / np.sum(h)

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)
            conv = np.fft.ifft(H * H).real

            m = np.max(conv)
            beta = 50.0
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            val = m + np.log(np.sum(exp_conv)) / beta
            return val, grad_h

        try:
            result = minimize(
                lambda h: smooth_peak_obj_and_grad(h)[0],
                best_h.copy(),
                jac=lambda h: smooth_peak_obj_and_grad(h)[1],
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 300}
            )
            if result.success and result.fun < best_val:
                final_h = np.maximum(result.x, 0)
                if np.sum(final_h) > 0:
                    final_h = final_h / np.sum(final_h)
                    final_val = compute_autoconv_max(final_h)
                    if final_val < best_val:
                        best_val = final_val
                        best_h = final_h.copy()
        except Exception:
            pass

    # Stage 4: Multi-restart L-BFGS-B with different initializations
    if time_left() > 20:
        n_restarts = int(min(5, time_left() // 5))
        for r in range(n_restarts):
            if time_left() < 10:
                break

            # Create perturbed initialization
            h_init = best_h.copy()
            # Add smooth random perturbation
            freqs = np.random.choice([2, 3, 4, 5, 6], size=2, replace=False)
            phases = np.random.uniform(0, 2 * np.pi, size=2)
            amps = np.random.uniform(0.05, 0.15, size=2)
            for f, p, a in zip(freqs, phases, amps):
                h_init += a * np.cos(2 * np.pi * f * i / N + p)
            h_init = np.maximum(h_init, 0)
            h_init = h_init / np.sum(h_init)

            try:
                result = minimize(
                    lambda h: smooth_peak_obj_and_grad(h)[0],
                    h_init,
                    jac=lambda h: smooth_peak_obj_and_grad(h)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 150}
                )
                if result.success:
                    final_h = np.maximum(result.x, 0)
                    if np.sum(final_h) > 0:
                        final_h = final_h / np.sum(final_h)
                        final_val = compute_autoconv_max(final_h)
                        if final_val < best_val:
                            best_val = final_val
                            best_h = final_h.copy()
            except Exception:
                pass

    # Stage 5: Fine-grained coordinate descent with line search
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(10, time_left() // 2))

        for round_num in range(max_rounds):
            if time_left() < 5:
                break

            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break

                # Try both increasing and decreasing
                for direction in [1, -1]:
                    # Binary search for best step
                    lo, hi = 0.0, 0.5
                    best_step = 0.0
                    best_new_val = current_val
                    for _ in range(6):
                        mid = (lo + hi) / 2
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] * (1 + direction * mid))
                        s = np.sum(new_h)
                        if s < 1e-10:
                            break
                        new_h /= s
                        new_val = compute_autoconv_max(new_h)
                        if new_val < best_new_val:
                            best_new_val = new_val
                            best_step = mid
                            lo = mid
                        else:
                            hi = mid

                    if best_step > 0 and best_new_val < current_val - 1e-12:
                        new_h = current_h.copy()
                        new_h[idx] = max(0, current_h[idx] * (1 + direction * best_step))
                        s = np.sum(new_h)
                        new_h /= s
                        current_h = new_h
                        current_val = best_new_val
                        improved = True
                        break

            if not improved:
                break

        if current_val < best_val:
            best_val = current_val
            best_h = current_h

    return best_h.tolist()
