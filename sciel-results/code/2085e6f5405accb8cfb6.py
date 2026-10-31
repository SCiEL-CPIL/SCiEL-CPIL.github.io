import numpy as np
import time
import random
from scipy.optimize import differential_evolution

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
        if time_left() < 15:
            break
        h = init / np.sum(init)
        val = compute_autoconv_max(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()

    if best_h is None:
        best_h = np.ones(N) / N
        best_val = compute_autoconv_max(best_h)

    # Stage 2: Differential Evolution on Gaussian Mixture Parameters
    # Represent h as a sum of K Gaussians with parameters (amplitude, center, width)
    if time_left() > 40:
        K = 8  # Number of Gaussian components
        # Parameters: for each Gaussian, [log_amp, center (0..1), log_width]
        # Total parameters: 3*K
        # We map center to [0, N-1] and width to a reasonable range
        
        x_grid = np.arange(N, dtype=float)
        
        def params_to_h(params):
            """Convert Gaussian mixture parameters to step function heights."""
            h = np.zeros(N)
            for k in range(K):
                log_amp = params[3*k]
                center_frac = params[3*k + 1]
                log_width = params[3*k + 2]
                
                amp = np.exp(log_amp)
                center = center_frac * (N - 1)
                width = np.exp(log_width)
                
                h += amp * np.exp(-0.5 * ((x_grid - center) / width) ** 2)
            return np.maximum(h, 1e-10)

        def objective_gm(params):
            """Objective function for differential evolution."""
            h = params_to_h(params)
            s = np.sum(h)
            if s < 1e-10:
                return 1e10
            h_norm = h / s
            val = compute_autoconv_max(h_norm)
            return val

        # Define bounds for each parameter
        # log_amp: [-5, 2] (amp in [0.006, 7.4])
        # center_frac: [0, 1]
        # log_width: [-2, 2] (width in [0.13, 7.4])
        bounds = []
        for k in range(K):
            bounds.append((-5, 2))   # log_amp
            bounds.append((0, 1))    # center_frac
            bounds.append((-2, 2))   # log_width

        # Initial population for DE
        # We can seed DE with parameters derived from our best_h or random
        # Let's just use random initialization within bounds, but maybe bias towards known good structures
        
        # Create a few good initial guesses for the population
        # 1. Uniform-like: many small Gaussians spread out
        # 2. Bimodal: two large Gaussians at 1/4 and 3/4
        
        # DE will handle the search, so we just need to run it.
        
        # To speed up DE, we can limit maxiter and popsize
        # Time budget: ~30s
        # N_params = 24
        # popsize = 10 * N_params = 240
        # Each eval is ~1ms (FFT based)
        # 240 * 24 * 1ms = ~5.7s per generation
        # We can afford ~5 generations in 30s? Maybe less.
        # Let's try maxiter=10, popsize=15
        
        try:
            # Use a fixed seed for reproducibility in DE if needed, but random is fine
            # scipy's DE uses random internally
            result_de = differential_evolution(
                objective_gm,
                bounds,
                maxiter=10,
                popsize=15,
                tol=1e-6,
                seed=42,
                polish=False,  # Disable polish, we'll do our own refinement
                init='latinhypercube'
            )
            
            if result_de.success or result_de.fun < best_val:
                best_params = result_de.x
                h_de = params_to_h(best_params)
                h_de = h_de / np.sum(h_de)
                val_de = compute_autoconv_max(h_de)
                if val_de < best_val:
                    best_val = val_de
                    best_h = h_de.copy()
                    
        except Exception as e:
            pass

    # Stage 3: Cosine series parameterization with Remez-like equioscillation
    if time_left() > 30:
        M = 12  # Number of cosine terms
        H_freq = np.fft.rfft(best_h)
        coeffs = np.zeros(M + 1)
        coeffs[0] = H_freq[0].real / N
        for m in range(1, M + 1):
            if m < len(H_freq):
                coeffs[m] = 2 * H_freq[m].real / N

        def coeffs_to_h(c):
            """Convert cosine coefficients to heights."""
            h = np.zeros(N)
            for m in range(len(c)):
                h += c[m] * np.cos(2 * np.pi * m * i / N)
            return h

        def compute_grad_coeff(c, beta=30.0):
            """Compute gradient of smooth max autoconvolution w.r.t. coefficients."""
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
            exp_conv = np.exp((conv - m) * beta)
            weights = exp_conv / np.sum(exp_conv)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            grad_c = np.zeros(len(c))
            for m in range(len(c)):
                grad_c[m] = np.dot(grad_h, np.cos(2 * np.pi * m * i / N))

            return grad_c, m + np.log(np.sum(exp_conv)) / beta

        # Gradient descent with adaptive step size
        current_coeffs = coeffs.copy()
        current_val = best_val

        eta = 0.1
        for k in range(200):
            if time_left() < 10:
                break

            h_curr = coeffs_to_h(current_coeffs)
            h_curr = np.maximum(h_curr, 1e-10)
            h_curr = h_curr / np.sum(h_curr)
            curr_val = compute_autoconv_max(h_curr)

            if curr_val < best_val:
                best_val = curr_val
                best_h = h_curr.copy()

            grad_c, smooth_val = compute_grad_coeff(current_coeffs)

            grad_norm = np.linalg.norm(grad_c)
            if grad_norm > 1e-10:
                grad_c = grad_c / grad_norm

            new_coeffs = current_coeffs - eta * grad_c

            new_h = coeffs_to_h(new_coeffs)
            if np.any(new_h < -1e-6):
                eta *= 0.5
                if eta < 1e-8:
                    break
                continue

            new_h = np.maximum(new_h, 0)
            new_h = new_h / np.sum(new_h)
            new_val = compute_autoconv_max(new_h)

            if new_val < curr_val:
                current_coeffs = new_coeffs
                eta = min(eta * 1.1, 0.5)
            else:
                eta *= 0.7

        final_h = coeffs_to_h(current_coeffs)
        final_h = np.maximum(final_h, 0)
        if np.sum(final_h) > 0:
            final_h = final_h / np.sum(final_h)
            final_val = compute_autoconv_max(final_h)
            if final_val < best_val:
                best_val = final_val
                best_h = final_h.copy()

    # Stage 4: IPF-inspired optimization to minimize L1/L2 norm of autoconvolution
    if time_left() > 40:
        current_h = best_h.copy()
        current_val = compute_autoconv_max(current_h)

        def compute_l1_grad(h, beta=10.0):
            """Compute gradient of smooth L1 surrogate."""
            conv = compute_autoconv(h)
            abs_conv = np.abs(conv)
            m = np.max(abs_conv)
            if m < 1e-15:
                return np.zeros_like(h), 0.0

            exp_vals = np.exp((abs_conv - m) * beta)
            weights = exp_vals / np.sum(exp_vals)

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = h
            H = np.fft.fft(h_pad, n=size)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights * np.sign(conv)
            W = np.fft.fft(w_full, n=size)
            grad_h = 2.0 * np.fft.ifft(H * W).real[:N]

            val = m + np.log(np.sum(exp_vals)) / beta
            return grad_h, val

        n_ipf = int(min(300, time_left() * 3))
        for k in range(n_ipf):
            if time_left() < 15:
                break

            grad_l1, l1_val = compute_l1_grad(current_h, beta=15.0)

            conv = compute_autoconv(current_h)
            m = np.max(conv)
            exp_conv = np.exp((conv - m) * 20.0)
            weights = exp_conv / np.sum(exp_conv)

            size = 2 * N
            h_pad = np.zeros(size)
            h_pad[:N] = current_h
            H = np.fft.fft(h_pad, n=size)

            w_full = np.zeros(size)
            w_full[:len(conv)] = weights
            W = np.fft.fft(w_full, n=size)
            grad_peak = 2.0 * np.fft.ifft(H * W).real[:N]

            combined_grad = 0.7 * grad_l1 + 0.3 * grad_peak

            grad_norm = np.linalg.norm(combined_grad)
            if grad_norm > 1e-10:
                combined_grad = combined_grad / grad_norm

            eta = 0.05
            new_h = current_h - eta * combined_grad
            new_h = np.maximum(new_h, 0)
            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h = new_h / s

            new_val = compute_autoconv_max(new_h)
            if new_val < current_val:
                current_h = new_h
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_h = current_h.copy()
            else:
                eta *= 0.5
                new_h = current_h - eta * combined_grad
                new_h = np.maximum(new_h, 0)
                s = np.sum(new_h)
                if s > 1e-10:
                    new_h = new_h / s
                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val:
                        current_h = new_h
                        current_val = new_val
                        if current_val < best_val:
                            best_val = current_val
                            best_h = current_h.copy()

    # Stage 5: Simulated Annealing refinement
    if time_left() > 20:
        current_h = best_h.copy()
        current_val = best_val
        best_h_sa = best_h.copy()
        best_val_sa = best_val

        T = 0.01
        T_min = 1e-6
        cooling = 0.995
        n_iter = int(min(5000, time_left() * 50))

        for k in range(n_iter):
            if time_left() < 5:
                break

            n_perturb = random.randint(1, 5)
            indices = random.sample(range(N), n_perturb)

            new_h = current_h.copy()
            for idx in indices:
                noise = random.gauss(0, T * 0.1)
                new_h[idx] = max(0, current_h[idx] + noise)

            s = np.sum(new_h)
            if s < 1e-10:
                continue
            new_h /= s

            new_val = compute_autoconv_max(new_h)

            if new_val < current_val or random.random() < np.exp(-(new_val - current_val) / T):
                current_h = new_h
                current_val = new_val

                if current_val < best_val_sa:
                    best_val_sa = current_val
                    best_h_sa = current_h.copy()

            T *= cooling
            if T < T_min:
                T = T_min

        if best_val_sa < best_val:
            best_val = best_val_sa
            best_h = best_h_sa

    # Stage 6: Local coordinate descent refinement
    if time_left() > 15:
        current_h = best_h.copy()
        current_val = best_val
        improved = True
        max_rounds = int(min(20, time_left() // 2))

        for round_num in range(max_rounds):
            if time_left() < 5:
                break

            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time_left() < 3:
                    break

                for delta in [0.1, 0.05, 0.02, -0.1, -0.05, -0.02]:
                    new_h = current_h.copy()
                    new_h[idx] = max(0, current_h[idx] + delta * current_h[idx])
                    s = np.sum(new_h)
                    if s < 1e-10:
                        continue
                    new_h /= s

                    new_val = compute_autoconv_max(new_h)
                    if new_val < current_val - 1e-12:
                        current_h = new_h
                        current_val = new_val
                        improved = True
                        break

            if not improved:
                break

        if current_val < best_val:
            best_val = current_val
            best_h = current_h

    return best_h.tolist()
