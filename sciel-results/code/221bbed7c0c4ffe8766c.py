import numpy as np
import time
import random
from scipy.optimize import minimize

N = 600

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def time_left():
        return deadline - time.time()

    def compute_metric(h):
        """Compute normalized max of autoconvolution: max(f*f) / (sum(h))^2"""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        peak = np.max(acorr)
        return peak / (total * total)

    def smooth_max_metric(h, temperature=0.01):
        """Smooth surrogate for max using log-sum-exp, scaled to approximate max."""
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total == 0:
            return float('inf')
        size = 2 * N
        H = np.fft.rfft(h, n=size)
        acorr = np.fft.irfft(H * np.conj(H), n=size)
        acorr = np.maximum(acorr, 0)
        # log-sum-exp trick
        a = acorr / temperature
        a_max = np.max(a)
        log_sum_exp = a_max + np.log(np.sum(np.exp(a - a_max)))
        smooth_max = temperature * log_sum_exp
        return smooth_max / (total * total)

    def gaussian_mixture_to_heights(params, K):
        """Convert Gaussian mixture parameters to step heights."""
        i = np.arange(N, dtype=np.float64)
        h = np.zeros(N, dtype=np.float64)
        for j in range(K):
            w_j = max(params[3 * j], 0.0)
            c_j = params[3 * j + 1]
            s_j = max(params[3 * j + 2], 1e-4)
            h += w_j * np.exp(-((i - c_j) ** 2) / (2.0 * s_j ** 2))
        return h

    def gaussian_optimize(K, n_restarts, max_iters=200, use_smooth=True):
        """Optimize Gaussian mixture with smooth max surrogate via L-BFGS-B."""
        best_h = None
        best_m = float('inf')
        mid = (N - 1) / 2.0

        for restart in range(n_restarts):
            if time_left() < 15:
                break

            # Initialize parameters
            if restart == 0:
                # Uniform-ish: spread centers
                centers = np.linspace(0, N - 1, K)
                weights = np.ones(K) * 1.0
                sigmas = np.ones(K) * (N / 6.0)
            elif restart == 1:
                # Bimodal
                centers = np.concatenate([
                    np.linspace(0, N // 4, K // 2),
                    np.linspace(3 * N // 4, N - 1, K - K // 2)
                ])
                weights = np.ones(K) * 1.0
                sigmas = np.ones(K) * (N / 12.0)
            elif restart == 2:
                # Tri-modal
                centers = np.concatenate([
                    np.linspace(0, N // 6, K // 3),
                    np.linspace(N // 3, 2 * N // 3, K // 3),
                    np.linspace(5 * N // 6, N - 1, K - 2 * (K // 3))
                ])
                weights = np.ones(K) * 1.0
                sigmas = np.ones(K) * (N / 15.0)
            elif restart == 3:
                # Random
                centers = np.random.uniform(0, N - 1, K)
                weights = np.random.uniform(0.5, 2.0, K)
                sigmas = np.random.uniform(N / 20, N / 4, K)
            else:
                # Random with narrower sigmas
                centers = np.random.uniform(0, N - 1, K)
                weights = np.random.uniform(0.3, 1.5, K)
                sigmas = np.random.uniform(N / 30, N / 6, K)

            params = np.concatenate([weights, centers, sigmas])

            def objective(p):
                if time_left() < 10:
                    return 1e10
                h = gaussian_mixture_to_heights(p, K)
                total = np.sum(h)
                if total < 1e-10:
                    return 1e10
                h_norm = h / total
                if use_smooth:
                    return smooth_max_metric(h_norm, temperature=0.01)
                else:
                    return compute_metric(h_norm)

            def gradient(p):
                if time_left() < 10:
                    return np.zeros_like(p)
                h = gaussian_mixture_to_heights(p, K)
                total = np.sum(h)
                if total < 1e-10:
                    return np.zeros_like(p)
                h_norm = h / total

                # Numerical gradient for the smooth objective
                eps = 1e-6
                grad = np.zeros_like(p)
                for k in range(len(p)):
                    p_plus = p.copy()
                    p_plus[k] += eps
                    h_plus = gaussian_mixture_to_heights(p_plus, K)
                    total_plus = np.sum(h_plus)
                    if total_plus < 1e-10:
                        grad[k] = 0
                        continue
                    h_plus_norm = h_plus / total_plus
                    f_plus = smooth_max_metric(h_plus_norm, temperature=0.01)

                    p_minus = p.copy()
                    p_minus[k] = max(p_minus[k] - eps, 1e-8)
                    h_minus = gaussian_mixture_to_heights(p_minus, K)
                    total_minus = np.sum(h_minus)
                    if total_minus < 1e-10:
                        grad[k] = 0
                        continue
                    h_minus_norm = h_minus / total_minus
                    f_minus = smooth_max_metric(h_minus_norm, temperature=0.01)

                    grad[k] = (f_plus - f_minus) / (2 * eps)
                return grad

            bounds = []
            for j in range(K):
                bounds.append((0.01, 10.0))   # w_j
                bounds.append((0.0, N - 1))   # c_j
                bounds.append((1.0, N / 2))   # s_j

            try:
                res = minimize(
                    objective,
                    params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iters, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                h_opt = gaussian_mixture_to_heights(res.x, K)
                total = np.sum(h_opt)
                if total > 1e-10:
                    h_opt = h_opt / total
                    m_opt = compute_metric(h_opt)
                    if m_opt < best_m:
                        best_m = m_opt
                        best_h = h_opt.copy()
            except Exception:
                pass

        return best_h, best_m

    def direct_polish(h_init, max_iters=50000):
        """Direct coordinate descent: pick random index, adjust height slightly."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        rng = np.random.RandomState(123)

        for it in range(max_iters):
            if time_left() < 8:
                break

            idx = rng.randint(N)
            # Try small perturbations
            for delta_frac in [0.02, -0.02, 0.05, -0.05, 0.01, -0.01]:
                delta = h[idx] * delta_frac
                if h[idx] + delta < 0:
                    continue
                new_h = h.copy()
                new_h[idx] += delta
                m_new = compute_metric(new_h)
                if m_new < best_m - 1e-12:
                    best_m = m_new
                    best_h = new_h.copy()
                    h = new_h
                    break

        return best_h, best_m

    def spectral_polish(h_init, max_iters=1000):
        """Quick spectral flattening as a final polish."""
        h = h_init.copy().astype(np.float64)
        best_h = h.copy()
        best_m = compute_metric(h)
        size = 2 * N

        for it in range(max_iters):
            if time_left() < 8:
                break
            H = np.fft.rfft(h, n=size)
            power = np.abs(H) ** 2
            mean_power = np.mean(power)
            threshold = mean_power * 1.3
            excess = np.maximum(power - threshold, 0)
            power_new = np.maximum(power - 0.8 * excess, 0)
            phase = np.angle(H)
            H_new = np.sqrt(power_new) * np.exp(1j * phase)
            h_new = np.fft.irfft(H_new, n=size)[:N]
            h_new = np.maximum(h_new, 0.0)
            if np.all(h_new == 0):
                break
            m_new = compute_metric(h_new)
            if m_new < best_m:
                best_m = m_new
                best_h = h_new.copy()
            h = h_new

        return best_h, best_m

    # Phase 1: Gaussian mixture optimization with multiple K values and restarts
    best_h = None
    best_m = float('inf')

    K_values = [6, 10, 14, 20]
    for K in K_values:
        if time_left() < 25:
            break
        h_opt, m_opt = gaussian_optimize(K, n_restarts=4, max_iters=150, use_smooth=True)
        if h_opt is not None and m_opt < best_m:
            best_m = m_opt
            best_h = h_opt

    # Phase 2: Direct coordinate descent polish
    if best_h is not None and time_left() > 20:
        h_ref, m_ref = direct_polish(best_h, max_iters=30000)
        if m_ref < best_m:
            best_m = m_ref
            best_h = h_ref

    # Phase 3: Spectral polish
    if best_h is not None and time_left() > 12:
        h_spec, m_spec = spectral_polish(best_h, max_iters=800)
        if m_spec < best_m:
            best_m = m_spec
            best_h = h_spec

    # Phase 4: Final direct polish
    if best_h is not None and time_left() > 10:
        h_final, m_final = direct_polish(best_h, max_iters=20000)
        if m_final < best_m:
            best_m = m_final
            best_h = h_final

    return list(best_h)

# EVOLVE-BLOCK-END
