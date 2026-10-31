import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 140  # safety margin before 150s hard limit

def construct():
    start_time = time.time()

    def elapsed():
        return time.time() - start_time

    def fast_metric(h):
        """Compute normalized max using FFT for speed."""
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return np.max(c) / (s * s)

    def smooth_max(c, temp):
        """Smooth approximation of max(c) using log-sum-exp."""
        c_shifted = c / temp
        c_shifted = c_shifted - np.max(c_shifted)
        return temp * (np.max(c_shifted) + np.log(np.sum(np.exp(c_shifted))))

    def smooth_metric(h, temp):
        s = np.sum(h)
        if s < 1e-15:
            return 1e10
        nfft = 1024
        H = np.fft.rfft(h, n=nfft)
        C = np.fft.irfft(H * H, n=nfft)
        c = C[:2*N-1]
        return smooth_max(c, temp) / (s * s)

    n = np.arange(N)

    # Phase 1: Structured candidates
    candidates = []
    candidates.append(np.ones(N))

    x = np.linspace(0, np.pi, N)
    candidates.append(0.5 * (1 + np.cos(x)))

    center = N / 2 - 0.5
    width = 0.3
    sinc_h = np.sinc((n - center) / (N * width))
    sinc_h = np.clip(sinc_h, 0, None)
    if sinc_h.sum() > 0:
        candidates.append(sinc_h)

    xg = np.linspace(-3, 3, N)
    candidates.append(np.exp(-0.5 * xg**2))

    blackman = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    candidates.append(np.clip(blackman, 0, None))

    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hamming, 0, None))

    hanning = 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
    candidates.append(np.clip(hanning, 0, None))

    flat_top = (0.21557895 + 0.41663158 * np.cos(2*np.pi*n/(N-1)) +
                0.277263158 * np.cos(4*np.pi*n/(N-1)) +
                0.083578947 * np.cos(6*np.pi*n/(N-1)) +
                0.006947368 * np.cos(8*np.pi*n/(N-1)))
    candidates.append(np.clip(flat_top, 0, None))

    two_plat = np.ones(N)
    two_plat[N//4:3*N//4] = 0.5
    candidates.append(two_plat)

    inv_plat = np.ones(N)
    inv_plat[:N//4] = 0.3
    inv_plat[3*N//4:] = 0.3
    candidates.append(inv_plat)

    tri = np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
    candidates.append(tri)

    # Additional diverse candidates
    for freq in [1, 2, 3]:
        for amp in [0.05, 0.1, 0.2]:
            h_test = 1.0 + amp * np.cos(2 * np.pi * freq * n / N)
            h_test = np.clip(h_test, 0, None)
            if h_test.sum() > 0:
                candidates.append(h_test)

    best_val = 1e10
    best_h = np.ones(N)
    for h in candidates:
        if np.sum(h) < 1e-10:
            continue
        v = fast_metric(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()

    # Phase 2: Cosine parameterization optimization with adaptive temperature
    def cosine_basis(coeffs, order):
        c0 = coeffs[0]
        h = np.full(N, c0)
        for m in range(1, order + 1):
            cm = coeffs[1 + 2*(m-1)]
            dm = coeffs[2 + 2*(m-1)]
            h += cm * np.cos(2 * np.pi * m * n / N) + dm * np.sin(2 * np.pi * m * n / N)
        return h

    def fit_cosine(h, order):
        coeffs = np.zeros(1 + 2*order)
        coeffs[0] = np.mean(h)
        for m in range(1, order + 1):
            coeffs[1 + 2*(m-1)] = 2.0 * np.mean(h * np.cos(2 * np.pi * m * n / N))
            coeffs[2 + 2*(m-1)] = 2.0 * np.mean(h * np.sin(2 * np.pi * m * n / N))
        return coeffs

    def cosine_objective(coeffs, order, temp):
        if elapsed() > DEADLINE - 10:
            return 1e10
        h = cosine_basis(coeffs, order)
        h = np.clip(h, 0, None)
        if h.sum() < 1e-15:
            return 1e10
        return smooth_metric(h, temp)

    best_cosine_val = best_val
    best_cosine_h = best_h.copy()

    for order in [4, 6, 8, 10, 12]:
        if elapsed() > DEADLINE - 40:
            break

        coeffs0 = fit_cosine(best_h, order)

        # Multi-temperature annealing
        temps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
        current_coeffs = coeffs0.copy()

        for temp in temps:
            if elapsed() > DEADLINE - 20:
                break
            try:
                result = minimize(
                    lambda c: cosine_objective(c, order, temp),
                    current_coeffs,
                    method='L-BFGS-B',
                    options={'maxiter': 150, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 500}
                )
                if result.fun < 1e9:
                    current_coeffs = result.x
                    h_opt = cosine_basis(current_coeffs, order)
                    h_opt = np.clip(h_opt, 0, None)
                    v = fast_metric(h_opt)
                    if v < best_cosine_val:
                        best_cosine_val = v
                        best_cosine_h = h_opt.copy()
            except Exception:
                pass

    if best_cosine_val < best_val:
        best_val = best_cosine_val
        best_h = best_cosine_h.copy()

    # Phase 3: Full-vector gradient optimization with log-sum-exp surrogate
    def full_objective(x, temp):
        if elapsed() > DEADLINE - 10:
            return 1e10
        x_shifted = x - np.max(x)
        h = np.exp(x_shifted)
        return smooth_metric(h, temp)

    x0 = np.log(best_h + 1e-15)
    x0 = x0 - np.max(x0)

    for temp in [0.02, 0.01, 0.005, 0.002, 0.001]:
        if elapsed() > DEADLINE - 15:
            break
        try:
            result = minimize(
                lambda x: full_objective(x, temp),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 400}
            )
            if result.fun < 1e9:
                h_opt = np.exp(result.x - np.max(result.x))
                v = fast_metric(h_opt)
                if v < best_val:
                    best_val = v
                    best_h = h_opt.copy()
                    x0 = result.x - np.max(result.x)
        except Exception:
            pass

    # Phase 4: Coordinate descent with FFT-based metric
    current_h = best_h.copy()
    current_val = best_val
    np.random.seed(123)

    no_improve = 0
    max_no_improve = 5000

    while elapsed() < DEADLINE - 30 and no_improve < max_no_improve:
        idx = np.random.randint(N)
        orig = current_h[idx]

        mean_h = np.mean(current_h)
        step_factors = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
        step_sizes = step_factors * mean_h
        candidates_val = np.concatenate([orig + step_sizes, orig - step_sizes, [0.0]])
        candidates_val = np.array([max(0.0, v) for v in candidates_val])

        best_local_val = current_val
        best_local_v = orig

        for cv in candidates_val:
            if abs(cv - orig) < 1e-12:
                continue
            current_h[idx] = cv
            v = fast_metric(current_h)
            if v < best_local_val:
                best_local_val = v
                best_local_v = cv

        if best_local_val < current_val - 1e-12:
            current_h[idx] = best_local_v
            current_val = best_local_val
            no_improve = 0
            if current_val < best_val:
                best_val = current_val
                best_h = current_h.copy()
        else:
            current_h[idx] = orig
            no_improve += 1

        if no_improve % 300 == 0:
            pert = current_h.copy()
            pert += 0.05 * np.random.randn(N) * np.mean(current_h)
            pert = np.clip(pert, 0, None)
            if pert.sum() > 0:
                pv = fast_metric(pert)
                if pv < current_val:
                    current_h = pert
                    current_val = pv
                    no_improve = 0
                    if current_val < best_val:
                        best_val = current_val
                        best_h = current_h.copy()

    # Phase 5: Fine polish with tiny paired moves
    if elapsed() < DEADLINE - 5:
        step_frac = 0.002
        for _ in range(2000):
            if elapsed() > DEADLINE - 3:
                break
            i, j = np.random.choice(N, 2, replace=False)
            delta = step_frac * np.random.uniform(-1, 1) * np.mean(best_h)
            new_h = best_h.copy()
            new_h[i] += delta
            new_h[j] -= delta
            if new_h[i] >= 0 and new_h[j] >= 0:
                v = fast_metric(new_h)
                if v < best_val:
                    best_val = v
                    best_h = new_h

    best_h = np.maximum(best_h, 0)
    if np.sum(best_h) < 1e-15:
        best_h = np.ones(N)

    return best_h.tolist()
