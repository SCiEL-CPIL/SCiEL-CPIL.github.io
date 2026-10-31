import numpy as np
import time
from scipy.optimize import minimize

N = 600

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def compute_metric(h):
        h = np.asarray(h, dtype=np.float64)
        total = np.sum(h)
        if total < 1e-15:
            return float('inf')
        h_norm = h / total
        H = np.fft.rfft(h_norm, n=2 * N - 1)
        acorr = np.fft.irfft(H * np.conj(H), n=2 * N - 1)
        return np.max(acorr)

    def build_gaussians(params, K):
        x = np.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        f = np.zeros(N)
        for k in range(K):
            c = params[3 * k]
            w = max(params[3 * k + 1], 1e-4)
            a = max(params[3 * k + 2], 0.0)
            f += a * np.exp(-0.5 * ((x - c) / w) ** 2)
        f = np.maximum(f, 0.0)
        total = np.sum(f)
        if total < 1e-15:
            return np.ones(N) / N
        return f / total

    def objective(params, K):
        f = build_gaussians(params, K)
        return compute_metric(f)

    # Phase 1: Parametric search with Gaussians
    best_h = np.ones(N) / N
    best_m = compute_metric(best_h)

    rng = np.random.RandomState(42)

    for K in [1, 2, 3, 4, 5]:
        if time.time() > deadline - 30:
            break

        n_inits = 8
        for init_idx in range(n_inits):
            if time.time() > deadline - 25:
                break

            if init_idx == 0:
                centers = np.linspace(-0.2, 0.2, K)
                widths = np.full(K, 0.08)
                amps = np.full(K, 1.0)
            else:
                centers = rng.uniform(-0.22, 0.22, K)
                widths = rng.uniform(0.03, 0.15, K)
                amps = rng.uniform(0.3, 2.0, K)

            params0 = np.concatenate([centers, widths, amps])

            try:
                result = minimize(
                    objective, params0, args=(K,),
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                f_cand = build_gaussians(result.x, K)
                m_cand = compute_metric(f_cand)
                if m_cand < best_m:
                    best_m = m_cand
                    best_h = f_cand.copy()
            except Exception:
                pass

    # Phase 2: Fast local refinement via random pair-moves
    h = best_h.copy()
    current_m = best_m
    local_deadline = deadline - 5.0

    np.random.seed(777)

    while time.time() < local_deadline:
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue

        # Try moving a fraction of mass from j to i
        for frac in [0.02, 0.05, 0.1, 0.2]:
            delta = h[j] * frac
            new_h = h.copy()
            new_h[j] -= delta
            new_h[i] += delta

            if new_h[j] < 0:
                continue

            m_new = compute_metric(new_h)
            if m_new < current_m:
                current_m = m_new
                h = new_h.copy()
                if current_m < best_m:
                    best_m = current_m
                    best_h = h.copy()
                break

    # Phase 3: Final IPF-style refinement if time allows
    if time.time() < deadline - 3:
        h_ipf = h.copy()
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            H = np.fft.rfft(h_ipf, n=2 * N - 1)
            acorr = np.fft.irfft(H * np.conj(H), n=2 * N - 1)
            peak_val = np.max(acorr)
            top_lags = np.argsort(acorr)[-5:]
            top_lags = top_lags[top_lags > 0]
            subgrad = np.zeros(N)
            for lag in top_lags:
                lag = int(lag)
                if lag < N:
                    contrib = np.zeros(N)
                    contrib[:N - lag] += h_ipf[lag:]
                    contrib[lag:] += h_ipf[:N - lag]
                    subgrad += contrib * (acorr[lag] / (peak_val + 1e-15))
            subgrad = subgrad / (np.max(subgrad) + 1e-15)
            lr = 0.005
            h_new = h_ipf - lr * subgrad
            h_new = np.maximum(h_new, 0.0)
            s = np.sum(h_new)
            if s < 1e-15:
                break
            h_new = h_new / s
            m_new = compute_metric(h_new)
            if m_new < current_m:
                current_m = m_new
                h_ipf = h_new.copy()
                if current_m < best_m:
                    best_m = current_m
                    best_h = h_ipf.copy()

    if best_h is None or np.sum(best_h) < 1e-15:
        best_h = np.ones(N) / N

    return list(best_h)
