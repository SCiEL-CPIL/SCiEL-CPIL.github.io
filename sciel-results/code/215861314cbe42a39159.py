import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _score(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    x_max = np.max(x)
    e = np.exp(alpha * (x - x_max))
    p = e / np.sum(e)
    return p

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    h_best = np.full(N, 0.5)
    score_best = _score(h_best)

    x = np.arange(N)

    for K in [2, 3, 4, 5, 6, 8, 10, 12]:
        if time.time() > deadline - 15:
            break

        n_params = 2 * K

        basis = np.zeros((N, 2*K))
        for k in range(1, K+1):
            basis[:, 2*(k-1)] = np.cos(2 * np.pi * k * x / N)
            basis[:, 2*(k-1)+1] = np.sin(2 * np.pi * k * x / N)

        def get_h(params):
            return 0.5 + basis @ params

        def project(h):
            h = h.copy()
            for _ in range(100):
                h = np.clip(h, 0.0, 1.0)
                diff = N/2.0 - np.sum(h)
                if abs(diff) < 1e-14:
                    break
                h += diff / N
            return h

        def objective(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)
            return _smooth_max(overlap, alpha=200.0)

        def objective_grad(params):
            h = get_h(params)
            h_c = project(h)
            shifted = np.zeros(N)
            shifted[:-1] = h_c[1:]
            overlap = h_c * (1.0 - shifted)

            p = _smooth_max_grad(overlap, alpha=200.0)

            grad_hc = np.zeros(N)
            h_next = np.zeros(N)
            h_next[:-1] = h_c[1:]
            grad_hc += p * (1.0 - h_next)
            h_prev = np.zeros(N)
            h_prev[1:] = h_c[:-1]
            p_prev = np.zeros(N)
            p_prev[1:] = p[:-1]
            grad_hc -= p_prev * h_c

            mask = (h > 0.01) & (h < 0.99)
            grad_h = grad_hc * mask

            return basis.T @ grad_h

        for init_seed in [42, 123, 456, 789, 1000]:
            if time.time() > deadline - 10:
                break

            np.random.seed(init_seed)
            h0 = np.random.randn(2*K) * 0.02

            try:
                res = minimize(
                    objective,
                    h0,
                    jac=objective_grad,
                    method='L-BFGS-B',
                    options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-12}
                )

                h_candidate = project(get_h(res.x))
                sc = _score(h_candidate)
                if sc < score_best:
                    score_best = sc
                    h_best = h_candidate.copy()

            except Exception:
                pass

    return h_best
