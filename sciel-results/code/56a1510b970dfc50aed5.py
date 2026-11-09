"""Second autocorrelation inequality: maximize ratio via smooth optimization with u-substitution."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    x = np.linspace(-0.25, 0.25, N)
    dx = x[1] - x[0]

    def compute_ratio(h):
        """Compute the target ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) for h >= 0."""
        h = np.asarray(h, dtype=float)
        h = np.maximum(h, 0.0)
        norm2_sq = np.sum(h**2)
        if norm2_sq < 1e-15:
            return 0.0
        norm1 = np.sum(h)
        if norm1 < 1e-15:
            return 0.0
        ac = np.convolve(h, h)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 0.0
        return norm2_sq / (norm1 * sup_ac)

    def ratio_from_u(u):
        """Compute ratio given u where h = u^2."""
        h = u**2
        return compute_ratio(h)

    def soft_max(v, beta):
        """Smooth max via log-sum-exp."""
        vmax = np.max(v)
        if vmax < 1e-30:
            return 1e-30
        shifted = v / beta
        shifted = shifted - np.max(shifted)
        s = np.sum(np.exp(shifted))
        return beta * (np.log(s) + np.max(shifted))

    def smooth_obj_u(u, beta):
        """Minimize ||h||_1 * soft_max(|h*h|) subject to h = u^2, ||h||_2 = 1."""
        h = u**2
        norm2_sq = np.sum(h)
        if norm2_sq < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(norm2_sq)
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        smax = soft_max(np.abs(ac), beta)
        if norm1 < 1e-15 or smax < 1e-15:
            return 1e10
        return norm1 * smax

    def smooth_grad_u(u, beta):
        """Numerical gradient of smooth_obj_u with respect to u."""
        eps = 1e-6
        grad = np.zeros(N)
        f0 = smooth_obj_u(u, beta)
        for i in range(N):
            u_p = u.copy()
            u_p[i] += eps
            fp = smooth_obj_u(u_p, beta)
            grad[i] = (fp - f0) / eps
        return grad

    def nonsmooth_obj_u(u):
        """Minimize ||h||_1 * ||h*h||_inf subject to h = u^2, ||h||_2 = 1."""
        h = u**2
        norm2_sq = np.sum(h)
        if norm2_sq < 1e-15:
            return 1e10
        h_norm = h / np.sqrt(norm2_sq)
        norm1 = np.sum(h_norm)
        ac = np.convolve(h_norm, h_norm)
        sup_ac = np.max(np.abs(ac))
        if sup_ac < 1e-15:
            return 1e10
        return norm1 * sup_ac

    # Multi-start optimization
    best_score = 0.0
    best_h = None

    rng = np.random.RandomState(42)
    n_starts = 20

    # Initialization strategies for h (nonnegative), then convert to u = sqrt(h)
    def init_h(s):
        idx = np.arange(N)
        mid = N / 2.0
        if s == 0:
            return np.ones(N)
        elif s == 1:
            return np.exp(-((idx - mid)**2) / (2 * (N/4)**2))
        elif s == 2:
            h = np.zeros(N)
            h[0] = 1.0
            h[-1] = 1.0
            return h
        elif s == 3:
            h = np.zeros(N)
            idx_sel = rng.choice(N, size=N//5, replace=False)
            h[idx_sel] = 1.0
            return h
        elif s == 4:
            h = 1.0 - np.abs(idx - mid) / (mid)
            return np.maximum(h, 0)
        elif s == 5:
            return np.cos(np.pi * idx / N)**2
        elif s == 6:
            return np.abs(rng.randn(N)) + 0.1
        elif s == 7:
            h = np.zeros(N)
            h[N//4] = 1.0
            h[3*N//4] = 1.0
            return h
        elif s == 8:
            return np.exp(-((idx - mid)**2) / (2 * (N/8)**2))
        elif s == 9:
            return np.exp(-((idx - mid)**2) / (2 * (N/2)**2))
        elif s == 10:
            h = np.sinc(np.linspace(-1, 1, N))
            return np.maximum(h, 0)
        elif s == 11:
            h = np.zeros(N)
            h[N//3:2*N//3] = 1.0
            return h
        elif s == 12:
            h = np.zeros(N)
            h[::2] = 1.0
            return h
        elif s == 13:
            h = np.zeros(N)
            h[::3] = 1.0
            return h
        elif s == 14:
            return np.concatenate([np.linspace(0, 1, N//2), np.linspace(1, 0, N - N//2)])
        elif s == 15:
            return np.exp(-((idx - N*0.3)**2) / (2*(N*0.12)**2)) + np.exp(-((idx - N*0.7)**2) / (2*(N*0.12)**2))
        elif s == 16:
            return np.abs(np.sin(np.pi * idx / N))**2
        elif s == 17:
            return np.ones(N) * (1.0 + 0.3 * rng.randn(N))
        elif s == 18:
            return np.exp(-((idx - mid)**2) / (2 * (N/6)**2)) * (1.0 + 0.2 * rng.randn(N))
        else:
            return np.abs(rng.randn(N)) * np.exp(-((idx - mid)**2) / (2 * (N/3)**2))

    for s in range(n_starts):
        if time.time() > deadline - 5:
            break

        h0 = init_h(s)
        h0 = np.maximum(h0, 0.0)
        n2 = np.sum(h0**2)
        if n2 < 1e-15:
            continue
        h0 = h0 / np.sqrt(n2)
        u0 = np.sqrt(h0)

        # Phase 1: Direct optimization on h with SLSQP (nonsmooth ratio)
        if time.time() < deadline - 15:
            def obj_h(h):
                return -compute_ratio(h)

            bounds = [(0.0, None)] * N
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h**2) - 1.0,
                            'jac': lambda h: 2 * h}]
            try:
                res = minimize(obj_h, h0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 150, 'ftol': 1e-12})
                h_opt = np.maximum(res.x, 0.0)
                n2 = np.sum(h_opt**2)
                if n2 > 1e-15:
                    h_opt = h_opt / np.sqrt(n2)
                    score = compute_ratio(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass

        # Phase 2: Optimization on u = sqrt(h) with L-BFGS-B (smooth surrogate)
        if time.time() < deadline - 10:
            for beta in [0.03, 0.08, 0.15]:
                if time.time() > deadline - 5:
                    break

                u0_cur = np.sqrt(np.maximum(best_h, 0)) if best_h is not None else u0

                try:
                    res = minimize(lambda u: smooth_obj_u(u, beta), u0_cur,
                                  method='L-BFGS-B',
                                  jac=lambda u: smooth_grad_u(u, beta),
                                  options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-7})
                    h_opt = res.x**2
                    n2 = np.sum(h_opt)
                    if n2 > 1e-15:
                        h_opt = h_opt / np.sqrt(n2)
                        score = compute_ratio(h_opt)
                        if score > best_score:
                            best_score = score
                            best_h = h_opt.copy()
                except Exception:
                    pass

        # Phase 3: Direct ratio maximization on u with Nelder-Mead (derivative-free)
        if time.time() < deadline - 8:
            u0_cur = np.sqrt(np.maximum(best_h, 0)) if best_h is not None else u0
            try:
                res = minimize(lambda u: -ratio_from_u(u), u0_cur,
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
                h_opt = res.x**2
                n2 = np.sum(h_opt)
                if n2 > 1e-15:
                    h_opt = h_opt / np.sqrt(n2)
                    score = compute_ratio(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt.copy()
            except Exception:
                pass

    # Final polish
    if best_h is not None and time.time() < deadline - 5.0:
        # Perturbation search
        for _ in range(30):
            if time.time() > deadline - 2:
                break
            h_pert = best_h + 0.005 * rng.standard_normal(N)
            h_pert = np.maximum(h_pert, 0)
            n2 = np.sum(h_pert**2)
            if n2 < 1e-15:
                continue
            h_pert = h_pert / np.sqrt(n2)
            score = compute_ratio(h_pert)
            if score > best_score:
                best_score = score
                best_h = h_pert

        # High-precision SLSQP polish
        if time.time() < deadline - 3:
            def obj_final(h):
                return -compute_ratio(h)

            bounds = [(0.0, None)] * N
            constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h**2) - 1.0,
                            'jac': lambda h: 2 * h}]
            try:
                res = minimize(obj_final, best_h, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 500, 'ftol': 1e-14})
                h_polish = np.maximum(res.x, 0.0)
                n2 = np.sum(h_polish**2)
                if n2 > 1e-15:
                    h_polish = h_polish / np.sqrt(n2)
                    score = compute_ratio(h_polish)
                    if score > best_score:
                        best_score = score
                        best_h = h_polish
            except Exception:
                pass

        # One more round of u-space optimization starting from best
        if time.time() < deadline - 2:
            u0_cur = np.sqrt(best_h)
            try:
                res = minimize(lambda u: -ratio_from_u(u), u0_cur,
                              method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-14})
                h_opt = res.x**2
                n2 = np.sum(h_opt)
                if n2 > 1e-15:
                    h_opt = h_opt / np.sqrt(n2)
                    score = compute_ratio(h_opt)
                    if score > best_score:
                        best_score = score
                        best_h = h_opt
            except Exception:
                pass

    if best_h is None:
        return [1.0] * N

    return list(best_h)
