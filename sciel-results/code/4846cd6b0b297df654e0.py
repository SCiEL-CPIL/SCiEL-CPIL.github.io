"""Refined Cohn-Goncalves Laguerre construction - log-parameterized gradient optimization."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre

def construct():
    start_time = time.time()
    deadline = 580.0  # safety margin before 630s wall-clock kill

    def remaining():
        return deadline - (time.time() - start_time)

    # Quadrature nodes for weight exp(-x); we multiply by x for alpha=1 weight
    xq, wq = roots_genlaguerre(400)
    xq = np.asarray(xq, dtype=float)
    wq = np.asarray(wq, dtype=float)
    # log of x * exp(-x) * wq  (wq already includes exp(-x) factor in Gauss-Laguerre)
    # Gauss-Laguerre: integral f(x) e^{-x} dx ≈ sum wq_i f(xq_i)
    # We want integral f(x) x e^{-x} dx = integral (x f(x)) e^{-x} dx
    # So integrand weight is xq * wq
    log_xw = np.log(np.maximum(xq, 1e-300)) + np.log(np.maximum(wq, 1e-300))

    n_roots = 12

    def energy_from_roots(roots):
        """Compute integral of Q(x)^2 * x * exp(-x) dx where Q = prod (x-ri)^2."""
        r = np.asarray(roots, dtype=float)
        if np.any(r <= 0):
            return 1e30
        # Q(x)^2 = prod (x - ri)^4
        # log(Q^2) = 4 * sum log|x - ri|
        # log integrand = 4*sum log|x-ri| + log(x) + log(wq)
        diffs = np.abs(xq[None, :] - r[:, None])  # (12, 400)
        diffs = np.maximum(diffs, 1e-15)
        log_Q_sq = 4.0 * np.sum(np.log(diffs), axis=0)  # (400,)
        log_integrand = log_Q_sq + log_xw
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        val = np.sum(np.exp(log_integrand))
        if not np.isfinite(val):
            return 1e30
        return float(val)

    def energy_from_log(log_roots):
        """Energy in log-parameterization."""
        roots = np.exp(np.asarray(log_roots, dtype=float))
        return energy_from_roots(roots)

    # ------------------------------------------------------------------
    # Generate diverse starting points in log-space
    # ------------------------------------------------------------------
    candidates_log = []

    # 1. Laguerre roots L_12^(1) - the natural starting point
    r_lag1, _ = roots_genlaguerre(12, 1)
    r_lag1 = np.sort(np.asarray(r_lag1, dtype=float))
    candidates_log.append(np.log(r_lag1))

    # 2. Laguerre roots L_12^(0)
    r_lag0, _ = roots_genlaguerre(12, 0)
    r_lag0 = np.sort(np.asarray(r_lag0, dtype=float))
    candidates_log.append(np.log(r_lag0))

    # 3. Scaled Laguerre roots
    for s in [0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.0, 3.0, 5.0]:
        candidates_log.append(np.log(r_lag1 * s))
        candidates_log.append(np.log(r_lag0 * s))

    # 4. Equally spaced
    eq = np.linspace(2.0, 50.0, 12)
    candidates_log.append(np.log(eq))

    # 5. Geometric progression
    geo = 3.0 * np.power(15.0, np.arange(12) / 11.0)
    candidates_log.append(np.log(geo))

    # 6. Asymptotic Laguerre distribution
    n = 12
    alpha = 1
    k = np.arange(n)
    asym = (4 * n + 2 * alpha + 2) * np.sin((k + 0.25) * np.pi / (2 * n + alpha + 1)) ** 2
    asym = np.sort(asym)
    candidates_log.append(np.log(asym))

    # 7. Random starts (seeded)
    rng = np.random.RandomState(123)
    for trial in range(8):
        # Log-uniform in [0.5, 100]
        lr = np.sort(rng.uniform(np.log(0.5), np.log(100.0), 12))
        candidates_log.append(lr)

    # 8. Perturbed Laguerre roots
    for trial in range(6):
        pert = r_lag1 * np.exp(rng.normal(0, 0.3, 12))
        pert = np.sort(pert)
        candidates_log.append(np.log(pert))

    # ------------------------------------------------------------------
    # Evaluate all candidates, pick best
    # ------------------------------------------------------------------
    best_val = float('inf')
    best_log_roots = candidates_log[0].copy()
    for cl in candidates_log:
        if remaining() < 10.0:
            break
        v = energy_from_log(cl)
        if v < best_val:
            best_val = v
            best_log_roots = cl.copy()

    # ------------------------------------------------------------------
    # Gradient-based optimization from top candidates
    # ------------------------------------------------------------------
    # Sort candidates by energy
    cand_scores = []
    for cl in candidates_log:
        if remaining() < 10.0:
            break
        v = energy_from_log(cl)
        cand_scores.append((v, cl))
    cand_scores.sort(key=lambda x: x[0])

    # Take top 8 distinct starting points
    top_starts = []
    seen = set()
    for v, cl in cand_scores:
        key = tuple(np.round(cl, 2))
        if key not in seen:
            top_starts.append(cl.copy())
            seen.add(key)
        if len(top_starts) >= 8:
            break

    # L-BFGS-B in log-space (unbounded, which is natural for log)
    for sp in top_starts:
        if remaining() < 15.0:
            break
        try:
            res = minimize(
                energy_from_log,
                sp,
                method='L-BFGS-B',
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best_log_roots = res.x.copy()
        except Exception:
            pass

    # Nelder-Mead in log-space (robust, no gradient needed)
    for sp in top_starts[:5]:
        if remaining() < 20.0:
            break
        try:
            res = minimize(
                energy_from_log,
                sp,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best_log_roots = res.x.copy()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Fine-grained local search in log-space (coordinate descent)
    # ------------------------------------------------------------------
    if remaining() > 30.0:
        out = best_log_roots.copy()
        for iteration in range(100):
            if remaining() < 10.0:
                break
            improved = False
            for i in range(n_roots):
                if remaining() < 5.0:
                    break
                base_val = energy_from_log(out)
                for delta in [-0.05, -0.02, -0.005, 0.005, 0.02, 0.05,
                              -0.001, 0.001, -0.01, 0.01]:
                    trial = out.copy()
                    trial[i] += delta
                    v = energy_from_log(trial)
                    if v < base_val - 1e-14:
                        base_val = v
                        out = trial.copy()
                        improved = True
            if not improved:
                break
        best_log_roots = out
        best_val = energy_from_log(best_log_roots)

    # ------------------------------------------------------------------
    # Convert back to root space and validate
    # ------------------------------------------------------------------
    out = np.exp(best_log_roots)
    out = np.sort(out)
    out = np.maximum(out, 1e-6)
    # Ensure strict increase
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6

    return [float(v) for v in out]
