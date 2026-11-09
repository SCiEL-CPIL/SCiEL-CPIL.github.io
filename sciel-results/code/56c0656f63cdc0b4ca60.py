"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 595.0  # safety margin before the 630 s wall-clock kill

    def remaining():
        return deadline - (time.time() - start_time)

    # ------------------------------------------------------------------
    # Objective: weighted L2 energy of Q(x) = prod_i (x - r_i)^2
    # under the Laguerre weight w(x) = x * exp(-x)  (alpha = 1).
    # The Cohn-Goncalves Fourier sign uncertainty upper bound C4 is
    # controlled by this energy (lower energy -> lower bound), so we
    # minimise it.  Computed with Gauss-Laguerre quadrature (weight
    # exp(-x)); we multiply in the extra factor x for alpha = 1.
    # ------------------------------------------------------------------
    # Precompute quadrature nodes/weights once (weight exp(-x)).
    try:
        from numpy.polynomial.laguerre import laggauss
        xq, wq = laggauss(200)
        xq = np.asarray(xq, dtype=float)
        wq = np.asarray(wq, dtype=float)
        xw = xq * wq  # factor x * exp(-x) baked in
    except Exception:
        xq = np.linspace(0.0, 200.0, 4000)
        wq = np.exp(-xq) * (200.0 / (4000 - 1))
        xw = xq * wq

    def energy(roots):
        r = np.asarray(roots, dtype=float)
        if np.any(r <= 0.0):
            return 1e30
        # Q(x) = prod (x - r_i)^2 ; compute log to avoid overflow
        # log Q = 2 * sum_i log|x - r_i|
        logQ = 0.0
        for ri in r:
            logQ += 2.0 * np.log(np.abs(xq - ri))
        # Q^2 -> 2*logQ ; energy = sum wq * x * exp(2*logQ)
        log_integrand = np.log(xq) + 2.0 * logQ + np.log(wq)
        # clip to avoid inf
        log_integrand = np.clip(log_integrand, -700.0, 700.0)
        val = np.sum(np.exp(log_integrand))
        if not np.isfinite(val):
            return 1e30
        return float(val)

    # ------------------------------------------------------------------
    # Candidate starting points
    # ------------------------------------------------------------------
    candidates = []
    # 1. Seed (equally spaced 4,8,...,48)
    candidates.append(np.array([4.0 * (i + 1) for i in range(12)], dtype=float))

    # 2. Roots of L_12^(1) (the natural Laguerre double-root construction)
    try:
        from scipy.special import roots_genlaguerre
        r_lag, _ = roots_genlaguerre(12, 1)
        r_lag = np.sort(np.asarray(r_lag, dtype=float))
        candidates.append(r_lag)
    except Exception:
        pass

    # 3. Roots of L_12^(0)
    try:
        from scipy.special import roots_genlaguerre
        r_lag0, _ = roots_genlaguerre(12, 0)
        r_lag0 = np.sort(np.asarray(r_lag0, dtype=float))
        candidates.append(r_lag0)
    except Exception:
        pass

    # 4. Geometric progression over the seed range
    geo = 4.0 * np.power(48.0 / 4.0, np.arange(12) / 11.0)
    candidates.append(geo)

    # ------------------------------------------------------------------
    # Pick the best starting candidate by energy
    # ------------------------------------------------------------------
    best_val = float("inf")
    best_roots = candidates[0].copy()
    for c in candidates:
        if remaining() < 5.0:
            break
        v = energy(c)
        if v < best_val:
            best_val = v
            best_roots = c.copy()

    # ------------------------------------------------------------------
    # Local refinement with Nelder-Mead from the best start
    # ------------------------------------------------------------------
    try:
        from scipy.optimize import minimize

        def objective(roots_flat):
            r = np.asarray(roots_flat, dtype=float)
            if np.any(r <= 0.0):
                return 1e30
            return energy(r)

        # Refine from the best candidate, and a couple of scaled variants
        start_points = [best_roots.copy()]
        try:
            from scipy.special import roots_genlaguerre
            r_lag, _ = roots_genlaguerre(12, 1)
            r_lag = np.sort(np.asarray(r_lag, dtype=float))
            for s in (0.7, 1.0, 1.3):
                start_points.append(r_lag * s)
        except Exception:
            pass

        for sp in start_points:
            if remaining() < 30.0:
                break
            try:
                res = minimize(
                    objective,
                    sp,
                    method="Nelder-Mead",
                    options={"maxiter": 1500, "xatol": 1e-6, "fatol": 1e-8,
                             "adaptive": True},
                )
                if res.fun < best_val and np.all(res.x > 0.0):
                    best_val = float(res.fun)
                    best_roots = np.sort(res.x).copy()
            except Exception:
                pass
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Final validity: 12 positive, strictly increasing double roots
    # ------------------------------------------------------------------
    out = np.sort(np.asarray(best_roots, dtype=float))
    out = np.maximum(out, 1e-6)
    # ensure strict increase
    for i in range(1, len(out)):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + 1e-6

    return [float(v) for v in out]
# EVOLVE-BLOCK-END
