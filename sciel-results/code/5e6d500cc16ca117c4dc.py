"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead optimizer for BBOB f17 (Schaffer's F7)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 330s limit, leave 50s margin

        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            fx = float(func(x))
            calls_made += 1
            return fx

        def time_ok():
            return time.time() < deadline and calls_made < budget

        # Phase 1: Initial sampling for early progress
        points = [np.zeros(d)]
        for corner in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
            pts = []
            for i in range(d):
                if i % 2 == 0:
                    pts.append(corner[0] * 3.0)
                else:
                    pts.append(corner[1] * 3.0)
            points.append(np.array(pts))

        for i in range(d):
            x = np.zeros(d)
            x[i] = 2.5
            points.append(x.copy())
            x[i] = -2.5
            points.append(x.copy())

        n_random = max(1, min(15, budget - len(points)))
        for _ in range(n_random):
            points.append(rng.uniform(-4, 4, d))

        for x in points:
            if not time_ok():
                break
            x = np.clip(x, -5, 5)
            fx = call_func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = call_func(best_x)

        # Phase 2: CMA-ES with adaptive population size
        def cma_es(x0, sigma0, max_calls, rng_local):
            """Run CMA-ES starting from x0 with initial sigma."""
            nonlocal best_x, best_fx, calls_made

            n = d
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w**2)

            cs = (mueff + 2) / (n + mueff + 5)
            ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))

            c1 = (2 + 2 * mueff / n) / ((n + 2)**2 + 2 * mueff / n)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + 2 * mueff / n))

            C = np.eye(n)
            C_eigval = np.ones(n)
            C_eigvec = np.eye(n)
            C_eigdec = True

            xmean = x0.copy()
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            local_best_x = x0.copy()
            local_best_fx = np.inf

            def C_eig_decomp():
                nonlocal C, C_eigval, C_eigvec, C_eigdec
                if not C_eigdec:
                    C_eigval, C_eigvec = np.linalg.eigh(C)
                    C_eigval = np.maximum(C_eigval, 1e-20)
                    C_eigdec = True
                return C_eigval, C_eigvec

            def C_sqrt():
                nonlocal C, C_eigval, C_eigvec
                C_eig_decomp()
                return C_eigvec @ np.diag(np.sqrt(C_eigval))

            CS = C_sqrt()

            while calls_made < max_calls and time.time() < deadline:
                if not time_ok():
                    break

                ys = []
                for _ in range(lam):
                    z = rng_local.standard_normal(n)
                    y = xmean + sigma * (CS @ z)
                    ys.append(y)

                fvals = []
                for y in ys:
                    if not time_ok():
                        break
                    y_clipped = np.clip(y, -5, 5)
                    f = call_func(y_clipped)
                    fvals.append(f)

                if not time_ok():
                    break

                indices = np.argsort(fvals)
                ys_sorted = [ys[i] for i in indices]
                fvals_sorted = [fvals[i] for i in indices]

                if fvals_sorted[0] < local_best_fx:
                    local_best_fx = fvals_sorted[0]
                    local_best_x = ys_sorted[0].copy()
                    if local_best_fx < best_fx:
                        best_fx = local_best_fx
                        best_x = local_best_x.copy()

                y_w = np.array(ys_sorted[:mu])
                xmean_new = np.sum(w[:, None] * y_w, axis=0)

                C_eig_decomp()
                CS_old = C_eigvec @ np.diag(np.sqrt(C_eigval))
                y_sigma = (xmean_new - xmean) / (sigma * chi_n)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_sigma

                hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**2)) < (1.4 + 2 / (n + 1)) * chi_n
                hs = float(hsig)

                p_c = (1 - cs) * p_c + hs * np.sqrt(cs * (2 - cs) * mueff) * (xmean_new - xmean) / sigma

                C_eig_decomp()
                C_old = C_eigvec @ np.diag(C_eigval) @ C_eigvec.T

                delta_p = p_c / sigma
                C_rank1 = (1 + c1 * (2 - c1) - mueff * c1) * C_old + c1 * (2 - c1) * np.outer(delta_p, delta_p)

                C_rankmu = C_old.copy()
                for i in range(mu):
                    y_i = y_w[i] - xmean
                    C_rankmu += cmu * w[i] * np.outer(y_i, y_i)

                C = C_rank1 + (C_rankmu - C_old) * (cmu / c1) if c1 > 0 else C_rankmu
                C = 0.5 * (C + C.T)

                C_eigval, C_eigvec = np.linalg.eigh(C)
                C_eigval = np.maximum(C_eigval, 1e-20)
                C_eigdec = True

                sigma = sigma * np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                xmean = xmean_new

                if sigma < 1e-12 or np.all(np.abs(xmean) > 4.9):
                    break

            return local_best_x, local_best_fx

        # Phase 3: Nelder-Mead local refinement
        def nelder_mead(x0, max_calls, rng_local):
            """Run Nelder-Mead local refinement starting from x0."""
            nonlocal best_x, best_fx, calls_made

            x0 = np.clip(x0, -5, 5)
            f0 = call_func(x0)
            if f0 < best_fx:
                best_fx = f0
                best_x = x0.copy()

            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(d):
                x_i = x0.copy()
                step = 0.25 if abs(x0[i]) < 4.5 else 0.1
                x_i[i] += step
                x_i = np.clip(x_i, -5, 5)
                simplex.append(x_i)

            fvals = [call_func(v) for v in simplex]

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            max_iters = max(1, (max_calls - calls_made) // (d + 1))

            for _ in range(max_iters):
                if not time_ok():
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                worst = simplex[-1]
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = call_func(xr)
                if not time_ok():
                    break

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = call_func(xe)
                    if not time_ok():
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = call_func(xc)
                    if not time_ok():
                        break
                    if fc <= fr:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = call_func(simplex[i])
                            if not time_ok():
                                break

                if not time_ok():
                    break

            return best_x, best_fx

        # Multi-restart CMA-ES
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            first_run_calls = min(remaining, 300)
            x0 = best_x.copy()
            sigma0 = 1.0
            _, _ = cma_es(x0, sigma0, calls_made + first_run_calls, rng)

        # Nelder-Mead refinement from CMA-ES result
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 20:
                nm_calls = min(remaining, 120)
                _, _ = nelder_mead(best_x.copy(), calls_made + nm_calls, rng)

        # Second CMA-ES run from a different starting point if budget allows
        if calls_made < budget and time.time() < deadline:
            remaining = budget - calls_made
            if remaining > 50:
                x0_2 = rng.uniform(-3, 3, d)
                fx2 = call_func(np.clip(x0_2, -5, 5))
                if fx2 < best_fx:
                    best_fx = fx2
                    best_x = x0_2.copy()

                if calls_made < budget and time.time() < deadline:
                    second_run_calls = min(budget - calls_made, 80)
                    sigma0_2 = 0.5
                    _, _ = cma_es(best_x.copy(), sigma0_2, calls_made + second_run_calls, rng)

                # Another Nelder-Mead pass
                if calls_made < budget and time.time() < deadline:
                    remaining2 = budget - calls_made
                    if remaining2 > 20:
                        nm_calls2 = min(remaining2, 50)
                        _, _ = nelder_mead(best_x.copy(), calls_made + nm_calls2, rng)

        # Phase 4: Final polish with very small steps
        if calls_made < budget and time.time() < deadline:
            polish_calls = min(30, budget - calls_made)
            for _ in range(polish_calls):
                if not time_ok():
                    break
                y = best_x + 0.01 * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = call_func(y)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
