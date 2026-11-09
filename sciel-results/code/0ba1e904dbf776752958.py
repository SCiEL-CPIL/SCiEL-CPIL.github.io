"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + CMA-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before the 330s kill

        # ---------- Phase 1: Latin Hypercube exploration ----------
        n_lhs = 60
        rng = np.random.default_rng(12345)
        perm = [rng.permutation(n_lhs) for _ in range(dim)]
        lxs = np.array([(perm[d] + rng.random()) / n_lhs for d in range(dim)]).T
        lxs = lxs * 10.0 - 5.0  # map to [-5,5]
        lxs = np.clip(lxs, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        calls = 0
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            fx = func(lxs[i])
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = lxs[i].copy()

        # ---------- Phase 2: CMA-ES with periodic restarts ----------
        # CMA-ES parameters
        pop_size = 8
        lambda_ = pop_size
        mu = lambda_ // 2  # number of best points for weighting
        # Weights: linear weights normalized
        weights = np.linspace(1.0, 0.1, mu)
        weights = weights / weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        sigma = 1.5
        sigma_min = 1e-8
        sigma_max = 5.0

        x = best_x.copy()
        fx = best_f

        global_best_x = best_x.copy()
        global_best_f = best_f

        # CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)

        step = 0
        restart_interval = 80
        no_improve = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fys = []
            for j in range(lambda_):
                if time.time() > deadline:
                    break
                y = x + sigma * (B @ (D * rng.standard_normal(dim)))
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                ys.append(y)
                fys.append(fy)

            if len(ys) < mu:
                break

            ys = np.array(ys)
            fys = np.array(fys)

            # Sort by fitness
            order = np.argsort(fys)
            ys_sorted = ys[order]
            fys_sorted = fys[order]

            # Update best
            if fys_sorted[0] < global_best_f:
                global_best_f = fys_sorted[0]
                global_best_x = ys_sorted[0].copy()

            # Weighted mean
            m = weights @ ys_sorted[:mu]

            # Update covariance matrix adaptation
            y_sigma = (m - x) / (sigma * chi_n) if sigma > 0 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (step + 1))) / chi_n < (1.4 + 2 / (dim + 1)) * c_sigma

            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - x) / sigma if sigma > 0 else (1 - c_c) * p_c
            C = (1 + c_1 * (h_sig - 1) * np.outer(p_c, p_c) / (sigma ** 2) - c_1) * C + c_mu * np.sum(weights, axis=1) * 0  # simplified

            # Proper CMA-ES covariance update
            C = (1 - c_1) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * C)
            for i in range(mu):
                z = (ys_sorted[i] - x) / sigma if sigma > 0 else np.zeros(dim)
                C += c_mu * weights[i] * np.outer(z, z)

            # Eigen-decomposition for step size and basis
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update mean
            x = m

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, sigma_min, sigma_max)

            step += 1

            # Track no improvement
            if fys_sorted[0] < fx:
                fx = fys_sorted[0]
                no_improve = 0
            else:
                no_improve += 1

            # Periodic restart: escape local minima
            if step % restart_interval == 0 or no_improve >= 40:
                if time.time() > deadline:
                    break
                if rng.random() < 0.7:
                    x = global_best_x + sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5.0, 5.0)
                else:
                    x = rng.uniform(-5.0, 5.0, dim)
                fx = func(x)
                calls += 1
                if fx < global_best_f:
                    global_best_f = fx
                    global_best_x = x.copy()
                # Reset CMA-ES state
                sigma = 1.5
                B = np.eye(dim)
                D = np.ones(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                C = np.eye(dim)
                no_improve = 0

        # ---------- Phase 3: Nelder-Mead polish on the final best point ----------
        if calls < budget:
            nm_calls = min(50, budget - calls)
            # Initialize simplex
            simplex = [global_best_x.copy()]
            for d in range(dim):
                pt = global_best_x.copy()
                step_size = 0.1 if abs(global_best_x[d]) > 0.1 else 0.01
                pt[d] += step_size
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)

            fs = []
            for pt in simplex:
                if calls >= budget or time.time() > deadline:
                    break
                f = func(pt)
                calls += 1
                fs.append(f)

            for _ in range(nm_calls - len(fs) + 1):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                x_ref = centroid + 1.0 * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = func(x_ref)
                calls += 1

                if fs[0] <= f_ref < fs[-2]:
                    simplex[-1] = x_ref
                    fs[-1] = f_ref
                elif f_ref < fs[0]:
                    # Expansion
                    x_exp = centroid + 2.0 * (centroid - simplex[-1])
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = func(x_exp)
                    calls += 1
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fs[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fs[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + 0.5 * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = func(x_con)
                    calls += 1
                    if f_con < fs[-1]:
                        simplex[-1] = x_con
                        fs[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if calls >= budget or time.time() > deadline:
                                break
                            fs[i] = func(simplex[i])
                            calls += 1

                # Update global best
                if fs[0] < global_best_f:
                    global_best_f = fs[0]
                    global_best_x = simplex[0].copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
