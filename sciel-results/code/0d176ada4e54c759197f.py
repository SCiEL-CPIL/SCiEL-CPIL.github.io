"""Combined CMA-ES + Nelder-Mead refinement for BBOB f12 (Bent Cigar)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES global search with a budget-safe Nelder-Mead local refinement
    for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        # Reserve budget for the local refinement phase
        reserve = max(20, min(80, budget // 5))
        cma_budget = max(10, budget - reserve)

        best_x = np.random.uniform(-5, 5, d)
        best_f = func(best_x)
        calls = 1

        # ---- CMA-ES parameters (dim=5 tuned) ----
        lam = 4 + int(3 * np.log(d))
        lam = max(10, lam)
        mu = lam // 2

        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (20 * d * d))

        # State: start near the box centre, moderately wide
        m = np.zeros(d)
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        sigma = 2.0

        p_sigma = np.zeros(d)
        p_c = np.zeros(d)

        def eval_point(x):
            nonlocal calls, best_x, best_f
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- CMA-ES main loop ----
        while calls < cma_budget:
            if time.time() >= deadline:
                break
            if cma_budget - calls < lam:
                break

            Y = np.zeros((lam, d))
            X = np.zeros((lam, d))
            F = np.zeros(lam)

            m_old = m.copy()
            gen_ok = True
            for i in range(lam):
                if calls >= cma_budget:
                    gen_ok = False
                    break
                if time.time() >= deadline:
                    gen_ok = False
                    break
                z = np.random.randn(d)
                x_i = m + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                Y[i] = z
                X[i] = x_i
                F[i] = eval_point(x_i)

            if not gen_ok:
                break

            # Sort
            order = np.argsort(F)
            Y = Y[order]
            X = X[order]
            F = F[order]

            # Update Mean
            w_mu = w[:mu]
            m_new = np.sum(w_mu[:, None] * X[:mu], axis=0)

            # Update Paths
            c_sqrt_sigma = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)
            step = (m_new - m_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sqrt_sigma * w_mu[0] * (B.T @ (D * step))

            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1) / d))
            h_sigma = 1.0
            if denom > 0:
                if norm_p_sigma / denom >= (1.4 + 2 / (d + 1)) * chi_n:
                    h_sigma = 0.0

            c_sqrt_c = np.sqrt(c_c * (2 - c_c) * mu_eff)
            p_c = (1 - c_c) * p_c + h_sigma * c_sqrt_c * w_mu[0] * ((m_new - m_old) / sigma)

            # Update C
            V = (X[:mu] - m_old) / sigma
            factor = 1 - c_1 - c_mu + h_sigma * c_1 * (2 - c_1)
            C = factor * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c))
            C += c_mu * (V.T * w_mu) @ V

            C = (C + C.T) / 2
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-12)
            D = np.sqrt(eigval)
            B = eigvec

            # Update Sigma
            sigma = sigma * np.exp(1.0 / d_sigma * (norm_p_sigma / chi_n - 1))
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 100:
                sigma = 100

            m = m_new

        # ---- Nelder-Mead local refinement (budget-safe) ----
        if calls < budget - 5 and time.time() < deadline:
            self._nelder_mead(func, best_x, budget - calls, d, eval_point,
                              lambda: time.time() >= deadline)
            # best_x / best_f updated via eval_point closure above

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget, d, eval_point, timeout_check):
        """Nelder-Mead simplex refinement. Uses eval_point so every call is
        counted against the shared budget."""
        # Build initial simplex
        simplex = [x0.copy()]
        fvals = [eval_point(x0)]

        for i in range(d):
            xi = x0.copy()
            if abs(x0[i]) < 4.5:
                xi[i] += 0.5
            else:
                xi[i] -= 0.5
            xi = np.clip(xi, -5, 5)
            simplex.append(xi)
            fvals.append(eval_point(xi))

        calls = d + 1
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        for _ in range(budget - calls):
            if calls >= budget:
                break
            if timeout_check():
                break

            # Sort simplex
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflect
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = eval_point(x_ref)
            calls += 1

            if f_ref < fvals[0]:
                # Try expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = eval_point(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    fvals[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
            elif f_ref < fvals[-2]:
                simplex[-1] = x_ref
                fvals[-1] = f_ref
            else:
                # Contraction
                x_con = centroid + rho * (simplex[-1] - centroid)
                x_con = np.clip(x_con, -5, 5)
                f_con = eval_point(x_con)
                calls += 1
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = eval_point(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
# EVOLVE-BLOCK-END
