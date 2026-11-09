"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with a cheap deterministic warm-start.

    CMA-ES (population, covariance-adapted) is the state-of-the-art for
    non-separable continuous problems and adapts to the hidden rotation.
    A small deterministic/LHS warm-start gives early AOCC credit and a
    good seed for each restart.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    # ---- deterministic low-discrepancy (Halton) sampler -------------------
    @staticmethod
    def _halton(dim, n, seed=1):
        pts = np.empty((n, dim))
        for d in range(dim):
            base = d + 2  # 2,3,5,...
            seq = np.empty(n)
            f, r = 1.0, 0.0
            for i in range(1, n + 1):
                i2 = i + seed
                while i2 > 0:
                    r += f * (i2 % base)
                    i2 //= base
                    f /= base
                seq[i - 1] = r
            pts[:, d] = seq
        return pts  # in [0,1]^dim

    # ---- one CMA-ES run (budgeted) ---------------------------------------
    def _cmaes(self, func, x0, budget_left, deadline, rng, best_state):
        dim = self.dim
        lam = max(4, 4 + int(3 * np.log(dim)))  # ~13 for dim=5
        # step size: start moderately; the box is [-5,5]
        sigma = 1.0
        mean = np.array(x0, dtype=float)
        # covariance
        C = np.eye(dim)
        C_inv = np.eye(dim)
        # precompute Cholesky of C (identity)
        B = np.eye(dim)
        D = np.ones(dim)
        # adaptation rates
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1) / ((dim + 1) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        d_corr = 1 + c_sigma * (2 / (dim + 1) - 1)

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # evaluate mean
        fx_mean = func(clip(mean))
        if fx_mean < best_state[1]:
            best_state[0], best_state[1] = clip(mean).copy(), fx_mean

        gen = 0
        while budget_left > 0:
            if time.time() > deadline:
                break
            # sample population
            z = rng.standard_normal((lam, dim))
            pop = mean[None, :] + sigma * (z @ B.T) / D[None, :]
            pop = np.clip(pop, -5.0, 5.0)
            # evaluate
            fvals = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    break
                fvals[i] = func(pop[i])
                budget_left -= 1
            if budget_left <= 0:
                break
            # track best
            i_min = int(np.argmin(fvals))
            if fvals[i_min] < best_state[1]:
                best_state[0], best_state[1] = pop[i_min].copy(), fvals[i_min]
            # sort
            order = np.argsort(fvals)
            sorted_pop = pop[order]
            sorted_f = fvals[order]
            # weighted mean
            y_w = (w[:, None] * (sorted_pop[:mu] - mean[None, :])).sum(axis=0)
            # sigma adaptation
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y_w))
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            # update mean
            mean = mean + sigma * y_w
            mean = np.clip(mean, -5.0, 5.0)
            # covariance adaptation
            y = (sorted_pop[:mu] - mean[None, :]) / sigma
            delta_s = np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1)) - 1
            p_c = (1 - c_c) * p_c + c_c * delta_s * y_w
            rank_one = np.outer(p_c, p_c)
            rank_mu = (w[:, None] * y) @ y.T
            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * (rank_one - (1 - c_c) * (1 - 2 * c_c) * np.eye(dim)) + c_mu * rank_mu
            # ensure symmetry
            C = 0.5 * (C + C.T)
            # recompute Cholesky of C_inv via C
            try:
                # eigendecompose for stable inverse/sqrt
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                D = np.sqrt(evals)
                B = evecs.T
                C_inv = (evecs / evals) @ evecs.T
            except np.linalg.LinAlgError:
                # reset
                C = np.eye(dim)
                C_inv = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
            gen += 1
        return budget_left

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0  # 30s margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        best_state = [np.zeros(dim), np.inf]
        remaining = budget

        # ---- Phase 1: cheap deterministic + LHS warm-start ----------------
        # origin + a coarse Halton set + a few random points
        warm_n = min(24, budget)
        candidates = [np.zeros(dim)]
        hal = self._halton(dim, max(0, warm_n - 1))
        candidates.extend(hal * 8.0 - 4.0)  # map to [-4,4]
        for c in candidates:
            if remaining <= 0 or time.time() > deadline:
                break
            c = np.clip(c, -5.0, 5.0)
            fc = func(c)
            remaining -= 1
            if fc < best_state[1]:
                best_state[0], best_state[1] = c.copy(), fc

        # ---- Phase 2: multi-restart CMA-ES --------------------------------
        # Restart from the current best, then from a couple of diverse points.
        seeds = [best_state[0].copy()]
        seeds.append(rng.uniform(-4, 4, dim))
        seeds.append(rng.uniform(-4, 4, dim))

        n_restarts = len(seeds)
        # give each restart a share; the first (from best) gets the most
        shares = [0.5, 0.25, 0.25]
        for i, s in enumerate(seeds):
            if remaining <= 0 or time.time() > deadline:
                break
            alloc = int(remaining * shares[i])
            alloc = max(1, alloc)
            remaining_before = remaining
            remaining = self._cmaes(func, s, alloc, deadline, rng, best_state)
            remaining = min(remaining, remaining_before)
            if remaining <= 0:
                break

        # safety: return best found
        return best_state[0], best_state[1]
# EVOLVE-BLOCK-END
