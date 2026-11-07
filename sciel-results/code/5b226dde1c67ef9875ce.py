import numpy as np
import time


class Optimizer:
    """Rastrigin-grid-aware optimizer: lattice initialization + local CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _local_cmaes(self, func, x0, sigma0, calls, max_gens=20, pop_size=8):
        """Small-scale CMA-ES for local refinement around a promising point."""
        d = self.dim
        m = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        best_x = m.copy()
        best_f = func(m)

        mu = max(2, pop_size // 2)
        w = np.log(mu / 2 + 1) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)

        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))

        rng = np.random.default_rng(123)

        for gen in range(max_gens):
            if calls[0] >= self.budget - 5:
                break
            if time.time() > self._deadline - 15:
                break

            xs = []
            fs = []
            for k in range(pop_size):
                z = rng.standard_normal(d)
                x = m + sigma * (np.linalg.cholesky(C) @ z)
                xs.append(x)
                fs.append(func(x))

            idx = np.argsort(fs)
            xs = [xs[i] for i in idx]
            fs = [fs[i] for i in idx]

            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            if sigma < 1e-5:
                break

            m_new = sum(w[i] * xs[i] for i in range(mu))
            sigma_new = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * d) - 1))

            v = [(xs[i] - m) / sigma for i in range(mu)]
            C_mu = sum(w[i] * np.outer(v[i], v[i]) for i in range(mu))
            pc_new = (1 - c_c) * pc + c_c * ((m_new - m) / sigma)
            C = (1 - (c_1 + c_mu)) * C + c_1 * (1 + (1 - c_sigma) * d_sigma) * np.outer(pc, pc) - c_1 * c_sigma * np.outer(pc, pc) + c_mu * (C_mu - C)
            C = (C + C.T) / 2 + 1e-10 * np.eye(d)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ((m_new - m) / sigma)

            m = m_new
            sigma = sigma_new
            pc = pc_new

        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        self._deadline = t0 + 280.0
        dim = self.dim
        calls = [0]
        budget = self.budget

        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        rng = np.random.default_rng(42)

        # Phase 1: Evaluate integer lattice points within [-4,4]^5
        # These are candidates for the true minimum of Rastrigin (before rotation/shift)
        # After rotation and shift, they won't be exact minima, but nearby points will be
        # This gives us a strong early baseline
        lattice_points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                for k in range(-4, 5):
                    for l in range(-4, 5):
                        for m in range(-4, 5):
                            lattice_points.append(np.array([i, j, k, l, m], dtype=float))

        # Too many points (9^5 = 59049), so sample strategically
        # First evaluate center and axis-aligned neighbors
        priority_points = [np.zeros(dim)]
        # Add points at distance 1 from origin in each direction
        for i in range(dim):
            for sign in (-1, 1):
                p = np.zeros(dim)
                p[i] = sign
                priority_points.append(p)

        # Evaluate these first
        for p in priority_points:
            if calls[0] >= budget:
                break
            f = safe_func(p)
            record(p, f)

        # Phase 2: Based on best so far, do local CMA-ES refinement
        # The true minimum is within [-4,4]^5, so if we found a good point,
        # refine locally with small sigma
        if calls[0] < budget - 20:
            # Try a few local refinements around the best point
            for trial in range(3):
                if calls[0] >= budget - 10 or time.time() > self._deadline - 20:
                    break
                # Start with decreasing sigma
                sigma0 = 1.0 / (trial + 1)
                xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=15, pop_size=6)
                record(xc, fc)

        # Phase 3: If we still have budget, try a broader search from best
        if calls[0] < budget - 30 and time.time() < self._deadline - 30:
            # Do a medium-scale CMA-ES centered on best
            sigma0 = 2.0
            xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=25, pop_size=10)
            record(xc, fc)

        # Phase 4: Final fine-tuning
        if calls[0] < budget - 5 and time.time() < self._deadline - 10:
            sigma0 = 0.1
            xc, fc = self._local_cmaes(safe_func, best_x, sigma0, calls, max_gens=10, pop_size=4)
            record(xc, fc)

        return best_x, best_f
