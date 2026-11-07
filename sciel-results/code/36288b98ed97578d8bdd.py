"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with smaller population and local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin below the 330 s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        calls = 0
        best_x = None
        best_f = np.inf

        def ev(x):
            nonlocal calls, best_f, best_x
            fx = func(clip(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initialization
        # Use 10 random points to find a good starting point
        init_n = 10
        init_xs = rng.uniform(-4.0, 4.0, (init_n, dim))
        init_fs = np.array([ev(x) for x in init_xs])
        if calls >= budget:
            return best_x, best_f
        
        idx_best = np.argmin(init_fs)
        x_mean = init_xs[idx_best].copy()
        sigma = 1.0  # Initial sigma

        # Phase 2: CMA-ES
        lam = 10
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0.0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)

        stagnation_counter = 0
        stagnation_limit = 10  # Generations without improvement

        while calls + lam <= budget and (time.time() - start) < deadline:
            zs = rng.standard_normal((lam, dim))
            ys = np.array([x_mean + sigma * (B @ (D * z)) for z in zs])
            fs = np.array([ev(y) for y in ys])
            
            if (time.time() - start) > deadline:
                break

            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Check for stagnation
            if fs[0] >= best_f + 1e-8:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            if stagnation_counter >= stagnation_limit:
                break

            x_old = x_mean.copy()
            x_mean = np.sum(w[:, None] * ys[:mu], axis=0)
            y = (x_mean - x_old) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D * y))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 1 / dim) * np.sqrt(dim):
                pc = (1 - c_c) * pc + c_c * y
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (np.outer(pc, pc) + (c_c - c_c ** 2) * np.eye(dim))
                     + c_mu * np.sum(w[:mu, None, None] *
                                    np.outer((ys[:mu] - x_old) / sigma, (ys[:mu] - x_old) / sigma),
                                    axis=0))
                ev_vals, ev_vecs = np.linalg.eigh(C)
                ev_vals = np.abs(ev_vals)
                idx = ev_vals.argsort()[::-1]
                ev_vals = ev_vals[idx]
                ev_vecs = ev_vecs[:, idx]
                D = np.sqrt(ev_vals)
                B = ev_vecs

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            sigma = min(max(sigma, 1e-10), 50.0)

        # Phase 3: Local Polish (1+1)-ES
        x = best_x.copy()
        fx = best_f
        s = 0.1
        while calls < budget and (time.time() - start) < deadline:
            y = clip(x + s * rng.standard_normal(dim))
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                s *= 1.1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                s *= 0.9
            if s < 1e-6:
                break

        return best_x, best_f
