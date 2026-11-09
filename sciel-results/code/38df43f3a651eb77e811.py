"""CMA-ES with adaptive population size + coordinate polish for Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive pop size and a final coordinate-descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0          # 5s safety margin under 300s wall-clock
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(12345)

        # Initial population: a few random starts, keep the best
        n_init = min(8, budget)
        best_x = np.zeros(dim)
        best_f = np.inf
        for _ in range(n_init):
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        budget -= n_init

        # CMA-ES state
        x_mean = best_x.copy()
        sigma = 2.0
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)          # C^{-1/2}
        # step-size control
        cs = 1 / (dim + 4)
        ds = 1 + 2 * max(0, (dim - 2) / (dim + 2)) + cs
        c_sigma = (dim + 1.7) / (dim + 9 + cs * ds)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 2.1) ** 2 + cs * ds)
        c_mu = min(1 - c_1, (2 + 2 / (dim + 1.8)) / ((dim + 1.8) ** 2 + cs * ds))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        def update_sqrt(C):
            # Re-derive C^{-1/2} via eigendecomposition (dim=5, cheap)
            w, V = np.linalg.eigh(C)
            w = np.maximum(w, 1e-12)
            inv_sqrt = (V / np.sqrt(w)) @ V.T
            return inv_sqrt

        C_inv_sqrt = update_sqrt(C)

        # Adaptive population size: start small, grow as we approach optimum
        # (heuristic: scale with log of current spread)
        def pop_size(f_best, it):
            # start at 4, grow to ~2*dim+4 as iterations progress
            base = 4
            maxp = 2 * dim + 4
            # smooth ramp over first 60% of budget
            frac = min(1.0, it / max(1, 0.6 * budget))
            return int(base + (maxp - base) * frac)

        it = 0
        no_improve = 0
        stagnation_threshold = 30

        while budget > 0:
            if time.time() > deadline:
                break

            n = pop_size(best_f, it)
            n = max(4, min(n, budget))

            # Sample population
            ys = np.empty((n, dim))
            fs = np.empty(n)
            for i in range(n):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5.0, 5.0)
                ys[i] = y
                fs[i] = func(y)
                budget -= 1
                if budget <= 0:
                    break

            if budget <= 0:
                # evaluate the rest of ys if we ran out
                # (we already evaluated the ones we could; break)
                pass

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                no_improve = 0
            else:
                no_improve += 1

            # mu = number of recombination weights
            mu = n // 2
            if mu < 1:
                mu = 1
            # weights: log-linear
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)
            c_mu_eff = min(1 - c_1, c_mu * mu_eff / 2)

            # Recombination
            y_w = np.sum(w[:, None] * ys[:mu], axis=0)
            d = y_w - x_mean

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ d) / sigma
            h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) / chi_n < (1.4 + 2 / (dim + 1))
            h_sig = float(h_sig)

            p_c = (1 - cs) * p_c + h_sig * np.sqrt(cs * (2 - cs) * mu_eff) * d / sigma

            # Rank-one and rank-mu updates
            delta_h = (h_sig - 1) * cs * (1 - cs)
            C = (1 + c_1 * delta_h - c_1 - c_mu_eff) * C \
                + c_1 * np.outer(p_c, p_c) \
                + c_mu_eff * np.sum(w[:, None, None] * np.outer(ys[:mu] - x_mean, ys[:mu] - x_mean), axis=0) / sigma ** 2

            # Ensure PSD
            C = 0.5 * (C + C.T)

            # Update sigma (step-size control)
            sigma *= np.exp(c_sigma / cs * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-8, 10.0))

            # Update mean
            x_mean = y_w

            # Periodically re-derive C^{-1/2} and renormalize C
            if it % 5 == 0:
                # Renormalize C to keep trace ~ dim (prevents drift)
                tr = np.trace(C)
                if tr > 0:
                    C = C * (dim / tr)
                C_inv_sqrt = update_sqrt(C)

            it += 1

            # If stagnating for a long time, do a small random perturbation to escape
            if no_improve > stagnation_threshold and budget > 20:
                x_mean = x_mean + 0.5 * sigma * rng.standard_normal(dim)
                x_mean = np.clip(x_mean, -5.0, 5.0)
                no_improve = 0

        # ---- Final polish: coordinate descent with golden-section line search ----
        # Use remaining budget (if any) and a time check
        x = best_x.copy()
        fx = best_f
        for _ in range(3):  # a few passes of coordinate descent
            if time.time() > deadline - 2.0:
                break
            for d in range(dim):
                if budget <= 2:
                    break
                # Golden-section search along axis d
                a = x[d] - 2.0 * max(sigma, 0.1)
                b = x[d] + 2.0 * max(sigma, 0.1)
                a = max(-5.0, a)
                b = min(5.0, b)
                if b - a < 1e-6:
                    continue

                gr = (np.sqrt(5.0) - 1.0) / 2.0
                c = b - gr * (b - a)
                dval = a + gr * (b - a)

                xc = x.copy(); xc[d] = c
                fd = x.copy(); fd[d] = dval
                fc = func(xc)
                fdd = func(fd)
                budget -= 2

                for _ in range(8):
                    if budget <= 2:
                        break
                    if fc < fdd:
                        b = dval
                        dval = c
                        fdd = fc
                        c = b - gr * (b - a)
                        xc[d] = c
                        fc = func(xc)
                    else:
                        a = c
                        c = dval
                        fc = fdd
                        dval = a + gr * (b - a)
                        fd[d] = dval
                        fdd = func(fd)
                    budget -= 2
                    if time.time() > deadline - 1.0:
                        break

                # Best of the line
                mid = (a + b) / 2.0
                xm = x.copy(); xm[d] = mid
                fm = func(xm)
                budget -= 1
                if fm < fx:
                    x = xm
                    fx = fm
                elif fc < fx:
                    x = xc
                    fx = fc
                elif fdd < fx:
                    x = fd
                    fx = fdd

                if time.time() > deadline - 1.0:
                    break

        # Final best
        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
