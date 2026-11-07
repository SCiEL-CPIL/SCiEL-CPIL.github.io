"""Rewritten program using Nelder-Mead simplex optimization."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex optimizer for the BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        def sort_simplex(simplex, simplex_f):
            idx = np.argsort(simplex_f)
            simplex = [simplex[i] for i in idx]
            simplex_f = [simplex_f[i] for i in idx]
            return simplex, simplex_f

        # Track global best
        global_best_x = None
        global_best_f = np.inf
        calls_used = 0

        def nm_run(x0, step):
            """Run Nelder-Mead from x0 with initial step size until budget exhausted."""
            nonlocal calls_used, global_best_x, global_best_f

            simp = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simp.append(xi)
            for i in range(len(simp)):
                simp[i] = np.clip(simp[i], -5, 5)

            sf = [func(x) for x in simp]
            calls_used += dim + 1

            simp, sf = sort_simplex(simp, sf)

            if sf[0] < global_best_f:
                global_best_f = sf[0]
                global_best_x = simp[0].copy()

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            no_improve = 0

            while calls_used < budget:
                f_spread = sf[-1] - sf[0]
                x_spread = np.max(np.abs(np.array(simp) - np.array(simp[0])))

                if f_spread < 1e-12 and x_spread < 1e-9:
                    break

                centroid = np.mean(simp[:-1], axis=0)

                x_r = centroid + alpha * (centroid - simp[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                calls_used += 1

                if f_r < sf[0]:
                    if calls_used < budget:
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5, 5)
                        f_e = func(x_e)
                        calls_used += 1
                        if f_e < f_r:
                            simp[-1] = x_e
                            sf[-1] = f_e
                        else:
                            simp[-1] = x_r
                            sf[-1] = f_r
                    else:
                        simp[-1] = x_r
                        sf[-1] = f_r
                    no_improve = 0
                elif f_r < sf[-2]:
                    simp[-1] = x_r
                    sf[-1] = f_r
                    no_improve = 0
                else:
                    if calls_used < budget:
                        if f_r < sf[-1]:
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            x_c = centroid - rho * (simp[-1] - centroid)
                        x_c = np.clip(x_c, -5, 5)
                        f_c = func(x_c)
                        calls_used += 1

                        if f_c < min(f_r, sf[-1]):
                            simp[-1] = x_c
                            sf[-1] = f_c
                            no_improve = 0
                        else:
                            for i in range(1, dim + 1):
                                if calls_used >= budget:
                                    break
                                simp[i] = simp[0] + sigma * (simp[i] - simp[0])
                                simp[i] = np.clip(simp[i], -5, 5)
                                sf[i] = func(simp[i])
                                calls_used += 1
                            no_improve += 1
                    else:
                        if f_r < sf[-1]:
                            simp[-1] = x_r
                            sf[-1] = f_r
                        no_improve += 1

                min_idx = np.argmin(sf)
                if sf[min_idx] < global_best_f:
                    global_best_f = sf[min_idx]
                    global_best_x = simp[min_idx].copy()

                simp, sf = sort_simplex(simp, sf)

                if no_improve > 15:
                    break

        # Phase 1: Coarse exploration with diverse starts (~35% budget)
        coarse_budget = int(budget * 0.35)
        starts = [
            np.zeros(dim),
            np.array([1.0] * dim),
            np.array([-1.0] * dim),
            np.array([2.0, -2.0, 1.0, -1.0, 2.0]),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        ]

        for x0 in starts:
            if calls_used >= coarse_budget:
                break
            x0 = np.clip(x0, -5, 5)
            f0 = func(x0)
            calls_used += 1
            if f0 < global_best_f:
                global_best_f = f0
                global_best_x = x0.copy()

            for d in range(dim):
                if calls_used >= coarse_budget:
                    break
                for delta in [0.5, -0.5, 1.5, -1.5]:
                    if calls_used >= coarse_budget:
                        break
                    xp = x0.copy()
                    xp[d] += delta
                    xp = np.clip(xp, -5, 5)
                    fp = func(xp)
                    calls_used += 1
                    if fp < global_best_f:
                        global_best_f = fp
                        global_best_x = xp.copy()

        # Phase 2: Focused NM from best point
        if global_best_x is None:
            global_best_x = np.zeros(dim)
            global_best_f = func(global_best_x)
            calls_used += 1

        nm_run(global_best_x, 1.0)

        # Phase 3: If budget remains, try a perturbed restart
        if calls_used < budget - 10:
            rng = np.random.default_rng(42)
            x_pert = global_best_x + 0.3 * rng.standard_normal(dim)
            x_pert = np.clip(x_pert, -5, 5)
            nm_run(x_pert, 0.5)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
