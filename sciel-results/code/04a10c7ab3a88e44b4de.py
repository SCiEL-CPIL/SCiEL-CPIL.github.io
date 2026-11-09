"""Fresh design: Nelder-Mead with restarts for anisotropic Bent Cigar (f12)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with diverse restarts, tuned for anisotropic f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill
        rng = np.random.default_rng(12345)

        def safe(x):
            return func(np.clip(np.asarray(x, dtype=float), -5.0, 5.0))

        # Phase 1: diverse exploration to find a promising basin
        # Spend ~30% of budget on random restarts
        n_explore = max(10, int(budget * 0.25))
        best_x = None
        best_f = np.inf
        candidates = []

        for i in range(n_explore):
            if time.time() > deadline - 5.0:
                break
            if i == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = safe(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        # Sort by fitness, keep top few for Nelder-Mead refinement
        candidates.sort(key=lambda t: t[0])
        top_k = min(5, len(candidates))

        # Phase 2: Nelder-Mead refinement on top candidates
        remaining = budget - n_explore
        per_refine = max(10, remaining // top_k)

        for rank in range(top_k):
            if time.time() > deadline - 5.0:
                break
            if remaining <= 0:
                break
            f0, x0 = candidates[rank]
            # Adaptive initial simplex size: start moderate, adapt to landscape
            alpha = 0.5 if rank == 0 else 0.3
            x_best, f_best = self._nelder_mead(
                safe, x0, f0, dim, per_refine, alpha,
                deadline=deadline - 5.0, rng=rng
            )
            remaining -= per_refine
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        # Phase 3: if budget remains, do a final fine-tune Nelder-Mead
        if remaining > 20 and best_x is not None and time.time() < deadline - 5.0:
            x_best, f_best = self._nelder_mead(
                safe, best_x, best_f, dim, remaining, 0.1,
                deadline=deadline - 2.0, rng=rng
            )
            if f_best < best_f:
                best_f = f_best
                best_x = x_best.copy()

        return best_x, best_f

    def _nelder_mead(self, safe, x0, f0, dim, max_iter, alpha, deadline, rng):
        """Nelder-Mead with time guard and anisotropy-friendly scaling."""
        # Initialize simplex: x0 plus dim perturbations
        simplex = [x0.copy()]
        fvals = [f0]
        for i in range(dim):
            dx = x0.copy()
            dx[i] += alpha * (1.0 if dx[i] < 4.0 else -1.0)
            dx = np.clip(dx, -5.0, 5.0)
            simplex.append(dx)
            fvals.append(safe(dx))

        best_idx = int(np.argmin(fvals))
        best_x = simplex[best_idx].copy()
        best_f = fvals[best_idx]

        it = 0
        n_evals = dim + 1
        # Standard Nelder-Mead parameters
        gamma = 2.0   # reflection
        rho = -1.0    # expansion
        sigma = 0.5   # contraction
        xi = 0.5      # shrink

        while it < max_iter and n_evals < max_iter and time.time() < deadline:
            # Sort simplex by fitness
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            f_worst = fvals[-1]

            # Reflection
            x_r = centroid + gamma * (centroid - worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = safe(x_r)
            n_evals += 1

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + rho * (x_r - centroid)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = safe(x_e)
                n_evals += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < f_worst:
                    x_c = centroid + sigma * (x_r - centroid)
                else:
                    x_c = centroid - sigma * (worst - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = safe(x_c)
                n_evals += 1
                if f_c <= min(f_r, f_worst):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + xi * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = safe(simplex[i])
                        n_evals += 1

            # Track best
            cur_best = int(np.argmin(fvals))
            if fvals[cur_best] < best_f:
                best_f = fvals[cur_best]
                best_x = simplex[cur_best].copy()

            it += 1
            # Terminate if simplex is very small (converged)
            spread = np.max(fvals) - np.min(fvals)
            if spread < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
