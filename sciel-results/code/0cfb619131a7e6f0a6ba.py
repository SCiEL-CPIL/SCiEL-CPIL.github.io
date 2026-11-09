"""Optimizer for BBOB-style Discus (f11) function."""
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for smooth anisotropic landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Track best
        best_x = None
        best_f = np.inf
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        def eval_point(x):
            nonlocal calls, best_x, best_f
            x = clip(x)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Diverse exploration (~40 calls)
        # Include origin, corners, and random points
        candidates = []
        candidates.append(np.zeros(dim))
        # Box corners (sample a subset)
        for mask in range(16):
            corner = np.array([(-5.0 if (mask >> i) & 1 else 5.0) for i in range(dim)])
            candidates.append(corner)
        # Random points
        for _ in range(20):
            candidates.append(rng.uniform(-5, 5, dim))

        # Evaluate all candidates
        for c in candidates:
            if calls >= budget or time.time() > deadline:
                break
            eval_point(c)

        # Phase 2: Nelder-Mead refinement from best point
        # Also try NM from second-best and third-best if budget allows
        # Re-evaluate to get sorted list
        # We already have best_x, best_f. Let's run NM from best.

        if calls < budget and time.time() < deadline:
            self._nelder_mead(func, best_x, best_f, budget, dim, clip, eval_point, deadline)

        # Phase 3: If budget remains, try a second NM from a different region
        if calls < budget - 20 and time.time() < deadline:
            # Pick a random point far from best
            far = rng.uniform(-5, 5, dim)
            # Ensure it's somewhat different
            if np.linalg.norm(far - best_x) < 3.0:
                far = best_x + rng.normal(0, 3, dim)
                far = clip(far)
            f_far = eval_point(far)
            if f_far < best_f + 50:  # only if not terrible
                if calls < budget - 20 and time.time() < deadline:
                    self._nelder_mead(func, far, f_far, budget, dim, clip, eval_point, deadline)

        # Phase 4: Final fine-tuning with small steps around best
        if calls < budget and time.time() < deadline:
            self._fine_tune(func, best_x, best_f, budget, dim, clip, eval_point, deadline, rng)

        return best_x, best_f

    def _nelder_mead(self, func, x0, f0, budget, dim, clip, eval_point, deadline):
        """Run Nelder-Mead from x0, f0."""
        # Initialize simplex
        alpha = 1.0
        beta = 0.5
        gamma = 2.0
        delta = 0.5

        simplex = [x0.copy()]
        fvals = [f0]
        for i in range(dim):
            xi = x0.copy()
            if xi[i] != 0:
                xi[i] *= (1 + delta)
            else:
                xi[i] = delta
            simplex.append(clip(xi))
            fvals.append(eval_point(simplex[-1]))

        # Sort simplex
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        max_iter = 200
        for it in range(max_iter):
            if calls >= budget or time.time() > deadline:
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_ref = centroid + gamma * (centroid - simplex[-1])
            x_ref = clip(x_ref)
            f_ref = eval_point(x_ref)

            if f_ref < fvals[0]:
                # Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = clip(x_exp)
                f_exp = eval_point(x_exp)
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
                x_con = centroid + beta * (simplex[-1] - centroid)
                x_con = clip(x_con)
                f_con = eval_point(x_con)
                if f_con < fvals[-1]:
                    simplex[-1] = x_con
                    fvals[-1] = f_con
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = clip(simplex[0] + alpha * (simplex[i] - simplex[0]))
                        fvals[i] = eval_point(simplex[i])

            # Re-sort
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Check convergence
            spread = np.std(fvals)
            if spread < 1e-10:
                break

    def _fine_tune(self, func, x0, f0, budget, dim, clip, eval_point, deadline, rng):
        """Small-step local search around best point."""
        sigma = 0.1
        for _ in range(100):
            if calls >= budget or time.time() > deadline:
                break
            y = clip(x0 + sigma * rng.standard_normal(dim))
            fy = eval_point(y)
            if fy < best_f:
                x0 = y
                sigma *= 1.2
            else:
                sigma *= 0.7
                if sigma < 1e-6:
                    break
