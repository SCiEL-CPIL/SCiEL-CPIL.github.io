"""Ridge-aware optimizer using adaptive coordinate descent with Nelder-Mead polishing for BBOB f13."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer for Sharp Ridge (BBOB f13).
    Strategy:
      1. Coarse random sampling to find the basin.
      2. Adaptive coordinate descent with line search to follow the ridge.
      3. Nelder-Mead for final polishing.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 285.0

        def elapsed_ok():
            return time.time() < deadline

        rng = np.random.default_rng(123)

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        best_x = None
        best_f = np.inf
        evaluated = 0

        def eval_point(x):
            nonlocal evaluated, best_x, best_f
            x = np.asarray(x, dtype=float)
            x = clip(x)
            f = float(func(x))
            evaluated += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: Random sampling to find the basin
        n_samples = 40
        for _ in range(n_samples):
            if not elapsed_ok() or evaluated >= budget:
                break
            x = rng.uniform(-4.0, 4.0, size=dim)
            eval_point(x)

        if best_x is None:
            return None, np.inf

        # Phase 2: Adaptive coordinate descent with line search
        # This is effective for ridge functions because it can follow
        # the ridge direction while making progress in all dimensions.
        x = best_x.copy()
        f = best_f
        step_sizes = np.ones(dim) * 2.0  # Initial step sizes
        no_improve = 0
        max_cd_iters = 200

        for it in range(max_cd_iters):
            if not elapsed_ok() or evaluated >= budget:
                break
            improved = False
            for i in range(dim):
                if not elapsed_ok() or evaluated >= budget:
                    break
                # Try positive and negative steps
                for sign in [1, -1]:
                    x_try = x.copy()
                    x_try[i] += sign * step_sizes[i]
                    x_try = clip(x_try)
                    f_try = eval_point(x_try)
                    if f_try < f:
                        # Success: move and possibly increase step
                        x = x_try
                        f = f_try
                        improved = True
                        no_improve = 0
                        # Adaptive step: increase if we improved significantly
                        step_sizes[i] = min(step_sizes[i] * 1.5, 5.0)
                        break
                    else:
                        # Failure: shrink step
                        step_sizes[i] *= 0.5
            if not improved:
                no_improve += 1
                # If no improvement for several iterations, try a random restart
                if no_improve > 15:
                    x = best_x + rng.normal(0, 0.5, size=dim)
                    x = clip(x)
                    f = eval_point(x)
                    step_sizes = np.ones(dim) * 1.0
                    no_improve = 0
                    if f > best_f + 1.0:
                        break  # Restart didn't help, move on

        # Phase 3: Nelder-Mead polishing
        if evaluated < budget and elapsed_ok():
            x0 = best_x.copy()
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            # Initialize simplex
            delta = 1.0
            simplex = [x0.copy()]
            for i in range(dim):
                x_pert = x0.copy()
                step = delta if x0[i] < 5.0 - delta else -delta
                x_pert[i] += step
                simplex.append(clip(x_pert))

            simplex_f = []
            for x in simplex:
                if evaluated >= budget or not elapsed_ok():
                    break
                simplex_f.append(eval_point(x))

            if len(simplex_f) == dim + 1:
                max_iters = 100
                for _ in range(max_iters):
                    if evaluated >= budget or not elapsed_ok():
                        break

                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]

                    if simplex_f[-1] - simplex_f[0] < 1e-10:
                        break

                    centroid = np.mean(simplex[:-1], axis=0)
                    x_ref = centroid + alpha * (centroid - simplex[-1])
                    x_ref = clip(x_ref)
                    f_ref = eval_point(x_ref)

                    if evaluated >= budget or not elapsed_ok():
                        break

                    if f_ref < simplex_f[0]:
                        x_exp = centroid + gamma * (x_ref - centroid)
                        x_exp = clip(x_exp)
                        f_exp = eval_point(x_exp)
                        if evaluated >= budget or not elapsed_ok():
                            break
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            simplex_f[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            simplex_f[-1] = f_ref
                    elif f_ref < simplex_f[-2]:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                    else:
                        if f_ref < simplex_f[-1]:
                            x_con = centroid + rho * (x_ref - centroid)
                        else:
                            x_con = centroid - rho * (centroid - simplex[-1])
                        x_con = clip(x_con)
                        f_con = eval_point(x_con)
                        if evaluated >= budget or not elapsed_ok():
                            break
                        if f_con < min(f_ref, simplex_f[-1]):
                            simplex[-1] = x_con
                            simplex_f[-1] = f_con
                        else:
                            for i in range(1, len(simplex)):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = clip(simplex[i])
                                if evaluated >= budget or not elapsed_ok():
                                    break
                                simplex_f[i] = eval_point(simplex[i])

        return best_x, best_f
# EVOLVE-BLOCK-END
