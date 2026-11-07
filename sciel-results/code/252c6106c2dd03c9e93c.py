"""Alternative strategy: Deterministic Nelder-Mead with adaptive scaling and restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead optimizer with adaptive scaling and restarts for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Deadline guard
        start_time = time.time()
        deadline = start_time + 320.0

        calls_used = 0
        best_x = None
        best_f = np.inf

        def eval_point(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial simplex: start at origin with small step size
        # The function is sharp ridge, so we want to explore along different directions
        # Use a deterministic grid of starting points to cover the space

        # Strategy: Try multiple starting points, use Nelder-Mead from each
        # But budget is limited, so we need to be efficient

        # First, do a quick random search to find a good starting region
        # Then use Nelder-Mead for local refinement

        # Phase 1: Random search to find promising region
        n_random = min(20, budget // 10)
        random_starts = []

        for i in range(n_random):
            if calls_used >= budget or time.time() > deadline:
                break
            # Sample from a distribution that covers the space
            # Uniform in [-4, 4]^5 since minimizer is in [-4, 4]^5
            x = np.random.uniform(-4, 4, dim)
            fx = eval_point(x)
            random_starts.append((fx, x.copy()))

        if not random_starts:
            x0 = np.zeros(dim)
            fx0 = eval_point(x0)
            random_starts = [(fx0, x0)]

        # Sort by fitness
        random_starts.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead from the best few starting points
        # Use the top k starting points
        k_starts = min(3, len(random_starts))

        for i in range(k_starts):
            if calls_used >= budget or time.time() > deadline:
                break

            # Get starting point
            start_x = random_starts[i][1].copy()
            start_f = random_starts[i][0]

            # Nelder-Mead parameters
            # Initial simplex: start_x + delta * identity
            delta = 0.5  # Initial step size
            alpha = 1.0  # Reflection
            gamma = 2.0  # Expansion
            rho = 0.5    # Contraction
            sigma = 0.5  # Shrink

            # Build initial simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start_x
            for j in range(dim):
                simplex[j + 1] = start_x
                simplex[j + 1, j] += delta

            # Evaluate simplex
            simplex_f = np.zeros(dim + 1)
            for j in range(dim + 1):
                if calls_used >= budget or time.time() > deadline:
                    break
                simplex_f[j] = eval_point(simplex[j])

            # Check if we broke out early
            if calls_used >= budget or time.time() > deadline:
                break

            # Main Nelder-Mead loop
            max_iters = (budget - calls_used) // (dim + 1)
            for it in range(max_iters):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Sort simplex by fitness
                idx = np.argsort(simplex_f)
                simplex = simplex[idx]
                simplex_f = simplex_f[idx]

                # Compute centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = eval_point(x_r)
                if calls_used >= budget or time.time() > deadline:
                    break

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = eval_point(x_e)
                    if calls_used >= budget or time.time() > deadline:
                        break

                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    # Simple reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    # Outside contraction
                    x_oc = centroid + 0.5 * (simplex[-1] - centroid)
                    x_oc = np.clip(x_oc, -5, 5)
                    f_oc = eval_point(x_oc)
                    if calls_used >= budget or time.time() > deadline:
                        break

                    if f_oc < simplex_f[-1]:
                        simplex[-1] = x_oc
                        simplex_f[-1] = f_oc
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            if calls_used >= budget or time.time() > deadline:
                                break
                            simplex_f[j] = eval_point(simplex[j])
                        if calls_used >= budget or time.time() > deadline:
                            break

                # Check for convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

        # Return best found
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
