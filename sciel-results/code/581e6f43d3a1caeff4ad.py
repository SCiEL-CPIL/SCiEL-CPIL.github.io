"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Halton space-filling exploration + adaptive CMA-ES local refinement for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280  # safety margin before 300s kill
        budget = self.budget
        dim = self.dim

        def time_left():
            return deadline - time.time()

        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        all_points = []  # list of (x, f)

        def eval_point(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time_left() < 0.5:
                return np.inf
            calls += 1
            f = func(x)
            all_points.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Halton sequence generator ---
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r

        def to_box(pt_unit):
            return pt_unit * 10.0 - 5.0

        # --- Phase 1: Halton-based global exploration ---
        # Use enough Halton points for good space-filling coverage of [-5,5]^dim
        n_halton = min(100, budget // 3)
        bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]  # first primes
        halton_pts = []
        for i in range(1, n_halton + 1):
            pt_unit = np.array([halton(i, bases[d]) for d in range(dim)])
            pt = to_box(pt_unit)
            halton_pts.append(pt)

        # Add strategic extra points: origin and axis midpoints
        extra_pts = [np.zeros(dim)]
        for d in range(dim):
            for s in [-1, 1]:
                pt = np.zeros(dim)
                pt[d] = s * 2.5
                extra_pts.append(pt)

        all_explore_pts = halton_pts + extra_pts

        for pt in all_explore_pts:
            if calls >= budget or time_left() < 2.0:
                break
            pt = np.clip(pt, -5, 5)
            eval_point(pt)

        if best_f == np.inf:
            # Fallback: evaluate origin
            eval_point(np.zeros(dim))

        # --- Phase 2: CMA-ES-like local refinement from best point ---
        # We implement a simplified CMA-ES that adapts the step size (sigma)
        # based on progress, and uses a population of candidate points.
        pop_size = 10  # population size for local search
        max_cma_iters = 30  # max iterations of CMA-ES
        sigma = 1.0  # initial step size
        sigma_decay = 0.8  # decay factor when no improvement
        sigma_grow = 1.3  # growth factor when improvement
        min_sigma = 0.01
        max_sigma = 5.0

        # Mean is initialized to best point
        mean = best_x.copy()
        mean_f = best_f

        # Track history of mean function values for adaptive sigma
        mean_history = [mean_f]

        for cma_iter in range(max_cma_iters):
            if calls >= budget - 5 or time_left() < 3.0:
                break

            # Generate population around current mean
            candidates = []
            for k in range(pop_size):
                # Use deterministic pseudo-random offsets based on iteration and index
                # This ensures reproducibility and avoids stochastic noise
                seed_val = cma_iter * 1000 + k
                rng = np.random.RandomState(seed_val)
                offset = rng.randn(dim) * sigma
                x_cand = mean + offset
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)

            # Evaluate all candidates
            cand_results = []
            for x_cand in candidates:
                if calls >= budget or time_left() < 1.0:
                    break
                f_cand = eval_point(x_cand)
                cand_results.append((x_cand, f_cand))

            if not cand_results:
                break

            # Sort candidates by function value
            cand_results.sort(key=lambda x: x[1])
            best_cand_x, best_cand_f = cand_results[0]

            # Update mean: weighted average of top candidates
            # Use top 3 candidates for new mean
            top_k = min(3, len(cand_results))
            weights = np.array([1.0, 0.5, 0.25][:top_k])
            weights /= weights.sum()
            new_mean = np.zeros(dim)
            for w, (xc, fc) in zip(weights, cand_results[:top_k]):
                new_mean += w * xc
            new_mean = np.clip(new_mean, -5, 5)

            # Adapt sigma based on progress
            if best_cand_f < mean_f - 1e-8:
                # Improvement: grow sigma slightly to explore more
                sigma = min(max_sigma, sigma * sigma_grow)
            else:
                # No improvement: shrink sigma
                sigma = max(min_sigma, sigma * sigma_decay)

            # Update mean
            mean = new_mean
            mean_f = best_cand_f
            mean_history.append(mean_f)

            # Convergence check: if sigma is very small and no recent improvement, stop
            if sigma < 0.05 and len(mean_history) > 5:
                recent = mean_history[-5:]
                if max(recent) - min(recent) < 1e-6:
                    break

        # --- Phase 3: Final coordinate descent polish ---
        # Do a few rounds of coordinate descent with decreasing step sizes
        step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
        for step in step_sizes:
            if calls >= budget - 2 or time_left() < 2.0:
                break
            for d in range(dim):
                if calls >= budget or time_left() < 1.0:
                    break
                for s in [-1, 1]:
                    if calls >= budget or time_left() < 1.0:
                        break
                    x_try = best_x.copy()
                    x_try[d] += s * step
                    x_try = np.clip(x_try, -5, 5)
                    # Check if already evaluated
                    already = False
                    for ep, ef in all_points[-50:]:
                        if np.allclose(ep, x_try, atol=1e-7):
                            already = True
                            break
                    if not already:
                        f_try = eval_point(x_try)
                        if f_try < best_f:
                            best_f = f_try
                            best_x = x_try.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
