"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead ensemble for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        rng = np.random.default_rng(42)

        dim = self.dim
        budget = self.budget
        call_count = 0

        def call_func(x):
            nonlocal call_count
            if call_count >= budget:
                return None
            call_count += 1
            return func(np.clip(x, -5, 5))

        # Latin Hypercube Sampling for 5 diverse starting points
        n_starts = 5
        starts = []
        for i in range(n_starts):
            # LHS: divide each dimension into n_starts equal intervals,
            # sample one point per interval, then permute
            pts = []
            for d in range(dim):
                intervals = np.linspace(-5, 5, n_starts + 1)
                idx = (i + rng.integers(0, n_starts)) % n_starts
                pts.append(rng.uniform(intervals[idx], intervals[idx + 1]))
            starts.append(np.array(pts))

        best_x = None
        best_f = np.inf

        # Track best value for early progress scoring
        # We'll track the best-so-far at each call to estimate AOCC

        # Phase 1: Initialize all simplex vertices and evaluate
        # Nelder-Mead simplex has dim+1 = 6 vertices
        # But we only have 5 starts. Let's use a different approach:
        # For each start, create a simplex around it.

        # Actually, let me reconsider. With 500 budget and 5 starts,
        # each start gets ~100 evaluations. Nelder-Mead in 5D needs
        # at least 6 evaluations per iteration (6 vertices).
        # That's about 16 iterations per start.

        # Let's do: for each of 5 starts, run Nelder-Mead with ~100 budget
        # But we need to track global best for AOCC.

        # Alternative: interleave evaluations from all 5 Nelder-Mead runs
        # to maximize early progress.

        # Let's implement a simple Nelder-Mead that we can interleave.

        class NelderMead:
            def __init__(self, x0, func, rng, dim):
                self.dim = dim
                self.rng = rng
                self.func = func
                # Create simplex: x0 plus dim perturbed points
                self.simplex = [x0.copy()]
                for i in range(dim):
                    perturbed = x0.copy()
                    perturbed[i] += 0.5
                    self.simplex.append(perturbed)
                self.values = [None] * (dim + 1)
                self.n_calls = 0
                self.initialized = False

            def evaluate_vertex(self, idx):
                """Evaluate vertex idx if not already done. Returns value or None if budget exceeded."""
                if self.values[idx] is not None:
                    return self.values[idx]
                val = self.func(self.simplex[idx])
                if val is None:
                    return None
                self.values[idx] = val
                self.n_calls += 1
                if all(v is not None for v in self.values):
                    self.initialized = True
                return val

            def step(self):
                """Perform one Nelder-Mead step. Returns True if successful, False if budget exceeded."""
                if not self.initialized:
                    # Need to evaluate remaining vertices
                    for i in range(self.dim + 1):
                        if self.values[i] is None:
                            val = self.evaluate_vertex(i)
                            if val is None:
                                return False
                    self.initialized = True

                # Sort simplex by function value
                order = np.argsort(self.values)
                self.simplex = [self.simplex[i] for i in order]
                self.values = [self.values[i] for i in order]

                x_worst = self.simplex[-1]
                x_best = self.simplex[0]
                f_worst = self.values[-1]
                f_best = self.values[0]

                # Centroid of all but worst
                centroid = np.mean(self.simplex[:-1], axis=0)

                # Reflection
                alpha = 1.0
                x_ref = centroid + alpha * (centroid - x_worst)
                f_ref = self.func(x_ref)
                if f_ref is None:
                    return False
                self.n_calls += 1

                if f_ref < self.values[0]:
                    # Expansion
                    gamma = 2.0
                    x_exp = centroid + gamma * (centroid - x_worst)
                    f_exp = self.func(x_exp)
                    if f_exp is None:
                        return False
                    self.n_calls += 1
                    if f_exp < f_ref:
                        self.simplex[-1] = x_exp
                        self.values[-1] = f_exp
                    else:
                        self.simplex[-1] = x_ref
                        self.values[-1] = f_ref
                elif f_ref < self.values[-2]:
                    self.simplex[-1] = x_ref
                    self.values[-1] = f_ref
                else:
                    # Contraction
                    rho = 0.5
                    x_con = centroid + rho * (x_worst - centroid)
                    f_con = self.func(x_con)
                    if f_con is None:
                        return False
                    self.n_calls += 1
                    if f_con < f_worst:
                        self.simplex[-1] = x_con
                        self.values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            self.simplex[i] = x_best + 0.5 * (self.simplex[i] - x_best)
                            self.values[i] = self.func(self.simplex[i])
                            if self.values[i] is None:
                                return False
                            self.n_calls += 1
                return True

            def best(self):
                if not self.initialized:
                    return None, np.inf
                min_idx = np.argmin(self.values)
                return self.simplex[min_idx], self.values[min_idx]

        # Create 5 Nelder-Mead optimizers
        nms = []
        for i in range(n_starts):
            nm = NelderMead(starts[i], call_func, rng, dim)
            nms.append(nm)

        # Interleave steps from all Nelder-Mead runs
        # Each "round" we take one step from each active NM
        active = list(range(n_starts))
        max_rounds = 100  # Safety: 5 * 100 * ~6 = 3000, but we stop at budget

        for round_num in range(max_rounds):
            if call_count >= budget:
                break
            if time.time() > deadline:
                break

            new_active = []
            for idx in active:
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                nm = nms[idx]
                success = nm.step()
                if success:
                    new_active.append(idx)
                    # Update global best
                    bx, bf = nm.best()
                    if bf < best_f:
                        best_f = bf
                        best_x = bx.copy()

            if not new_active:
                break
            active = new_active

        # If we have budget left, do a final polish with coordinate descent
        # on the best point found
        if best_x is not None and call_count < budget and time.time() < deadline:
            # Coordinate descent polish
            x = best_x.copy()
            fx = best_f
            for coord in range(dim):
                if call_count >= budget or time.time() > deadline:
                    break
                # Try small steps in both directions
                for delta in [0.5, 0.1, 0.01]:
                    if call_count >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if call_count >= budget or time.time() > deadline:
                            break
                        x_try = x.copy()
                        x_try[coord] += sign * delta
                        x_try = np.clip(x_try, -5, 5)
                        f_try = call_func(x_try)
                        if f_try is None:
                            break
                        if f_try < fx:
                            x, fx = x_try, f_try
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        if best_x is None:
            # Fallback: return first start
            best_x = starts[0]
            best_f = call_func(best_x)
            if best_f is None:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
