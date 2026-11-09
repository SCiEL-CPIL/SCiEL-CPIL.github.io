"""Portfolio optimizer for BBOB f23 (Katsuura) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: LHS exploration + (1+1)-ES exploitation with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before the 330s kill

        best_x = None
        best_f = np.inf
        calls_made = 0

        def call_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # ---------- Phase 1: Latin Hypercube Sampling for diverse initial points ----------
        # Use LHS to spread initial points across the domain
        n_lhs = min(20, budget // 10)
        lhs_points = []
        for d in range(dim):
            perm = np.random.default_rng(42 + d).permutation(n_lhs)
            lhs_points.append((perm + 0.5) / n_lhs * 10 - 5)
        lhs_matrix = np.array(lhs_points).T  # shape (n_lhs, dim)

        for i in range(n_lhs):
            if time.time() > deadline - 5:
                break
            x = lhs_matrix[i]
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # ---------- Phase 2: (1+1)-ES with adaptive sigma from best point ----------
        if best_x is None:
            x = np.zeros(dim)
            fx = call_func(x)
            best_x = x.copy()
            best_f = fx

        # Run multiple restarts of (1+1)-ES from the best point and nearby points
        rng = np.random.default_rng(123)
        
        # Generate a set of starting points: best + perturbations
        start_points = [best_x.copy()]
        for _ in range(5):
            pert = best_x + rng.uniform(-1, 1, dim)
            start_points.append(np.clip(pert, -5, 5))

        for start_idx, x in enumerate(start_points):
            if time.time() > deadline - 10:
                break
            if calls_made >= budget - 5:
                break

            # Evaluate starting point if not already evaluated
            if start_idx == 0:
                fx = best_f
            else:
                fx = call_func(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

            sigma = 1.0
            # Adaptive (1+1)-ES with success rule
            success_count = 0
            total_steps = 0
            max_steps_for_this_restart = min(budget - calls_made - 5, 150)

            for _ in range(max_steps_for_this_restart):
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget - 2:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                total_steps += 1

                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.25
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                # Adapt sigma based on success rate over last 10 steps
                if total_steps % 10 == 0:
                    if success_count > 5:
                        sigma *= 1.1
                    elif success_count < 3:
                        sigma *= 0.9
                    success_count = 0

                # If sigma gets too small, try a larger jump
                if sigma < 0.01:
                    sigma = 0.5

        # ---------- Phase 3: Fine-tuning with smaller steps from best ----------
        if time.time() < deadline - 5 and calls_made < budget - 2:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            fine_steps = min(budget - calls_made - 2, 50)
            
            for _ in range(fine_steps):
                if time.time() > deadline - 3:
                    break
                if calls_made >= budget - 1:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                if sigma < 0.001:
                    sigma = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
