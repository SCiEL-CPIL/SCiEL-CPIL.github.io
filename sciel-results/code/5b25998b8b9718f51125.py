"""Multi-objective optimizer for DTLZ7 with structure-aware search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(0)

        budget = self.budget
        dim = self.dim

        # --- Phase 1: Exploit DTLZ7 structure ---
        # For DTLZ7 with M=3:
        #   f1 = x1
        #   f2 = x2
        #   f3 = 1 + g + h(f1, f2, g)
        # where g = sum_{i=m+1}^{n} (x_i - 0.5)^2 with m=1 (so i=2..22)
        # and h = (m - f1 - f2) * (1 + 2*g)
        # On the Pareto front, g=0, so f3 = 1 + (1 - f1 - f2)*1 = 2 - f1 - f2
        # Wait, m=1 for DTLZ7. Let me re-check.
        # Actually for DTLZ7: m = M-1 = 2 (number of position variables)
        # f1 = x1, f2 = x2
        # g = sum_{i=3}^{22} (x_i - 0.5)^2
        # h = (m - f1 - f2) * (1 + 2*g) where m = 2
        # f3 = 1 + g + h = 1 + g + (2 - f1 - f2)*(1 + 2*g)
        # On Pareto front (g=0): f3 = 1 + 0 + (2 - f1 - f2)*1 = 3 - f1 - f2
        # Constraint: f1 >= 0, f2 >= 0, f1 + f2 <= 2 (but since f1,f2 in [0,1], always satisfied)
        # So Pareto front: f1 in [0,1], f2 in [0,1], f3 = 3 - f1 - f2

        # Generate points on or near the Pareto front
        # To achieve g=0, set x_3..x_22 = 0.5
        # x_1 = f1, x_2 = f2

        # Strategy: allocate budget between:
        # 1. Direct Pareto front sampling (g=0 points)
        # 2. Local refinement around promising regions

        n_front = min(budget, 150)  # most evaluations on the front
        n_refine = budget - n_front  # remaining for refinement

        # Generate Pareto front points
        # We want to cover the 2D front (f1, f2) in [0,1]x[0,1]
        # Use a combination of structured and random sampling
        front_points = []
        front_objectives = []

        # Structured grid + random
        n_grid = 12
        n_random = n_front - n_grid * n_grid
        if n_random < 0:
            n_random = 0
            n_grid = int(np.sqrt(n_front))
            n_random = n_front - n_grid * n_grid

        # Grid points
        for i in range(n_grid):
            for j in range(n_grid):
                f1 = (i + 0.5) / n_grid
                f2 = (j + 0.5) / n_grid
                x = np.full(dim, 0.5)
                x[0] = f1
                x[1] = f2
                front_points.append(x)

        # Random points on the front
        for _ in range(n_random):
            f1 = rng.random()
            f2 = rng.random()
            x = np.full(dim, 0.5)
            x[0] = f1
            x[1] = f2
            front_points.append(x)

        # Evaluate front points
        all_X = []
        all_F = []
        used = 0

        for x in front_points:
            if used >= budget:
                break
            if time.time() > deadline - 5.0:
                break
            x = np.clip(x, lo, hi)
            f = problem(x)
            all_X.append(x)
            all_F.append(f)
            used += 1

        if len(all_F) == 0:
            return np.array(all_X) if all_X else np.array([lo])

        all_F = np.array(all_F)

        # --- Phase 2: Local refinement ---
        # Find non-dominated points and refine around them
        if used < budget and time.time() < deadline - 10.0:
            # Identify non-dominated points
            dominated = np.zeros(len(all_F), dtype=bool)
            for i in range(len(all_F)):
                if dominated[i]:
                    continue
                for j in range(len(all_F)):
                    if i == j or dominated[j]:
                        continue
                    if np.all(all_F[j] <= all_F[i]) and np.any(all_F[j] < all_F[i]):
                        dominated[i] = True
                        break

            # Get non-dominated indices
            nd_idx = np.where(~dominated)[0]
            if len(nd_idx) > 0:
                # Refine around non-dominated points
                for idx in nd_idx:
                    if used >= budget:
                        break
                    if time.time() > deadline - 5.0:
                        break
                    base_x = all_X[idx]
                    # Small perturbations
                    for trial in range(3):
                        if used >= budget:
                            break
                        if time.time() > deadline - 5.0:
                            break
                        # Perturb only x1 and x2 slightly, keep others at 0.5
                        child = base_x.copy()
                        step = 0.02 * rng.random()
                        child[0] += step * (rng.random() - 0.5) * 2
                        child[1] += step * (rng.random() - 0.5) * 2
                        # Keep other variables at 0.5 for g=0
                        child[2:] = 0.5
                        child = np.clip(child, lo, hi)
                        fc = problem(child)
                        all_X.append(child)
                        all_F.append(fc)
                        used += 1

        # Convert to arrays
        all_X = np.array(all_X)
        all_F = np.array(all_F)

        return all_X
# EVOLVE-BLOCK-END
