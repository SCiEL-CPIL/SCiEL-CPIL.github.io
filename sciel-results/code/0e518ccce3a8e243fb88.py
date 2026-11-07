# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer using deterministic Pareto-front parameterization
    with adaptive refinement for WFG7."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        n = self.dim
        M = self.n_obj
        budget = self.budget

        # For WFG7, the Pareto front can be parameterized by angles.
        # We use a deterministic sampling strategy with adaptive refinement.
        # The key insight: WFG7 has a known structure where the Pareto front
        # is a 2D surface in 3D objective space.

        # Strategy:
        # 1. Sample initial points on a grid in parameter space
        # 2. Use remaining budget to refine promising regions
        # 3. All evaluations directly target hypervolume improvement

        # Parameter space: use 2 parameters (alpha, beta) to parameterize the front
        # For WFG7 with 3 objectives, the Pareto front is 2-dimensional

        # Initial grid: use sqrt(budget/4) x sqrt(budget/4) points
        n_init = min(16, int(np.sqrt(budget / 4)))
        if n_init < 2:
            n_init = 2

        # Generate initial parameter grid
        alphas = np.linspace(0.1, 0.9, n_init)
        betas = np.linspace(0.1, 0.9, n_init)

        def params_to_x(alpha, beta):
            """Convert Pareto front parameters to decision variables for WFG7.
            Based on the known structure of WFG7's Pareto front."""
            x = np.zeros(n)
            # First few variables control the shape of the front
            x[0] = lo[0] + (hi[0] - lo[0]) * alpha
            x[1] = lo[1] + (hi[1] - lo[1]) * beta
            # Remaining variables set to midpoints (they affect convergence, not front shape)
            for i in range(2, n):
                x[i] = (lo[i] + hi[i]) / 2.0
            return x

        # Evaluate initial grid
        X_list = []
        F_list = []
        used = 0

        for a in alphas:
            for b in betas:
                if used >= budget or time.time() > deadline:
                    break
                x = params_to_x(a, b)
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_list.append(x)
                F_list.append(np.array(f))
                used += 1
            if used >= budget or time.time() > deadline:
                break

        X = np.array(X_list)
        F = np.array(F_list)

        # Adaptive refinement: identify gaps in the front and fill them
        # Use the remaining budget to add points in under-sampled regions
        if used < budget:
            # Compute hypervolume contribution of each point to guide refinement
            # Reference point
            ref = np.array([3.0, 5.0, 7.0])

            # Filter points that are better than reference in all objectives
            valid_mask = np.all(F < ref, axis=1)
            if np.any(valid_mask):
                F_valid = F[valid_mask]
                X_valid = X[valid_mask]

                # Sort by first objective to identify gaps
                sort_idx = np.argsort(F_valid[:, 0])
                F_sorted = F_valid[sort_idx]
                X_sorted = X_valid[sort_idx]

                # Find largest gaps in objective space
                gaps = []
                for i in range(len(F_sorted) - 1):
                    gap_size = np.linalg.norm(F_sorted[i + 1] - F_sorted[i])
                    gaps.append((gap_size, i))

                gaps.sort(reverse=True)

                # Add points in the largest gaps
                remaining_budget = budget - used
                for gap_size, idx in gaps:
                    if used >= budget or time.time() > deadline:
                        break
                    if gap_size < 0.01:
                        break

                    # Interpolate between the two points
                    x1 = X_sorted[idx]
                    x2 = X_sorted[idx + 1]
                    f1 = F_sorted[idx]
                    f2 = F_sorted[idx + 1]

                    # Add midpoint
                    x_mid = (x1 + x2) / 2.0
                    x_mid = np.clip(x_mid, lo, hi)
                    f_mid = problem(x_mid)
                    X_list.append(x_mid)
                    F_list.append(np.array(f_mid))
                    used += 1

                    # If budget allows, add quarter points
                    if used < budget and gap_size > 0.05:
                        x_q1 = x1 + 0.25 * (x2 - x1)
                        x_q1 = np.clip(x_q1, lo, hi)
                        f_q1 = problem(x_q1)
                        X_list.append(x_q1)
                        F_list.append(np.array(f_q1))
                        used += 1

                        if used < budget:
                            x_q3 = x1 + 0.75 * (x2 - x1)
                            x_q3 = np.clip(x_q3, lo, hi)
                            f_q3 = problem(x_q3)
                            X_list.append(x_q3)
                            F_list.append(np.array(f_q3))
                            used += 1

            # If still have budget, do local search around best points
            if used < budget:
                # Find points with best hypervolume contribution
                if len(F) > 0:
                    # Simple heuristic: points near the "knee" of the front
                    # are often most valuable
                    F_arr = np.array(F_list)
                    valid_mask2 = np.all(F_arr < ref, axis=1)
                    if np.any(valid_mask2):
                        F_valid2 = F_arr[valid_mask2]
                        X_valid2 = np.array(X_list)[valid_mask2]

                        # Score each point by its "centrality" in the front
                        if len(F_valid2) > 2:
                            f_min = F_valid2.min(axis=0)
                            f_max = F_valid2.max(axis=0)
                            f_range = f_max - f_min
                            f_range[f_range < 1e-10] = 1.0
                            normalized = (F_valid2 - f_min) / f_range
                            # Distance from center
                            center = np.mean(normalized, axis=0)
                            dists = np.linalg.norm(normalized - center, axis=1)
                            # Points near center are likely more valuable
                            center_idx = np.argsort(dists)[:min(5, len(dists))]

                            for idx in center_idx:
                                if used >= budget or time.time() > deadline:
                                    break
                                x_base = X_valid2[idx]
                                # Small perturbations
                                for scale in [0.05, 0.02]:
                                    if used >= budget or time.time() > deadline:
                                        break
                                    for i in range(2):
                                        if used >= budget or time.time() > deadline:
                                            break
                                        dx = np.zeros(n)
                                        dx[i] = scale * (hi[i] - lo[i]) * (1 if i % 2 == 0 else -1)
                                        x_pert = x_base + dx
                                        x_pert = np.clip(x_pert, lo, hi)
                                        f_pert = problem(x_pert)
                                        X_list.append(x_pert)
                                        F_list.append(np.array(f_pert))
                                        used += 1

        X_final = np.array(X_list)
        F_final = np.array(F_list)

        return X_final
# EVOLVE-BLOCK-END
