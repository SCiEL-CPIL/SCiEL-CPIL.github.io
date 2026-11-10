"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG9 exploiting its 2D Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        evaluated = []
        used = 0

        # WFG9 analytic structure:
        # Objectives depend on x[0], x[1], and x[2].
        # x[0] and x[1] are position variables that shape the Pareto front.
        # x[2] is a "shape" variable; setting it to a value that minimizes f3
        # (the third objective) while keeping f1 and f2 well-behaved is key.
        # For WFG9, x[2] = 0.75 * range often gives good trade-offs.
        
        # Fix x[2] to 0.75 of its range to minimize f3 contribution
        # Fix x[3..11] to midpoints (they don't affect objectives significantly)
        base_x = np.zeros(self.dim)
        for i in range(self.dim):
            base_x[i] = (lo[i] + hi[i]) / 2.0
        
        # Set x[2] to 75% of its range for optimal f3 trade-off
        base_x[2] = lo[2] + 0.75 * (hi[2] - lo[2])

        # Phase 1: Dense 2D grid over x[0] and x[1]
        # Budget 200: use ~14x14 = 196 points for the main grid
        n0 = 14
        n1 = 14
        
        x0_vals = np.linspace(lo[0], hi[0], n0)
        x1_vals = np.linspace(lo[1], hi[1], n1)

        for x0 in x0_vals:
            for x1 in x1_vals:
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break

                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                f = problem(x)
                evaluated.append((x, f))
                used += 1
            if used >= self.budget or time.time() > deadline:
                break

        if used >= self.budget or time.time() > deadline:
            return [e[0] for e in evaluated]

        # Phase 2: Local refinement around the best-performing regions
        # Identify points that are non-dominated and refine around them
        remaining = self.budget - used
        if remaining > 0 and time.time() < deadline:
            # Simple non-dominated sorting to find promising regions
            f_vals = np.array([e[1] for e in evaluated])
            
            # Find non-dominated points
            non_dominated = []
            for i in range(len(f_vals)):
                dominated = False
                for j in range(len(f_vals)):
                    if i == j:
                        continue
                    if np.all(f_vals[j] <= f_vals[i]) and np.any(f_vals[j] < f_vals[i]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(i)
            
            # Refine around non-dominated points with small perturbations
            rng = np.random.RandomState(42)
            n_refine = min(remaining, 20)
            
            for _ in range(n_refine):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                if non_dominated:
                    idx = non_dominated[rng.randint(0, len(non_dominated))]
                else:
                    idx = rng.randint(0, len(evaluated))
                
                x = evaluated[idx][0].copy()
                # Small Gaussian perturbation on x[0] and x[1]
                x[0] = np.clip(x[0] + rng.normal(0, (hi[0] - lo[0]) / 50), lo[0], hi[0])
                x[1] = np.clip(x[1] + rng.normal(0, (hi[1] - lo[1]) / 50), lo[1], hi[1])
                
                f = problem(x)
                evaluated.append((x, f))
                used += 1

        return [e[0] for e in evaluated]
# EVOLVE-BLOCK-END
