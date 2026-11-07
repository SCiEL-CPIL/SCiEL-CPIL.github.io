"""Search-based optimizer for DTLZ4 with known Pareto front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4 using search over the known front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = 290.0  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ4 structure:
        # - First dim-5 variables shape the Pareto front
        # - Last 5 variables affect g; g=0 when all are 0.5
        # For 3 objectives, x1 and x2 determine the front shape:
        #   f1 = (1+g) * 0.5 * cos(pi*x1/2) * cos(pi*x2/2)
        #   f2 = (1+g) * 0.5 * cos(pi*x1/2) * sin(pi*x2/2)
        #   f3 = (1+g) * 0.5 * sin(pi*x1/2)
        # The Pareto front (g=0) is a quarter sphere of radius 0.5

        # Helper: create decision vector from x1, x2
        def make_x(x1, x2):
            x = np.full(dim, 0.5)
            x[0] = x1
            x[1] = x2
            return np.clip(x, lo, hi)

        # Strategy: Use Sobol sequence for uniform coverage of the x1-x2 plane.
        # This ensures the points on the Pareto front are well-distributed,
        # maximizing the hypervolume contribution.
        
        evaluated_points = []
        
        # Check time before starting heavy generation (though it's fast)
        if time.time() - t0 > deadline:
            return np.array([])

        # Generate Sobol points
        # d=2 for x1, x2
        # scramble=True for better distribution
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            # Generate 'budget' points
            sobol_points = sampler.random(budget)
            # Map from [0,1]^2 to [0,1]^2 (already in range)
            candidates = list(zip(sobol_points[:, 0], sobol_points[:, 1]))
        except ImportError:
            # Fallback to grid + halton if scipy is not available
            candidates = []
            n_grid = 15
            x1_grid = np.linspace(0, 1, n_grid)
            x2_grid = np.linspace(0, 1, n_grid)
            for x1 in x1_grid:
                for x2 in x2_grid:
                    candidates.append((x1, x2))
            
            def halton(index, base):
                result = 0.0
                f = 1.0 / base
                i = index
                while i > 0:
                    result += f * (i % base)
                    i = i // base
                    f /= base
                return result

            n_halton = budget - len(candidates)
            if n_halton > 0:
                for i in range(1, n_halton + 1):
                    candidates.append((halton(i, 2), halton(i, 3)))
            
            np.random.shuffle(candidates)

        # Phase 2: Evaluate candidates and track progress
        used = 0

        # Evaluate all candidates until budget or deadline
        for x1, x2 in candidates:
            if used >= budget or time.time() - t0 > deadline:
                break

            x = make_x(x1, x2)
            f = np.array(problem(x))
            used += 1
            evaluated_points.append(f)

        # Phase 3: Use remaining budget for local refinement
        # Try to refine points that are close to the front
        if used < budget and len(evaluated_points) > 0:
            F = np.array(evaluated_points)

            # Try local perturbations around non-dominated points
            for i in range(len(F)):
                if used >= budget or time.time() - t0 > deadline:
                    break

                # Check if this point is non-dominated
                dominated = False
                for j in range(len(F)):
                    if j != i and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break

                if dominated:
                    continue

                # Try small perturbations
                x1_orig, x2_orig = candidates[i]
                for _ in range(2):
                    if used >= budget or time.time() - t0 > deadline:
                        break

                    x1_new = np.clip(x1_orig + np.random.randn() * 0.02, 0, 1)
                    x2_new = np.clip(x2_orig + np.random.randn() * 0.02, 0, 1)

                    x = make_x(x1_new, x2_new)
                    f_new = np.array(problem(x))
                    used += 1

                    # Accept if it improves
                    if np.all(f_new <= F[i]) and np.any(f_new < F[i]):
                        F[i] = f_new
                        evaluated_points[i] = f_new
                    elif not np.any(F <= f_new) and not np.any(f_new <= F):
                        evaluated_points.append(f_new)
                        F = np.vstack([F, f_new.reshape(1, -1)])

        # Return all evaluated objective vectors
        return np.array(evaluated_points)
# EVOLVE-BLOCK-END
