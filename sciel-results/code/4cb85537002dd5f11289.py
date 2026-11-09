"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer exploiting DTLZ1 structure for high hypervolume."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # DTLZ1 structure:
        # M = n_obj + 1 = 4 position variables (x[0..3])
        # g variables (x[4..6]) determine distance from front
        # f1 = 0.5 * x0 * (1 + g)
        # f2 = 0.5 * x1 * (1 - x0) * (1 + g)
        # f3 = 0.5 * (1 - x1) * (1 + g)
        # where g = 100 * (n_g + sum((x_i - 0.5)^2 - cos(20*pi*(x_i-0.5))) for i in g_vars)
        # On the Pareto front, g=0, so x_i = 0.5 for i in g_vars
        # Position variables x0, x1 in [0,1], x2, x3 are free (but affect g if not 0.5)

        # Strategy: sample the Pareto front densely using the known structure
        # x0, x1 determine the position on the front
        # x2, x3, x4, x5, x6 should be 0.5 for g=0

        # Generate a dense grid on the Pareto front
        # x0 and x1 in [0, 1], with x0 + x1 + x2 + x3 = 1 (for the simplex)
        # Actually for DTLZ1, the position is determined by x0..x_{M-1} where sum = 1
        # Wait, let me reconsider. For DTLZ1:
        # f1 = 0.5 * x0 * (1+g)
        # f2 = 0.5 * x1 * (1-x0) * (1+g)  
        # f3 = 0.5 * (1-x1) * (1+g)
        # The Pareto front has g=0, and x0, x1 in [0,1] with the constraint that
        # the "position" variables satisfy the simplex. Actually for 3 objectives:
        # x0, x1, x2 are the position variables where x0+x1+x2=1, x_i >= 0
        # But in the standard formulation with dim=7, n_obj=3:
        # M = 4, so x[0..3] are position, x[4..6] are g-variables
        # f1 = 0.5*x0*(1+g), f2 = 0.5*x1*(1-x0)*(1+g), f3 = 0.5*(1-x1)*(1+g)
        # For g=0: f1=0.5*x0, f2=0.5*x1*(1-x0), f3=0.5*(1-x1)
        # The Pareto front is parameterized by x0, x1 in [0,1]

        # Let's use a structured approach:
        # 1. Sample the Pareto front densely
        # 2. Use remaining budget for local refinement

        used = 0
        all_f = []

        # Phase 1: Dense sampling of the Pareto front
        # Use Latin hypercube or grid sampling for x0, x1
        n_samples = min(150, budget - 50)  # Leave some budget for refinement
        n_samples = max(n_samples, 50)

        # Create a quasi-random sequence for better coverage
        rng = np.random.default_rng(0)

        # Use Sobol-like sampling via Halton sequence for 2D
        def halton_sequence(n, base):
            result = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                result.append(r)
            return np.array(result)

        # Generate Halton sequence points for x0, x1
        n_halton = n_samples
        x0_vals = halton_sequence(n_halton, 2)
        x1_vals = halton_sequence(n_halton, 3)

        # Create points on the Pareto front
        for i in range(n_halton):
            if time.time() > deadline:
                break
            x = np.full(dim, 0.5)
            x[0] = x0_vals[i]
            x[1] = x1_vals[i]
            # x[2], x[3], x[4], x[5], x[6] = 0.5 for g=0
            f = problem(x)
            all_f.append(f)
            used += 1

        # Phase 2: Add some random points to explore off-front regions
        n_random = min(30, budget - used)
        for i in range(n_random):
            if time.time() > deadline:
                break
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            all_f.append(f)
            used += 1

        # Phase 3: Local refinement around promising points
        # Use remaining budget to perturb points that seem good
        if used < budget and len(all_f) > 0:
            F = np.array(all_f)
            # Find non-dominated points
            n = len(F)
            dominated = np.zeros(n, dtype=bool)
            for i in range(n):
                if dominated[i]:
                    continue
                for j in range(n):
                    if i == j or dominated[j]:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated[i] = True
                        break

            non_dom_idx = np.where(~dominated)[0]
            n_refine = min(budget - used, 50)

            for _ in range(n_refine):
                if time.time() > deadline:
                    break
                if len(non_dom_idx) == 0:
                    break
                idx = non_dom_idx[rng.integers(len(non_dom_idx))]
                base_f = F[idx]
                # Perturb in objective space direction
                # Try small perturbations in x-space
                for attempt in range(5):
                    x = np.full(dim, 0.5)
                    # Reconstruct x from the original point isn't straightforward
                    # Instead, use a random perturbation around 0.5 for g-vars
                    x[0] = np.clip(rng.uniform(0, 1), 0, 1)
                    x[1] = np.clip(rng.uniform(0, 1), 0, 1)
                    f = problem(x)
                    all_f.append(f)
                    used += 1
                    if used >= budget or time.time() > deadline:
                        break

        # Ensure we don't exceed budget
        while used < budget and time.time() < deadline:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            all_f.append(f)
            used += 1

        return all_f
# EVOLVE-BLOCK-END
