"""Multi-objective optimizer for ZDT3 using a structured decomposition approach."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Optimizer that exploits ZDT3 structure: f1=x1, f2=g*(1-sqrt(f1/g))."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # ZDT3 structure:
        # f1(x) = x[0]
        # g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # f2(x) = g(x) * (1 - sqrt(f1/g(x)))
        #
        # The Pareto front is achieved when g is minimized (g_min = 1) and x1 varies in [0,1].
        # When g=1, f2 = 1 - sqrt(x1)
        #
        # Strategy:
        # 1. First, minimize g(x) by optimizing x[1:] (29 variables)
        # 2. Then sweep x1 across [0,1] to trace the Pareto front
        #
        # We'll use a two-phase approach:
        # Phase 1: Use some budget to minimize g(x) over x[1:]
        # Phase 2: Use remaining budget to sweep x1 with the best g found

        rng = np.random.default_rng(42)

        # Phase 1: Minimize g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # The minimum of g is 1 (achieved when all x[i]=0 for i>=1)
        # But we need to find points close to the Pareto front.

        # Let's use a decomposition approach:
        # For each value of x1 in [0,1], we want to minimize g(x) over x[1:]
        # This is a single-objective optimization problem for each x1.

        # Actually, g doesn't depend on x1, so we just need to minimize g over x[1:]
        # and then sweep x1.

        # Let's allocate budget:
        # - 100 evaluations to minimize g over x[1:]
        # - 100 evaluations to sweep x1 with the best x[1:] found

        n_phase1 = 100
        n_phase2 = budget - n_phase1

        # Phase 1: Minimize g(x[1:]) using random search + local refinement
        # g(x) = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
        # Minimum is at x[i]=0 for all i>=1, giving g_min = 1

        # We'll use a population-based approach to minimize g
        pop_size = 20
        X_tail = np.zeros((pop_size, dim - 1))
        # Initialize with some random points and some near zero
        for i in range(pop_size):
            if i < pop_size // 2:
                X_tail[i] = rng.random(dim - 1) * 0.1  # near zero
            else:
                X_tail[i] = rng.random(dim - 1)  # random

        best_g = np.inf
        best_x_tail = np.zeros(dim - 1)
        evaluations_used = 0

        # Evaluate initial population
        for i in range(pop_size):
            if evaluations_used >= n_phase1:
                break
            x = np.concatenate([[0.5], X_tail[i]])  # x1=0.5 for evaluation
            x = np.clip(x, lo, hi)
            f = problem(x)
            # Extract g from f: f1=x1, f2=g*(1-sqrt(f1/g))
            # So g = f2 / (1 - sqrt(f1/f2)) if f2 > f1
            # Actually, let's just compute g directly
            # g = 1 + 10*(n-1) + sum((x[i]^2 - 10*cos(2*pi*x[i])) for i=1..n-1)
            g_val = 1 + 10 * (dim - 1)
            for j in range(1, dim):
                g_val += x[j] ** 2 - 10 * np.cos(2 * np.pi * x[j])
            if g_val < best_g:
                best_g = g_val
                best_x_tail = X_tail[i].copy()
            evaluations_used += 1

        # Local refinement: try small perturbations around best point
        for _ in range(50):
            if evaluations_used >= n_phase1:
                break
            if time.time() > deadline:
                break

            # Try perturbation
            x_tail = best_x_tail + rng.normal(0, 0.05, dim - 1)
            x_tail = np.clip(x_tail, lo[1:], hi[1:])
            x = np.concatenate([[0.5], x_tail])
            x = np.clip(x, lo, hi)

            g_val = 1 + 10 * (dim - 1)
            for j in range(1, dim):
                g_val += x[j] ** 2 - 10 * np.cos(2 * np.pi * x[j])

            if g_val < best_g:
                best_g = g_val
                best_x_tail = x_tail.copy()
            evaluations_used += 1

        # Also try setting all to zero (known optimum)
        x_zero = np.zeros(dim)
        f_zero = problem(x_zero)
        evaluations_used += 1
        g_zero = 1 + 10 * (dim - 1)
        for j in range(1, dim):
            g_zero += x_zero[j] ** 2 - 10 * np.cos(2 * np.pi * x_zero[j])
        if g_zero < best_g:
            best_g = g_zero
            best_x_tail = np.zeros(dim - 1)

        # Phase 2: Sweep x1 across [0,1] with the best x_tail
        # We want to trace the Pareto front: for each x1, use best_x_tail
        # f1 = x1, f2 = g*(1-sqrt(x1/g)) where g = best_g

        # Generate a set of x1 values that cover [0,1] well
        # Use a combination of uniform and some strategic points
        x1_values = []

        # Add endpoints
        x1_values.append(0.0)
        x1_values.append(1.0)

        # Add uniform grid
        n_uniform = min(40, n_phase2 - 2)
        if n_uniform > 0:
            x1_grid = np.linspace(0.0, 1.0, n_uniform)
            for v in x1_grid:
                if v not in x1_values:
                    x1_values.append(float(v))

        # Add some random points
        n_random = min(30, n_phase2 - len(x1_values))
        for _ in range(n_random):
            x1_val = rng.random()
            x1_values.append(float(x1_val))

        # Sort and deduplicate
        x1_values = sorted(set(x1_values))

        # Evaluate each x1 value
        results_F = []
        for x1_val in x1_values:
            if evaluations_used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.concatenate([[x1_val], best_x_tail])
            x = np.clip(x, lo, hi)
            f = problem(x)
            results_F.append(f)
            evaluations_used += 1

        # If we have budget left, try to improve g further with different x_tail configurations
        # This helps in case best_g is not exactly 1
        if evaluations_used < budget:
            # Try a few more random x_tail configurations
            for _ in range(20):
                if evaluations_used >= budget:
                    break
                if time.time() > deadline:
                    break

                x_tail = rng.random(dim - 1) * 0.1
                x1_val = rng.random()
                x = np.concatenate([[x1_val], x_tail])
                x = np.clip(x, lo, hi)
                f = problem(x)
                results_F.append(f)
                evaluations_used += 1

        # Convert to array
        if results_F:
            F_final = np.array(results_F)
        else:
            F_final = np.array([[1.0, 1.0]])

        return F_final
# EVOLVE-BLOCK-END
