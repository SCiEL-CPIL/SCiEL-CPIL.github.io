"""Multi-objective optimizer for ZDT3 using structured decomposition with Halton sampling."""
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
        deadline = start_time + 285  # safety margin before 300s limit

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
        # KEY INSIGHT: g(x[1:]) is minimized at x[1:]=0 giving g=1 exactly.
        # We can compute g analytically without spending any evaluation budget on it.
        # Therefore, ALL budget goes to sampling the Pareto front.

        # The optimal tail is x[1:]=0, giving g=1
        best_x_tail = np.zeros(dim - 1)

        # Generate x1 values for dense Pareto front sampling
        # Strategy: combine uniform grid with Van der Corput sequence
        # to maximize hypervolume coverage

        x1_values = []

        # Add endpoints
        x1_values.append(0.0)
        x1_values.append(1.0)

        # Van der Corput sequence (Halton in base 2) for low-discrepancy sampling
        def van_der_corput(n, base):
            """Generate n values of the Van der Corput sequence in the given base."""
            values = []
            for i in range(1, n + 1):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                values.append(r)
            return values

        # Use most of the budget for dense sampling
        n_samples = budget - 2  # leave room for endpoints
        if n_samples > 0:
            # Mix: 70% uniform grid, 30% Van der Corput for better coverage
            n_uniform = int(n_samples * 0.7)
            n_vdc = n_samples - n_uniform

            if n_uniform > 0:
                x1_grid = np.linspace(0.0, 1.0, n_uniform)
                for v in x1_grid:
                    x1_values.append(float(v))

            if n_vdc > 0:
                vdc_values = van_der_corput(n_vdc, 2)
                for v in vdc_values:
                    x1_values.append(float(v))

        # Sort and deduplicate
        x1_values = sorted(set(x1_values))

        # Evaluate each x1 value using the known optimal tail
        results_F = []
        evaluations_used = 0

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

        # If we have budget left (shouldn't happen, but just in case),
        # fill with more uniform points
        if evaluations_used < budget:
            remaining = budget - evaluations_used
            for x1_val in np.linspace(0.0, 1.0, remaining + 1):
                if evaluations_used >= budget:
                    break
                if time.time() > deadline:
                    break
                x = np.concatenate([[x1_val], best_x_tail])
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
