"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Multi-start Adaptive Simulated Annealing (ASA) for BBOB f19 (Griewank-Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Phase 1: Broad exploration with multiple restarts
        # Use 6 restarts to cover more of the search space
        n_restarts = 6
        # Give more budget to later restarts for refinement
        # Budget allocation: roughly 15%, 15%, 15%, 15%, 15%, 25%
        budget_alloc = [int(budget * f) for f in [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]]
        # Ensure sum matches budget
        diff = budget - sum(budget_alloc)
        budget_alloc[-1] += diff

        for restart in range(n_restarts):
            b = budget_alloc[restart]
            if b < 5 or calls_used >= budget:
                break

            # Initialize starting point
            if restart == 0:
                # Start near origin
                x0 = np.zeros(dim)
            else:
                # Random restarts with increasing spread
                spread = 2.0 + 0.5 * restart
                x0 = rng.uniform(-spread, spread, dim)

            x = np.clip(x0, -5, 5)
            fx = func(x)
            calls_used += 1
            best_local_x = x.copy()
            best_local_f = fx

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # ASA parameters - use geometric cooling
            T0 = 2.5
            T_end = 0.005
            alpha = (T_end / T0) ** (1.0 / max(b - 1, 1))
            T = T0

            for t in range(b - 1):
                if calls_used >= budget:
                    break

                # Classic ASA: x_new = x + T * (u * |v|)
                # where u is uniform random direction, v is standard normal
                u = rng.standard_normal(dim)
                u = u / (np.linalg.norm(u) + 1e-10)
                v = abs(rng.standard_normal(dim))
                perturbation = u * v
                x_new = np.clip(x + T * perturbation, -5, 5)

                fx_new = func(x_new)
                calls_used += 1

                # Metropolis acceptance
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_local_f:
                        best_local_f = fx
                        best_local_x = x.copy()
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_local_f:
                                best_local_f = fx
                                best_local_x = x.copy()
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()

                T *= alpha
                T = max(T, T_end * 0.5)

        # Phase 2: Local refinement around best solution
        remaining = budget - calls_used
        if remaining > 3 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            T = 0.05
            alpha = (0.002 / T) ** (1.0 / max(remaining - 1, 1))
            for t in range(remaining - 1):
                if calls_used >= budget:
                    break
                u = rng.standard_normal(dim)
                u = u / (np.linalg.norm(u) + 1e-10)
                v = abs(rng.standard_normal(dim))
                perturbation = u * v
                x_new = np.clip(x + T * perturbation, -5, 5)
                fx_new = func(x_new)
                calls_used += 1
                if fx_new < fx:
                    x = x_new
                    fx = fx_new
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    delta_f = fx_new - fx
                    if delta_f / T < 50:
                        prob = np.exp(-delta_f / T)
                        if rng.random() < prob:
                            x = x_new
                            fx = fx_new
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                T *= alpha
                T = max(T, 0.001)

        return best_x, best_f
# EVOLVE-BLOCK-END
