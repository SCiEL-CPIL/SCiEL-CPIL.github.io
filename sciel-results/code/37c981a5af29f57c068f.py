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

        # Phase 1: Diverse Initialization
        # We use the first few calls to probe the landscape and find a good starting basin.
        # This acts as a lightweight "multi-start" phase.
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 1.0,
            np.ones(dim) * -1.0,
            rng.uniform(-3.0, 3.0, dim),
            rng.uniform(-4.0, 4.0, dim)
        ]
        
        # Ensure we have enough budget for initialization
        n_init = min(len(init_points), max(1, budget // 10))
        
        for i in range(n_init):
            x = np.clip(init_points[i], -5, 5)
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if calls_used >= budget:
                break

        if best_x is None:
            # Fallback if something went wrong (shouldn't happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_used += 1

        # Phase 2: Adaptive Simulated Annealing (ASA)
        remaining = budget - calls_used
        if remaining <= 0:
            return best_x, best_f

        # Current state
        x = best_x.copy()
        fx = best_f

        # Temperature parameters
        T0 = 1.5  # Initial temperature, allows steps of ~1.5 units
        T_end = 0.005
        alpha = (T_end / T0) ** (1.0 / max(remaining - 1, 1))
        T = T0

        # Stagnation tracking
        steps_since_improvement = 0
        stagnation_threshold = 40  # Reheat if no global improvement for 40 steps
        reheat_T = 0.8

        for t in range(remaining - 1):
            if calls_used >= budget:
                break

            # Generate ASA perturbation: u * |v|
            # u is random unit vector, v is vector of standard normals
            u = rng.standard_normal(dim)
            norm_u = np.linalg.norm(u)
            if norm_u > 1e-10:
                u = u / norm_u
            v = np.abs(rng.standard_normal(dim))
            perturbation = u * v
            
            # Step size is controlled by temperature T
            # In standard ASA, the step is T * perturbation
            # However, to ensure we cover the space well, we scale T
            x_new = np.clip(x + T * perturbation, -5, 5)
            
            fx_new = func(x_new)
            calls_used += 1

            # Metropolis Acceptance Criterion
            accept = False
            if fx_new < fx:
                accept = True
            else:
                delta_f = fx_new - fx
                # Avoid overflow in exp
                if delta_f / T < 50:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        accept = True

            if accept:
                x = x_new
                fx = fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
            else:
                steps_since_improvement += 1

            # Stagnation Handling: Reheat if stuck
            if steps_since_improvement >= stagnation_threshold:
                T = max(T, reheat_T)
                steps_since_improvement = 0

            # Cool down
            T *= alpha
            T = max(T, T_end * 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
