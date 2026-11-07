"""
Optimized search for BBOB f13 Sharp Ridge using Nelder-Mead with adaptive restarts.
"""
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with adaptive restarts for BBOB f13 (Sharp Ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: evaluator kills at 330s, limit is 300s.
        # We set our internal deadline to 280s to be safe.
        deadline = time.time() + 280.0

        rng = np.random.default_rng(42)
        calls = [0]

        def safe_call(x):
            if calls[0] >= budget:
                return float('inf')
            # Clip to box [-5, 5]
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                val = float(func(x_clipped))
                calls[0] += 1
                return val
            except Exception:
                calls[0] += 1
                return float('inf')

        best_x = np.zeros(dim)
        best_f = float('inf')

        # Initial exploration: Random point to ensure we are in the domain
        # f13 is usually well-behaved, but a random start helps avoid symmetry issues
        start_x = rng.uniform(-4.0, 4.0, dim)
        best_f = safe_call(start_x)
        best_x = start_x.copy()

        # Nelder-Mead Parameters
        # Standard NM parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage

        # Max iterations per NM run
        # In 5D, NM converges fast. 100 iterations is ~500 evaluations if no restarts,
        # but we will restart, so we limit iterations per run.
        max_iter_per_run = 50
        # Stagnation threshold: if no improvement in this many iterations, restart
        stagnation_limit = 20

        while calls[0] < budget and time.time() < deadline - 5.0:
            # Initialize Simplex around the current best point
            # The initial simplex size should be small enough to be local,
            # but large enough to capture the ridge if we are off it.
            # Since we restart from best_x, we use a small initial step.
            # However, if we are far from the optimum, a larger step might be needed.
            # A heuristic: start with step 1.0, reduce on restarts if stuck.
            initial_step = 1.0 if calls[0] < 50 else 0.5
            
            simplex = []
            simplex_f = []
            
            # Vertex 0: Best point
            simplex.append(best_x.copy())
            simplex_f.append(best_f)

            # Other vertices: Perturb best_x in each dimension
            for i in range(dim):
                v = best_x.copy()
                # Use a deterministic perturbation based on dimension index and rng
                # to ensure coverage
                delta = initial_step
                v[i] += delta
                v = np.clip(v, -5.0, 5.0)
                f_v = safe_call(v)
                simplex.append(v)
                simplex_f.append(f_v)
                
                # Update global best
                if f_v < best_f:
                    best_f = f_v
                    best_x = v.copy()

            # Check if we ran out of budget during initialization
            if calls[0] >= budget:
                break

            # Nelder-Mead Loop
            improved_in_run = False
            last_best_f = best_f
            
            for _ in range(max_iter_per_run):
                if calls[0] >= budget or time.time() > deadline - 5.0:
                    break

                # Sort simplex by fitness
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    improved_in_run = True

                # Check stagnation
                if best_f - last_best_f < 1e-10:
                    # No improvement in this iteration
                    pass # Continue to check for restart condition outside loop or count iterations
                
                # Identify worst and second worst
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_second_worst = simplex[-2]
                
                # Centroid of all points except the worst
                x_centroid = np.mean(simplex[:-1], axis=0)

                # 1. Reflection
                x_reflect = x_centroid + alpha * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = safe_call(x_reflect)

                if f_reflect < simplex_f[0]:
                    # Reflection is better than best
                    # 2. Expansion
                    x_expand = x_centroid + gamma * (x_reflect - x_centroid)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = safe_call(x_expand)
                    
                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        simplex_f[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        simplex_f[-1] = f_reflect
                elif f_reflect < f_worst:
                    # Reflection is between best and worst
                    simplex[-1] = x_reflect
                    simplex_f[-1] = f_reflect
                else:
                    # Reflection is worse than worst
                    # 3. Contraction
                    x_contract = x_centroid + rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = safe_call(x_contract)
                    
                    if f_contract < f_worst:
                        simplex[-1] = x_contract
                        simplex_f[-1] = f_contract
                    else:
                        # 4. Shrinkage
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = safe_call(simplex[i])
                        
                        if calls[0] >= budget:
                            break

                # Check if simplex has collapsed or converged
                # If the spread of the simplex is very small, we are likely at a local minimum
                spread = np.std(simplex, axis=0)
                if np.max(spread) < 1e-6:
                    break

            # Restart Logic
            # If we didn't improve significantly in this run, or if the simplex collapsed,
            # restart from the best point found.
            # To escape local minima, we can add a small random perturbation to the best point
            # for the next run's initialization, or simply rely on the "shrinkage" having
            # converged to a local min and restarting with a fresh simplex shape.
            
            # If no improvement in the last run, we might be stuck.
            # The NM simplex itself adapts. If it collapsed, we restart.
            # We loop back to the top of the while loop.
            # The `initial_step` will be reduced in subsequent iterations if we've used many calls.
            
            # If we are very close to the end of the budget, stop restarting to save calls for final polish?
            # NM is self-polishing. Just let it run until budget/time is out.

        return best_x, best_f
