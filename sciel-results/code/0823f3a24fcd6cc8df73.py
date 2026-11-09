"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive global exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                return np.inf
            calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Phase 1: Stratified global exploration
        n_init = min(40, budget // 10)
        for i in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            # Mix of scales for coverage
            if i < n_init // 5:
                x = rng.uniform(-4.8, 4.8, dim)
            elif i < 2 * n_init // 5:
                x = rng.uniform(-3.5, 3.5, dim)
            elif i < 4 * n_init // 5:
                x = rng.uniform(-2.0, 2.0, dim)
            else:
                x = rng.uniform(-0.5, 0.5, dim)
            evaluate(x)

        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Multi-start Nelder-Mead refinement
        # We use multiple starts from the best found point and a few other good points
        # Nelder-Mead is derivative-free and works well on smooth multimodal functions
        
        def nelder_mead(func, x0, budget_remaining, deadline, rng, dim, global_best_x, global_best_f, calls_counter):
            """Run Nelder-Mead starting from x0, using up to budget_remaining calls."""
            nonlocal calls, best_x, best_f
            
            # Initialize simplex: x0 and dim perturbed points
            simplex = [x0.copy()]
            # Initial step size for simplex expansion
            init_step = 0.5
            for i in range(dim):
                xi = x0.copy()
                xi[i] += init_step
                simplex.append(np.clip(xi, -5, 5))
            
            # Evaluate initial simplex
            f_values = []
            for i, sx in enumerate(simplex):
                if calls >= budget or time.time() > deadline:
                    break
                fx = evaluate(sx)
                f_values.append(fx)
            
            if len(f_values) < dim + 1:
                return

            # Nelder-Mead iterations
            max_iter = budget_remaining
            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                # Reflection
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = evaluate(xr)
                
                if fxr < f_values[0]:
                    # Expansion
                    xe = centroid + 2 * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = evaluate(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        f_values[-1] = fxe
                    else:
                        simplex[-1] = xr
                        f_values[-1] = fxr
                elif fxr < f_values[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    f_values[-1] = fxr
                else:
                    # Contraction
                    if fxr < f_values[-1]:
                        # Outside contraction
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = evaluate(xc)
                    if fxc < min(fxr, f_values[-1]):
                        simplex[-1] = xc
                        f_values[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = evaluate(simplex[i])
                
                # Check for convergence: if simplex is very small, break
                if np.max(f_values) - np.min(f_values) < 1e-8:
                    break

        # Start Nelder-Mead from the best point found in Phase 1
        if best_x is not None:
            x_start = best_x.copy()
        else:
            x_start = rng.uniform(-1, 1, dim)
            evaluate(x_start)

        # Run Nelder-Mead with a portion of the remaining budget
        nm_budget = (budget - calls) // 2
        if nm_budget > 0:
            nelder_mead(func, x_start, nm_budget, deadline, rng, dim, best_x, best_f, calls)

        # Phase 3: Fine-tuning with (1+1)-ES from the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            stagnation = 0
            max_stagnation = 30
            
            while calls < budget and time.time() < deadline:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = evaluate(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    stagnation = 0
                else:
                    sigma *= 0.9
                    stagnation += 1
                
                if sigma < 0.001 or stagnation > max_stagnation:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
