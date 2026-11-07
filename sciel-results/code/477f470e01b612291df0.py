"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start Nelder-Mead optimizer with adaptive restarts for f24 (Lunacek bi-Rastrigin).
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin

        best_x = None
        best_f = np.inf
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return func(x)

        def nelder_mead(x0, max_evals, func_wrap):
            """
            Robust Nelder-Mead with adaptive restarts.
            """
            dim = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                # Small initial step
                step = 0.5 if abs(xi[i]) < 4.5 else -0.5
                xi[i] += step
                simplex.append(xi)

            fs = [func_wrap(s) for s in simplex]
            evals = dim + 1

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_nm = 0.5

            best_f_local = min(fs)
            best_x_local = simplex[np.argmin(fs)].copy()
            no_improve_count = 0
            max_no_improve = 20  # Restart if no improvement for 20 iterations

            while evals < max_evals:
                # Sort
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Check for stagnation
                if fs[0] < best_f_local - 1e-10:
                    best_f_local = fs[0]
                    best_x_local = simplex[0].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > max_no_improve:
                    # Adaptive reset: Break to let outer loop restart
                    break

                # Centroid
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func_wrap(x_r)
                evals += 1

                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func_wrap(x_e)
                    evals += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func_wrap(x_c)
                    evals += 1
                    if f_c < min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fs[i] = func_wrap(simplex[i])
                            evals += 1
                            if evals >= max_evals:
                                break

                if evals >= max_evals:
                    break

            return best_x_local, best_f_local

        # Main loop: Multi-start NM
        rng = np.random.default_rng(42)

        # 1. Initial random search to find a few good starting points
        # Use 10% of budget for this
        init_budget = int(budget * 0.1)
        candidates = []
        for i in range(init_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = counted_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])
        
        # Take top few unique points (simple distance check)
        starts = []
        for fx, x in candidates:
            if not starts:
                starts.append(x)
            else:
                # Check distance to existing starts
                if all(np.linalg.norm(x - s) > 1.0 for s in starts):
                    starts.append(x)
            if len(starts) >= 5:
                break

        # If we didn't get enough diverse starts, add random ones
        while len(starts) < 5:
            starts.append(rng.uniform(-5, 5, dim))

        # 2. Run NM from these starts
        # Allocate remaining budget
        remaining_budget = budget - calls_made[0]
        num_restarts = min(len(starts), 5)
        # Distribute budget among restarts
        evals_per_restart = max(dim + 2, int(remaining_budget / num_restarts))
        
        for i in range(num_restarts):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline:
                break
            
            x0 = starts[i]
            # Cap the evals for this restart to not exceed budget
            current_evals = calls_made[0]
            max_evals_this = min(evals_per_restart, budget - current_evals)
            
            if max_evals_this < dim + 2:
                continue

            try:
                x_res, f_res = nelder_mead(x0, max_evals_this, counted_func)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res.copy()
            except (TimeoutError, Exception):
                pass

        # 3. Final polish on the global best with remaining budget
        if calls_made[0] < budget and best_x is not None:
            remaining = budget - calls_made[0]
            if remaining > dim + 2:
                try:
                    x_res, f_res = nelder_mead(best_x, remaining, counted_func)
                    if f_res < best_f:
                        best_f = f_res
                        best_x = x_res.copy()
                except (TimeoutError, Exception):
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
