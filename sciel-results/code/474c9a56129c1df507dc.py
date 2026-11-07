# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f7 (Step ellipsoidal).
    
    Uses adaptive coordinate-wise search and Nelder-Mead refinement with strict budget management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

        def count_eval(x):
            if self._call_count >= budget:
                return np.inf
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        def time_ok():
            return time.time() < deadline and self._call_count < budget

        # Phase 1: Random sampling for fast initial progress
        n_random = min(20, budget // 5)
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            count_eval(x)

        # Evaluate origin
        if time_ok():
            count_eval(np.zeros(dim))

        # Phase 2: Adaptive Coordinate-wise Search
        # This is the main workhorse. It iteratively moves along axes.
        
        def coordinate_search(x_center, max_iter=100):
            """Perform coordinate-wise search from x_center."""
            x = x_center.copy()
            f = count_eval(x)
            if f == np.inf:
                return x, f
            
            # Adaptive step sizes per coordinate
            steps = np.ones(dim) * 0.5
            no_improve_count = 0
            
            for _ in range(max_iter):
                if not time_ok():
                    break
                
                improved = False
                for d in range(dim):
                    if not time_ok():
                        break
                    
                    # Try positive and negative steps
                    for sign in [1, -1]:
                        if not time_ok():
                            break
                        x_try = x.copy()
                        x_try[d] += sign * steps[d]
                        x_try = np.clip(x_try, -5, 5)
                        
                        if np.allclose(x_try, x):
                            continue
                        
                        f_try = count_eval(x_try)
                        
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            improved = True
                        else:
                            # Shrink step if no improvement
                            steps[d] *= 0.7
                            if steps[d] < 1e-4:
                                steps[d] = 1e-4
                
                if not improved:
                    no_improve_count += 1
                    # If stuck for a while, try to expand steps slightly to escape local flat areas
                    if no_improve_count % 10 == 0:
                        steps = np.maximum(steps, 0.2)
                    if no_improve_count > 50:
                        break
                else:
                    no_improve_count = 0
            
            return x, f

        # Phase 3: Nelder-Mead local refinement
        def nelder_mead(x0, max_iter=50):
            """Simple Nelder-Mead implementation with strict budget checks."""
            dim = self.dim
            simplex = [x0.copy()]
            f0 = count_eval(x0)
            if f0 == np.inf:
                return x0, f0
            
            # Initialize simplex
            simplex.append(x0 + 0.1 * np.ones(dim))
            for d in range(1, dim):
                x_new = x0.copy()
                x_new[d] += 0.1
                simplex.append(x_new)
            
            fvals = [f0]
            for i in range(1, len(simplex)):
                if not time_ok():
                    return x0, f0
                fvals.append(count_eval(simplex[i]))
            
            alpha, gamma, rho = 1.0, 2.0, 0.5
            
            for _ in range(max_iter):
                if not time_ok():
                    break
                
                # Sort by function value
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                if not time_ok():
                    break
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = count_eval(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
                    if not time_ok():
                        break
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = count_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if not time_ok():
                        break
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = count_eval(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            if not time_ok():
                                break
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            fvals[i] = count_eval(simplex[i])
                        if not time_ok():
                            break
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
            
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 4: Main search loop
        best_x = self._best_x.copy() if self._best_x is not None else np.zeros(dim)
        
        while time_ok():
            # Check if we have enough budget for a meaningful round
            if self._call_count > budget - 10:
                break

            # Alternate between Coordinate Search and Nelder-Mead
            # Use a simple counter to alternate
            # We'll just do CS, then NM, then CS...
            # To be safe, we limit iterations per call to these functions
            x_cs, f_cs = coordinate_search(best_x, max_iter=20)
            
            if f_cs < self._best_f - 1e-12:
                best_x = x_cs
            
            if time_ok() and self._call_count < budget - 10:
                x_nm, f_nm = nelder_mead(best_x, max_iter=20)
                if f_nm < self._best_f - 1e-12:
                    best_x = x_nm

            # Random restart if stuck?
            # For BBOB f7, random restarts are less effective than local search due to the specific structure.
            # We'll stick to local search refinement as it's more efficient for the budget.
            
            # If we are very close to the end, just do simple local steps
            if self._call_count > budget - 5:
                break

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
