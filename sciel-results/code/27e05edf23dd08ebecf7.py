# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-strategy optimizer for BBOB f7 (Step ellipsoidal).
    
    Combines:
    1. Random sampling for fast initial progress
    2. Adaptive coordinate-wise search exploiting step-ellipsoidal structure
    3. Local Nelder-Mead refinement
    4. Global random restarts with memory
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
            self._call_count += 1
            val = func(x)
            if val < self._best_f:
                self._best_f = val
                self._best_x = x.copy()
            return val

        def time_ok():
            return time.time() < deadline and self._call_count < budget

        # Phase 1: Random sampling for fast initial progress (first ~50 calls)
        n_random = min(50, budget)
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            count_eval(x)

        # Also evaluate origin and some structured points
        structured_starts = [
            np.zeros(dim),
            np.full(dim, 1.0),
            np.full(dim, -1.0),
            np.full(dim, 0.5),
            np.full(dim, -0.5),
        ]
        for x in structured_starts:
            if not time_ok():
                break
            count_eval(x)

        # Phase 2: Adaptive coordinate-wise search
        # For step-ellipsoidal functions, the function is separable in a rotated
        # coordinate system with different scales per axis. We can exploit this
        # by doing coordinate-wise line searches.
        
        def coordinate_search(x_center, max_iter=200):
            """Perform coordinate-wise search from x_center."""
            x = x_center.copy()
            f = self._best_f if np.allclose(x, self._best_x) else count_eval(x)
            
            # Adaptive step sizes per coordinate
            steps = np.ones(dim) * 1.0
            improvements = 0
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
                        x_try = x.copy()
                        x_try[d] += sign * steps[d]
                        x_try = np.clip(x_try, -5, 5)
                        
                        if np.allclose(x_try, x):
                            continue
                        
                        f_try = count_eval(x_try)
                        
                        if f_try < f - 1e-12:
                            x = x_try
                            f = f_try
                            improvements += 1
                            no_improve_count = 0
                            improved = True
                        else:
                            # Shrink step if no improvement
                            steps[d] *= 0.5
                            if steps[d] < 1e-6:
                                steps[d] = 1e-6
                
                if not improved:
                    no_improve_count += 1
                    # Try larger steps to escape local minimum
                    if no_improve_count % 5 == 0:
                        steps = np.maximum(steps, 0.5)
                    if no_improve_count > 20:
                        break
                else:
                    # Gradually increase steps for coordinates that are improving
                    pass
            
            return x, f

        # Phase 3: Nelder-Mead local refinement
        def nelder_mead(x0, max_iter=100):
            """Simple Nelder-Mead implementation."""
            dim = self.dim
            # Initialize simplex
            simplex = [x0.copy()]
            f0 = count_eval(x0)
            simplex.append(x0 + 0.05 * np.ones(dim))
            for d in range(1, dim):
                x_new = x0.copy()
                x_new[d] += 0.05
                simplex.append(x_new)
            
            fvals = [count_eval(x) if i > 0 else f0 for i, x in enumerate(simplex)]
            # Actually we already evaluated x0, so adjust
            fvals[0] = f0
            
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
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = count_eval(x_r)
                
                if f_r < fvals[0]:
                    # Expansion
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
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = count_eval(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            if time_ok():
                                fvals[i] = count_eval(simplex[i])
                            else:
                                break
                        if not time_ok():
                            break
                
                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break
            
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Phase 4: Main search loop with alternating strategies
        best_x = self._best_x.copy() if self._best_x is not None else np.zeros(dim)
        
        while time_ok():
            # Try coordinate search from current best
            x_cs, f_cs = coordinate_search(best_x, max_iter=50)
            
            if f_cs < self._best_f - 1e-12:
                best_x = x_cs
            
            # Try Nelder-Mead from current best
            x_nm, f_nm = nelder_mead(best_x, max_iter=30)
            
            if f_nm < self._best_f - 1e-12:
                best_x = x_nm
            elif self._best_f < f_nm:
                # Keep previous best
                pass
            
            # Occasionally do a random restart to escape local minima
            if self._call_count < budget - 20 and rng.random() < 0.1:
                # Perturb from best
                x_restart = best_x + rng.normal(0, 0.5, dim)
                x_restart = np.clip(x_restart, -5, 5)
                f_restart = count_eval(x_restart)
                
                if f_restart < self._best_f - 1e-12:
                    best_x = x_restart
                    # Follow up with local search
                    x_cs, f_cs = coordinate_search(best_x, max_iter=30)
                    if f_cs < self._best_f - 1e-12:
                        best_x = x_cs

        return self._best_x, self._best_f
# EVOLVE-BLOCK-END
