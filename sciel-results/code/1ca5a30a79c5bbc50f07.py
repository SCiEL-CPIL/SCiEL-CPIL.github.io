"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f9 (Rosenbrock rotated)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        best_x = None
        best_f = float('inf')
        call_count = 0

        def safe_eval(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)
        
        # Generate diverse initial points
        # 1. Origin
        # 2. Random points
        # 3. Points along axes
        starts = [np.zeros(dim)]
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, dim))
        for i in range(dim):
            x = np.zeros(dim)
            x[i] = 3.0
            starts.append(x)
            
        # Shuffle starts to mix types
        rng.shuffle(starts)
        
        # Run Nelder-Mead from each start
        for x0 in starts:
            if not time_ok() or call_count >= budget:
                break
            self._nelder_mead(safe_eval, x0, budget, dim, deadline)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f

    def _nelder_mead(self, safe_eval, x0, budget, dim, deadline):
        """
        Nelder-Mead with large initial simplex.
        """
        # Initialize simplex
        simplex = []
        f0 = safe_eval(x0)
        if f0 is None:
            return
        simplex.append((f0, x0.copy()))
        
        # Large initial step to catch the valley
        step = 2.0 
        for i in range(dim):
            xi = x0.copy()
            xi[i] += step
            fi = safe_eval(xi)
            if fi is None:
                break
            simplex.append((fi, xi.copy()))
        
        if len(simplex) < dim + 1:
            return
            
        # NM Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma_nm = 0.5
        
        max_iter = 100 # Limit iterations per start to allow multiple starts
        
        for _ in range(max_iter):
            if time.time() >= deadline:
                break
            if safe_eval.__closure__[0].cell_contents >= budget: # Check budget via closure hack? No, better to pass count or check outside.
                 # Actually, safe_eval returns None if budget exceeded.
                 # We can just check if the last eval was None or use a counter.
                 # Let's use a local counter or rely on safe_eval returning None.
                 pass

            simplex.sort(key=lambda p: p[0])
            x_worst = simplex[-1][1]
            x_best = simplex[0][1]
            f_worst = simplex[-1][0]
            f_best = simplex[0][0]
            
            if abs(f_worst - f_best) < 1e-10:
                break
            
            centroid = np.mean([p[1] for p in simplex[:-1]], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - x_worst)
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = safe_eval(x_r)
            if f_r is None:
                break
                
            if f_r < f_best:
                x_e = centroid + gamma * (centroid - x_worst)
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = safe_eval(x_e)
                if f_e is None:
                    break
                if f_e < f_r:
                    simplex[-1] = (f_e, x_e)
                else:
                    simplex[-1] = (f_r, x_r)
            elif f_r < f_worst:
                simplex[-1] = (f_r, x_r)
            else:
                x_c = centroid + rho * (x_worst - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = safe_eval(x_c)
                if f_c is None:
                    break
                if f_c < f_worst:
                    simplex[-1] = (f_c, x_c)
                else:
                    for i in range(1, len(simplex)):
                        simplex[i][1] = x_best + sigma_nm * (simplex[i][1] - x_best)
                        simplex[i][1] = np.clip(simplex[i][1], -5.0, 5.0)
                        fi = safe_eval(simplex[i][1])
                        if fi is None:
                            return
                        simplex[i][0] = fi

# EVOLVE-BLOCK-END
