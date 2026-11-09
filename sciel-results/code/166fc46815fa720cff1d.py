import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Time guard: The evaluator kills at 330s, but we want to return well before.
        # Let's aim to finish within 280s to be safe.
        start_time = time.time()
        deadline = start_time + 280
        
        calls = 0
        best_x = None
        best_f = float('inf')
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if calls >= budget:
                return float('inf')
            if time.time() > deadline:
                return float('inf')
            
            # Ensure x is a numpy array of correct shape and within bounds
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            
            calls += 1
            try:
                val = float(func(x))
            except Exception:
                val = float('inf')
            
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Phase 1: Coarse Global Search to locate the basin
        # Bent Cigar is convex but has a narrow valley. We need to find the center of the valley.
        # We'll use a coarse grid/random search to pick the best starting point.
        
        # Spend ~20% of budget on initialization? No, Bent Cigar is convex, so any point in the basin works.
        # But we need to be in the basin. The shift is in [-4, 4].
        # Let's try 10 random points to find a good center.
        
        num_initial = 10
        candidates = []
        for _ in range(num_initial):
            if time.time() > deadline:
                break
            x_init = rng.uniform(-4.5, 4.5, dim)
            f_init = safe_func(x_init)
            candidates.append((x_init, f_init))
            
        if not candidates:
            return np.zeros(dim), float('inf')
            
        candidates.sort(key=lambda x: x[1])
        x0, f0 = candidates[0]
        
        # Phase 2: Powell's Method
        # Powell is excellent for functions with narrow valleys (like Bent Cigar) because
        # it aligns its search directions with the valley.
        # Nelder-Mead can struggle with very narrow valleys if the simplex gets skewed.
        
        # We will use scipy.optimize.minimize with method='Powell'.
        # We need to wrap the function to count calls and respect the deadline.
        
        def powell_func(x):
            return safe_func(x)
            
        # Options for Powell
        # maxiter: maximum number of iterations. Each iteration involves a line search along each dimension.
        # In 5D, one iteration is ~5+ calls. 
        # We have ~490 calls left. 490/5 = ~98 iterations.
        # Let's set maxiter to something reasonable, but rely on budget/deadline.
        
        try:
            res = minimize(
                powell_func, 
                x0, 
                method='Powell',
                bounds=[(-5, 5)] * dim,
                options={
                    'maxiter': 1000,
                    'xatol': 1e-8,
                    'fatol': 1e-12,
                    'disp': False
                }
            )
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception:
            pass
            
        # Phase 3: Refine with Nelder-Mead or CMA-ES if budget remains?
        # Powell should be very effective. If we have time/calls left, we can do a final polish.
        # However, Powell's convergence is usually quite good.
        
        # Let's check if we have calls left and time left.
        if calls < budget and time.time() < deadline - 5:
             # Try a quick Nelder-Mead refinement from the Powell result
             x_nm = best_x.copy()
             f_nm = best_f
             
             # Initialize simplex around x_nm with small delta
             delta = 0.1
             simplex = [x_nm.copy()]
             simplex_f = [f_nm]
             
             for i in range(dim):
                 x_i = x_nm.copy()
                 x_i[i] += delta
                 x_i = np.clip(x_i, -5.0, 5.0)
                 f_i = safe_func(x_i)
                 simplex.append(x_i)
                 simplex_f.append(f_i)
                 
                 if time.time() > deadline or calls >= budget:
                     break
             
             # Simple NM loop for refinement
             rho = 1.0
             chi = 2.0
             gamma = 0.5
             sigma = 0.5
             
             for _ in range(50): # Small cap for refinement
                 if time.time() > deadline or calls >= budget:
                     break
                 
                 indices = np.argsort(simplex_f)
                 simplex = [simplex[i] for i in indices]
                 simplex_f = [simplex_f[i] for i in indices]
                 
                 if simplex_f[-1] - simplex_f[0] < 1e-10:
                     break
                 
                 centroid = np.mean(simplex[:-1], axis=0)
                 
                 xr = centroid + rho * (centroid - simplex[-1])
                 xr = np.clip(xr, -5.0, 5.0)
                 fr = safe_func(xr)
                 
                 if fr < simplex_f[0]:
                     xe = centroid + chi * (xr - centroid)
                     xe = np.clip(xe, -5.0, 5.0)
                     fe = safe_func(xe)
                     if fe < fr:
                         simplex[-1] = xe
                         simplex_f[-1] = fe
                     else:
                         simplex[-1] = xr
                         simplex_f[-1] = fr
                 elif fr < simplex_f[-2]:
                     simplex[-1] = xr
                     simplex_f[-1] = fr
                 else:
                     xc = centroid + gamma * (simplex[-1] - centroid)
                     xc = np.clip(xc, -5.0, 5.0)
                     fc = safe_func(xc)
                     if fc < simplex_f[-1]:
                         simplex[-1] = xc
                         simplex_f[-1] = fc
                     else:
                         for i in range(1, dim + 1):
                             simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                             simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                             simplex_f[i] = safe_func(simplex[i])
                         
                         indices = np.argsort(simplex_f)
                         simplex = [simplex[i] for i in indices]
                         simplex_f = [simplex_f[i] for i in indices]
                         
                         if simplex_f[0] < best_f:
                             best_f = simplex_f[0]
                             best_x = simplex[0].copy()

        return best_x, best_f
