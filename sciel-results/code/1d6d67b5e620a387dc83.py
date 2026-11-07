import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, calls, max_iters=50, step0=1.0):
        """
        Efficient Nelder-Mead implementation.
        Returns best_x, best_f.
        """
        d = self.dim
        x = np.array(x0, dtype=float)
        f0 = func(x)
        
        # Initialize simplex
        simplex = [x.copy()]
        fs = [f0]
        for i in range(d):
            xi = x.copy()
            xi[i] += step0
            simplex.append(xi)
            fs.append(func(xi))
            
        best_x = x.copy()
        best_f = f0
        
        for _ in range(max_iters):
            if calls[0] >= self.budget:
                break
                
            # Sort simplex
            order = np.argsort(fs)
            simplex = [simplex[i] for i in order]
            fs = [fs[i] for i in order]
            
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = simplex[0].copy()
                
            # Check convergence
            if fs[-1] - fs[0] < 1e-8:
                break
                
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + (centroid - simplex[-1])
            fr = func(xr)
            
            if fr < fs[0]:
                # Extension
                xe = centroid + 2.0 * (centroid - simplex[-1])
                fe = func(xe)
                if fe < fr:
                    simplex[-1] = xe
                    fs[-1] = fe
                else:
                    simplex[-1] = xr
                    fs[-1] = fr
                continue
            elif fr < fs[-2]:
                # Accept reflection
                simplex[-1] = xr
                fs[-1] = fr
                continue
            else:
                # Contraction
                xc = centroid + 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fs[-1]:
                    simplex[-1] = xc
                    fs[-1] = fc
                    continue
                else:
                    # Shrink
                    for i in range(1, d + 1):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        fs[i] = func(simplex[i])
                        
        return best_x, best_f

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 275.0
        dim = self.dim
        calls = [0]
        budget = self.budget
        
        def safe_func(x):
            if calls[0] >= budget:
                return np.inf
            calls[0] += 1
            try:
                return float(func(np.asarray(x, dtype=float)))
            except Exception:
                return np.inf

        best_x = np.zeros(dim)
        best_f = np.inf

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)

        rng = np.random.default_rng(42)
        
        # Phase 1: Broad Exploration
        # Use a mix of random points and a coarse grid to hit the lattice valleys
        n_explore = 100
        points = []
        # Coarse grid points (step 2.0 to align with Rastrigin periods)
        # Rastrigin minima are at integers. Rotated, they are anywhere.
        # But sampling on a grid with step 1.0 or 2.0 helps.
        # Let's do 50 random points and 50 grid points.
        
        # Grid points: -4, -2, 0, 2, 4
        grid_vals = [-4.0, -2.0, 0.0, 2.0, 4.0]
        grid_points = []
        # We can't do full 5^5=3125 grid. We pick a subset.
        # Let's just use random points for simplicity and robustness, 
        # but bias towards 0 and integers?
        
        # Actually, just random points are fine if we have enough.
        # 100 random points in 5D is sparse, but Nelder-Mead will pull them to valleys.
        for _ in range(n_explore):
            x = rng.uniform(-5.0, 5.0, dim)
            f = safe_func(x)
            points.append((x, f))
            record(x, f)
            
        if time.time() > deadline - 20:
            return best_x, best_f

        # Sort points by fitness
        points.sort(key=lambda t: t[1])
        
        # Phase 2: Local Polish top 5
        top_k = 5
        for i in range(top_k):
            if calls[0] >= budget - 10: # leave some for final polish
                break
            if time.time() > deadline - 30:
                break
            x0, f0 = points[i]
            # Allocate remaining budget / remaining candidates
            remaining_calls = budget - calls[0]
            # Use a chunk of budget for each polish
            # Nelder-Mead uses ~10-20 calls per iteration. 
            # Let's allow up to 50 evaluations per polish.
            max_evals = 50
            # Create a local budget tracker? No, just use global calls.
            # We pass calls list to Nelder-Mead.
            
            # We need to be careful not to exceed budget.
            # Nelder-Mead will stop if calls[0] >= budget.
            
            x_polished, f_polished = self._nelder_mead(safe_func, x0, calls, max_iters=40, step0=1.0)
            record(x_polished, f_polished)
            
        # Phase 3: Final Polish on the absolute best
        if calls[0] < budget - 5 and time.time() < deadline - 10:
            x_final, f_final = self._nelder_mead(safe_func, best_x, calls, max_iters=50, step0=0.5)
            record(x_final, f_final)

        return best_x, best_f
