"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Powell's method optimizer for the Sphere function.
    Adapts search directions to the rotated principal axes for rapid convergence."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        calls = [0]
        best_x = np.zeros(dim)
        best_fx = float('inf')
        rng = np.random.default_rng(42)

        def guarded(x):
            if calls[0] >= budget:
                return best_fx
            v = func(np.asarray(x, dtype=float))
            calls[0] += 1
            return v

        def record(x, fx):
            nonlocal best_x, best_fx
            if fx < best_fx:
                best_x, best_fx = np.asarray(x, dtype=float).copy(), fx

        # 1D minimization using golden section search
        def minimize_1d(x, d, lo, hi):
            f0 = guarded(x)
            if calls[0] >= budget: 
                return x, f0
            
            invphi = (np.sqrt(5) - 1) / 2
            invphi2 = invphi * invphi
            
            t1 = lo + invphi2 * (hi - lo)
            t2 = hi - invphi2 * (hi - lo)
            
            f1 = guarded(x + t1 * d)
            f2 = guarded(x + t2 * d)
            
            if calls[0] >= budget: 
                return x, f0
            
            for _ in range(15): # Reduced to 15 to save budget
                if time.time() > deadline: break
                if f1 < f2:
                    hi = t2
                    t2 = t1
                    f2 = f1
                    t1 = lo + invphi2 * (hi - lo)
                    f1 = guarded(x + t1 * d)
                else:
                    lo = t1
                    t1 = t2
                    f1 = f2
                    t2 = hi - invphi2 * (hi - lo)
                    f2 = guarded(x + t2 * d)
                    
                if calls[0] >= budget: break
                if abs(hi - lo) < 1e-5:
                    break
            
            t_best = (lo + hi) / 2
            x_best = x + t_best * d
            f_best = guarded(x_best)
            
            return x_best, f_best

        # Initial exploration
        candidates = [np.zeros(dim)]
        for _ in range(4):
            candidates.append(rng.uniform(-4, 4, dim))
        
        best_start_x = None
        best_start_f = float('inf')
        for c in candidates:
            fc = guarded(c)
            if fc < best_start_f:
                best_start_f = fc
                best_start_x = c.copy()
        
        if best_start_x is None:
             best_start_x = np.zeros(dim)
             best_start_f = guarded(best_start_x)
             
        record(best_start_x, best_start_f)
        
        x = best_start_x
        fx = best_start_f
        
        # Directions: Coordinate axes
        directions = [np.eye(dim)[i] for i in range(dim)]
        
        # Powell's Main Loop
        while calls[0] < budget and time.time() < deadline:
            x0 = x.copy()
            f0 = fx
            
            x_prev = x.copy()
            f_prev = fx
            improvements = []
            
            for i in range(dim):
                if calls[0] >= budget or time.time() > deadline:
                    break
                x_new, f_new = minimize_1d(x_prev, directions[i], -15, 15)
                record(x_new, f_new)
                improvement = f_prev - f_new
                improvements.append(improvement)
                x_prev = x_new
                f_prev = f_new
            
            if calls[0] >= budget or time.time() > deadline:
                break
                
            x = x_prev
            fx = f_prev
            
            if abs(f0 - fx) < 1e-10:
                break
            
            # Find the index of the worst improvement
            worst_idx = np.argmax(improvements)
            
            # Update direction
            new_dir = x - x0
            if np.linalg.norm(new_dir) > 1e-10:
                directions[worst_idx] = new_dir / np.linalg.norm(new_dir)
            else:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
