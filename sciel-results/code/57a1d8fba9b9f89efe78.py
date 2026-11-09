"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Hybrid Nelder-Mead + L-BFGS-B optimizer for Rosenbrock with hidden shifts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        # Safety margin: 300s limit, use 290s
        self.deadline = self.start_time + 290.0
        n = self.dim
        total_budget = self.budget

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")
            
            x = np.clip(x, -5, 5)
            self.calls_made += 1
            val = func(x)
            
            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()
            
            return val

        # Define starting points ordered by likelihood
        # Rosenbrock global min is at (1,1,1,1,1).
        # With shift s in [-4,4], min is at 1+s, so in [-3, 5].
        
        rng = np.random.default_rng(42)
        
        # Priority-ordered starts:
        # 1. Standard Rosenbrock min (most likely if shift is small)
        # 2. Nearby perturbations of (1,1,1,1,1)
        # 3. Random in likely region
        # 4. Random in full box
        
        starts = [np.ones(self.dim)]
        
        # Small perturbations around (1,1,1,1,1) to handle small shifts
        for scale in [0.5, 1.0, 2.0]:
            starts.append(np.ones(self.dim) + rng.normal(0, scale, self.dim))
        
        # Random in likely region
        for _ in range(5):
            starts.append(rng.uniform(-3, 5, self.dim))
        
        # A few in full box for robustness
        for _ in range(3):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Remove duplicates
        unique_starts = []
        for s in starts:
            s = np.clip(s, -5, 5)
            is_dup = False
            for u in unique_starts:
                if np.allclose(s, u, atol=1e-6):
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(s)
        
        # Try each start with a hybrid approach:
        # 1. Quick Nelder-Mead for local basin identification (cheap, robust)
        # 2. L-BFGS-B for high-precision convergence (fast once in basin)
        
        for x0 in unique_starts:
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break
            
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 15:
                continue
            
            x0 = np.clip(x0, -5, 5)
            
            # Phase 1: Nelder-Mead for quick local refinement
            # Use a small budget for NM to avoid wasting calls if it's not converging
            nm_budget = min(30, remaining_budget // 2)
            
            try:
                # Nelder-Mead is derivative-free and handles the valley well initially
                # maxiter is limited to prevent infinite loops
                nm_res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'adaptive': True
                    }
                )
                
                # If NM improved the solution, use it as start for L-BFGS-B
                x_start_lbfgs = np.clip(nm_res.x, -5, 5) if nm_res.x is not None else x0
                
            except StopIteration:
                break
            except Exception:
                x_start_lbfgs = x0
            
            if self.calls_made >= self.budget or time.time() > self.deadline:
                break
            
            # Phase 2: L-BFGS-B for high-precision convergence
            # L-BFGS-B is very efficient for smooth functions like Rosenbrock once in the basin
            remaining_budget = self.budget - self.calls_made
            if remaining_budget < 10:
                continue
            
            try:
                lbfgs_res = scipy.optimize.minimize(
                    wrapped_func, 
                    x_start_lbfgs, 
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={
                        'maxiter': 100,
                        'ftol': 1e-12,
                        'gtol': 1e-10
                    }
                )
                
                if lbfgs_res.fun < self.best_f:
                    self.best_f = lbfgs_res.fun
                    self.best_x = np.clip(lbfgs_res.x, -5, 5)
                    
            except StopIteration:
                break
            except Exception:
                continue
        
        # Final high-precision polish if we have budget and time
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            remaining = self.budget - self.calls_made
            if remaining > 5:
                try:
                    final_res = scipy.optimize.minimize(
                        wrapped_func, 
                        self.best_x, 
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={
                            'maxiter': 50,
                            'ftol': 1e-15,
                            'gtol': 1e-12
                        }
                    )
                    if final_res.fun < self.best_f:
                        self.best_f = final_res.fun
                        self.best_x = np.clip(final_res.x, -5, 5)
                except StopIteration:
                    pass
                except Exception:
                    pass

        # If no calls were made or best_x is still None
        if self.best_x is None:
            x0 = np.ones(self.dim)
            self.best_x = x0
            self.best_f = func(np.clip(x0, -5, 5))

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
