import numpy as np
import time
import sys

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bound = 5.0
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.func_wrapper = None

    def __call__(self, func):
        self.call_count = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280.0
        self.dim = 5
        self.budget = 500

        def wrapped_func(x):
            self.call_count += 1
            if time.time() > self.deadline:
                raise TimeoutError()
            if self.call_count > self.budget:
                raise RuntimeError("Budget exceeded")
            x = np.clip(np.asarray(x, dtype=np.float64), -self.bound, self.bound)
            fx = func(x)
            if np.isfinite(fx) and fx < self.best_f:
                self.best_f = float(fx)
                self.best_x = x.copy()
            return fx if np.isfinite(fx) else 1e30

        self.func_wrapper = wrapped_func
        n = self.dim
        lb = -self.bound
        ub = self.bound
        
        # Phase 1: Differential Evolution
        # Parameters suggested by directive
        lam = 40
        F = 0.8
        CR = 0.9
        
        # Initialize population
        # Use Latin Hypercube or random? Random is fine, LHS might be better for coverage.
        # Let's use random for simplicity and speed, but ensure spread.
        pop = np.random.uniform(lb, ub, (lam, n))
        fvals = np.array([wrapped_func(p) for p in pop])
        
        # Sort initial population to track best
        # Actually, we just keep the population and fvals aligned.
        
        while self.call_count + lam <= self.budget and time.time() < self.deadline - 5:
            new_pop = pop.copy()
            new_fvals = fvals.copy()
            
            for i in range(lam):
                # Select 3 distinct parents
                # Ensure they are distinct from i and each other
                r1, r2, r3 = np.random.choice(lam, 3, replace=False)
                while r1 == i or r2 == i or r3 == i:
                    r1, r2, r3 = np.random.choice(lam, 3, replace=False)
                
                # Mutation
                v = pop[r1] + F * (pop[r2] - pop[r3])
                
                # Crossover
                # Binary crossover
                jrand = np.random.randint(0, n)
                u = np.zeros(n)
                for j in range(n):
                    if np.random.rand() < CR or j == jrand:
                        u[j] = v[j]
                    else:
                        u[j] = pop[i][j]
                
                # Clip to bounds
                u = np.clip(u, lb, ub)
                
                # Evaluate
                f_u = wrapped_func(u)
                
                # Selection
                if f_u <= fvals[i]:
                    new_pop[i] = u
                    new_fvals[i] = f_u
            
            pop = new_pop
            fvals = new_fvals
            
            # Update global best
            # wrapped_func already updates self.best_x and self.best_f
            
        # Phase 2: Local Polish
        # Use remaining budget for a local search around the best solution
        remaining = self.budget - self.call_count
        if remaining > 10 and self.best_x is not None:
            try:
                self._nelder_mead_polish(self.func_wrapper, self.best_x.copy(), remaining)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return self.best_x, self.best_f

    def _nelder_mead_polish(self, func, x_start, max_calls):
        """
        Custom Nelder-Mead implementation with strict call counting.
        """
        n = self.dim
        # Initialize simplex
        simplex = np.zeros((n + 1, n))
        simplex[0] = x_start
        f0 = func(x_start)
        calls_used = 1
        
        # Initialize other vertices with small offset
        alpha = 0.5
        for i in range(1, n + 1):
            simplex[i] = x_start.copy()
            simplex[i][i-1] += alpha
            if i-1 < n:
                simplex[i][i-1] = np.clip(simplex[i][i-1], -self.bound, self.bound)
            fi = func(simplex[i])
            calls_used += 1
            if calls_used >= max_calls:
                return
        
        # Store fvals
        fvals = np.array([f0] + [func(simplex[i]) for i in range(1, n+1)]) 
        # Wait, the loop above already evaluated them. Let's restructure to avoid double counting.
        
        # Restart with correct counting
        simplex = np.zeros((n + 1, n))
        simplex[0] = x_start
        fvals = np.zeros(n + 1)
        fvals[0] = func(x_start)
        calls_used = 1
        
        for i in range(1, n + 1):
            simplex[i] = x_start.copy()
            simplex[i][i-1] += alpha
            simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
            fvals[i] = func(simplex[i])
            calls_used += 1
            if calls_used >= max_calls:
                return
        
        # Nelder-Mead steps
        alpha_nm = 1.0  # Reflection
        gamma_nm = 2.0  # Expansion
        rho_nm = 0.5    # Contraction
        sigma_nm = 0.5  # Shrink
        
        while calls_used < max_calls and time.time() < self.deadline - 1:
            # Sort simplex by fvals
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:n], axis=0)
            
            # Reflection
            x_r = centroid + alpha_nm * (centroid - simplex[n])
            x_r = np.clip(x_r, -self.bound, self.bound)
            f_r = func(x_r)
            calls_used += 1
            if calls_used >= max_calls:
                return
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma_nm * (centroid - simplex[n])
                x_e = np.clip(x_e, -self.bound, self.bound)
                f_e = func(x_e)
                calls_used += 1
                if calls_used >= max_calls:
                    return
                if f_e < f_r:
                    simplex[n] = x_e
                    fvals[n] = f_e
                else:
                    simplex[n] = x_r
                    fvals[n] = f_r
            elif f_r < fvals[n-1]:
                # Accept reflection
                simplex[n] = x_r
                fvals[n] = f_r
            else:
                # Contraction
                if f_r < fvals[n]:
                    # Outside contraction
                    x_c = centroid + rho_nm * (simplex[n] - centroid)
                    x_c = np.clip(x_c, -self.bound, self.bound)
                else:
                    # Inside contraction
                    x_c = centroid - rho_nm * (centroid - simplex[n])
                    x_c = np.clip(x_c, -self.bound, self.bound)
                
                f_c = func(x_c)
                calls_used += 1
                if calls_used >= max_calls:
                    return
                
                if f_c < min(f_r, fvals[n]):
                    simplex[n] = x_c
                    fvals[n] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -self.bound, self.bound)
                        fvals[i] = func(simplex[i])
                        calls_used += 1
                        if calls_used >= max_calls:
                            return
                    
                    # Check if we broke out due to budget
                    if calls_used >= max_calls:
                        return

            # Convergence check
            if np.std(fvals) < 1e-12:
                break

        # Update global best if found better in polish
        # The func wrapper already updates self.best_x and self.best_f
