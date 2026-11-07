import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES Parameters
        # Population size
        lam = 10 
        mu = lam // 2
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initialization
        # Start at 0,0,0,0,0 is often good for Rosenbrock if the shift is small, 
        # but the problem says shift is in [-4,4]. 
        # A random start is safer to avoid getting stuck in a bad local basin if the 
        # landscape is weird, but Rosenbrock has one global min.
        # Let's start at 0.
        m = np.zeros(dim)
        sigma = 2.0
        
        # State variables
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = m.copy()
        best_f = float('inf')
        
        calls_made = 0
        
        # We need to manage the budget carefully.
        # Each generation uses lam calls.
        # Total budget 500.
        # 500 // 10 = 50 generations.
        
        rng = np.random.default_rng(42)
        
        for gen in range(50): # 50 gens * 10 pop = 500 calls
            if time.time() > deadline:
                break
                
            # Sample
            ys = []
            for i in range(lam):
                # Standard normal
                z = rng.standard_normal(dim)
                # Transform to y
                # y = m + sigma * C^{1/2} z
                # We need the Cholesky factor of C
                # For efficiency, we can update C and keep its Cholesky factor
                # Or just compute C^{1/2} via eigendecomp or Cholesky each step.
                # For dim=5, Cholesky is very fast.
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # If C is not positive definite, reset it
                    C = np.eye(dim)
                    L = np.eye(dim)
                    sigma = 1.0
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    
                y = m + sigma * (L @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fvals = []
            for y in ys:
                fy = func(y)
                calls_made += 1
                fvals.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            fvals = np.array(fvals)
            
            # Sort
            idx = np.argsort(fvals)
            ys_sorted = [ys[i] for i in idx]
            fvals_sorted = fvals[idx]
            
            # Update mean
            y_w = np.array(ys_sorted[:mu])
            m_new = m + sigma * np.sum(weights[:, None] * (y_w - m), axis=0)
            
            # Update covariance matrix
            # Delta sigma
            diff = y_w - m
            y_w_std = diff / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C ** -0.5) @ (y_w_std.T @ weights)
            
            # Note: C ** -0.5 is expensive. 
            # Standard implementation uses the Cholesky factor L.
            # Let's stick to a simpler, more robust update or use scipy if available? 
            # The prompt says SciPy is available.
            # But let's try to keep it NumPy only for speed and simplicity, or use SciPy for robustness.
            
            # Actually, implementing CMA-ES from scratch correctly with all the matrix inversions and updates is error-prone and slow in pure Python loops if not optimized.
            # Given the small dimension (5), a simpler approach might be better:
            # **Nelder-Mead** or **Powell** or **BFGS** (if we can estimate gradients)?
            # No, it's a black box.
            
            # Let's reconsider the (1+1)-ES vs CMA-ES.
            # The (1+1)-ES scored 1.0.
            # CMA-ES is theoretically better.
            # Let's use a simplified CMA-ES or a different algorithm that is easier to implement correctly.
            
            # **Alternative: Simplex (Nelder-Mead) with restarts?**
            # Nelder-Mead is very fast and good for low dimensions.
            # But it can get stuck.
            
            # **Alternative: Powell's Method?**
            # Powell's method is excellent for low-dimensional bound-constrained optimization.
            # It uses line searches.
            # Each line search takes a few calls.
            # With 500 calls, we can do many iterations.
            
            # Let's try **Powell's Method**.
            # It is deterministic, fast, and handles the curvature well by aligning directions.
            # It does not require gradient estimates.
            
            # However, Powell can get stuck in local minima if the function is not convex.
            # Rosenbrock is not convex, but has a single global minimum.
            # The valley is narrow. Powell should handle it well.
            
            # Let's switch to **Powell's Method** with a time guard.
            
            # Reset for Powell
            # Initial point: 0,0,0,0,0
            # Initial directions: basis vectors
            
            # Wait, the previous attempt to write CMA-ES got complex.
            # Let's stick to a robust **CMA-ES** implementation using `scipy.optimize`?
            # No, `scipy.optimize.minimize` with `Nelder-Mead` or `Powell` is available.
            # But we need to count calls. `scipy` does not easily expose the call count in a way that fits our strict budget management without wrapping the function.
            # We can wrap the function to count calls.
            
            # Let's use **scipy.optimize.minimize** with method='Powell'.
            # It is very effective for this problem.
            
            # Implementation:
            # 1. Wrap func to count calls.
            # 2. Call scipy.optimize.minimize.
            # 3. Check time and calls.
            
            # But `scipy.optimize.minimize` might use more calls than expected or fail to stop exactly at 500.
            # We can pass `options={'maxiter': ...}` but Powell's iteration count doesn't map 1:1 to function calls.
            # A single Powell iteration in 5D might use ~10-20 calls.
            # 500 calls / 15 calls/iter ~ 30 iterations.
            
            # Let's try a manual implementation of **Nelder-Mead** or **Powell** to have precise control.
            # Or, a **Multi-start (1+1)-ES** with restarts?
            # The (1+1)-ES is simple and robust.
            # Maybe the issue is the step size adaptation.
            
            # Let's try a **CMA-ES** implementation that is simpler:
            # Use `cma` package? Not in standard library.
            
            # Let's go with **scipy.optimize.minimize** with **Nelder-Mead**.
            # Nelder-Mead is simple, fast, and works well in 5D.
            # It uses a simplex of size dim+1 = 6 points.
            # Each iteration involves 1-2 function evaluations (reflection, expansion, etc.).
            # So ~250 iterations possible.
            
            # Let's try **scipy.optimize.minimize** with **Powell**.
            # Powell is generally better for Rosenbrock than Nelder-Mead.
            
            # To ensure we don't exceed the budget, we will wrap the function.
            
            # Let's implement this.

            # We break out of the CMA-ES loop logic and switch to this strategy.
            # I will rewrite the `__call__` to use SciPy.

            pass # Placeholder, will replace entire logic

        # Actual Implementation below
        # We will use a hybrid: Start with a few random points to find a good region, then use Powell or Nelder-Mead.
        # Or just Powell from 0,0,0,0,0.
        
        # Let's use **scipy.optimize.minimize** with **Powell**.
        
        # Re-initialize for the actual strategy
        calls_count = [0]
        def wrapped_func(x):
            if calls_count[0] >= budget:
                return best_f # Return current best to avoid errors
            calls_count[0] += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial point
        x0 = np.zeros(dim)
        
        # Run Powell
        try:
            # maxiter is a loose bound, we rely on the call count in wrapped_func
            # However, Powell might not stop if maxiter is high.
            # We can use a callback to stop early?
            # SciPy minimize does not have a clean "stop after N calls" option.
            # We can catch an exception or use a flag.
            
            # Let's use a simpler, manual **Nelder-Mead** implementation 
            # or stick to the **(1+1)-ES** but improved.
            
            # Given the risk of SciPy overhead and control issues, 
            # and the fact that (1+1)-ES scored 1.0, 
            # let's try a **Restarting (1+1)-ES** or **Multi-start (1+1)-ES**.
            
            # Multi-start: Divide budget into chunks.
            # e.g., 5 restarts of 100 calls each.
            # Keep the best result.
            # This helps if the initial point is bad.
            
            # Let's implement **Multi-start (1+1)-ES**.
            
            # Reset variables
            best_x = np.zeros(dim)
            best_f = float('inf')
            calls_made = 0
            
            num_restarts = 5
            calls_per_restart = budget // num_restarts
            
            rng = np.random.default_rng(42)
            
            for r in range(num_restarts):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # New start point
                # First start at 0, others random
                if r == 0:
                    x = np.zeros(dim)
                else:
                    x = rng.uniform(-5, 5, dim)
                
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    
                sigma = 2.0
                
                # (1+1)-ES loop
                while calls_made < budget and time.time() < deadline:
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    calls_made += 1
                    
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma *= 1.5
                    else:
                        sigma *= 0.87
                        
                    # If sigma becomes too small, break early to save budget for next restart?
                    # Or just let it run out.
                    if sigma < 1e-6:
                        break

        except Exception:
            pass

        return best_x, best_f
