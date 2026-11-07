"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with restarts and time limit."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Explicit deadline guard: 300s limit, use 295s for safety
        start_time = time.time()
        deadline = start_time + 295
        
        best_x = None
        best_fx = float('inf')
        calls = 0
        
        def evaluate(x):
            nonlocal best_x, best_fx, calls
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        
        # Phase 1: Global Exploration (Coarse Grid)
        # We have 500 calls. A grid of 3^5 = 243 points is affordable.
        # This ensures we find a good region quickly, which is crucial for AOCC.
        # We use a grid in [-4, 4] to avoid boundary issues and cover the likely minimizer location.
        # Actually, let's use a slightly wider range or just standard uniform randoms 
        # if grid is too slow to set up, but 243 points is trivial.
        
        # Grid points: -4, 0, 4 (3 points per dimension)
        # This covers the box well.
        grid_coords = [-4.0, 0.0, 4.0]
        
        # Generate all combinations
        # This is 3^5 = 243 points.
        # We will evaluate them.
        # To maximize early progress (AOCC), we should evaluate points that are likely to be good early on.
        # However, for Sphere, the center (0,0,0) is often close to the minimizer if shifts are small.
        # But shifts can be up to 4. 
        # Let's just iterate through the grid.
        
        # We can vectorize the generation but we must call func one by one to track calls and time.
        # Or we can just loop. 243 loops is fast.
        
        # Let's prioritize the center first?
        # The prompt says "minimizer lies in [-4, 4]^5".
        # Let's start with the origin, then expand.
        
        # Actually, a random sampling might be better for early "lucky" hits?
        # No, for Sphere, a coarse grid guarantees finding a point within a certain distance.
        # Distance from grid point to any point in [-4,4] is at most sqrt(5) * (4/2) = ~4.47?
        # Grid spacing is 4. Max distance to nearest grid point is sqrt(5) * 2 = ~4.47.
        # This is a very coarse net.
        
        # Let's do a mix:
        # 1. Evaluate origin (1 call)
        # 2. Evaluate 243 grid points (243 calls)
        # Total 244 calls.
        
        # But wait, if the minimizer is at (4, 4, 4, 4, 4), the grid point (4,4,4,4,4) is exact.
        # If it's at (2, 2, 2, 2, 2), the nearest grid point is (0,0,0,0,0) or (4,4,4,4,4)?
        # (2,2,2,2,2) is equidistant to (0,0,0,0,0) and (4,4,4,4,4) in L2? No.
        # Distance to (0,0,0,0,0) is sqrt(5)*2 = 4.47.
        # Distance to (4,4,4,4,4) is sqrt(5)*2 = 4.47.
        # This is bad. The grid is too coarse.
        
        # Better: Use random sampling for the first 100 calls to find a good basin,
        # then use the quadratic fit on the best point found.
        
        # Let's stick to the directive: "Replace the explicit quadratic coefficient fitting with a stochastic gradient estimation (SGD) approach..."
        # Wait, the directive says: "Replace the explicit quadratic coefficient fitting with a stochastic gradient estimation (SGD) approach that uses finite differences on a small set of random perturbations to directly approximate the Hessian and gradient..."
        
        # So the strategy is:
        # 1. Find a decent starting point (maybe via a few random samples or just origin).
        # 2. Use finite differences to estimate Gradient and Hessian at that point.
        # 3. Solve for the minimizer of the local quadratic approximation.
        # 4. Evaluate that minimizer.
        # 5. If it's better, repeat (Newton step) until convergence or budget exhausted.
        
        # This is essentially a derivative-free Newton method using finite differences.
        # It is much more efficient for smooth convex functions like Sphere than (1+1)-ES.
        
        # Phase 1: Initial Point Selection
        # Let's evaluate the origin first.
        x = np.zeros(dim)
        fx = evaluate(x)
        
        # If we have budget left, let's do a few random restarts to find a better start?
        # Or just proceed with Newton from origin?
        # Sphere is convex, so any starting point works. But if the shift is large,
        # the local quadratic approximation at origin might point in the wrong direction if the curvature is estimated poorly?
        # No, for Sphere f(x) = (x-c)^T A (x-c) + f0.
        # The Hessian is 2A. The Gradient is 2A(x-c).
        # The Newton step x_new = x - H^-1 G = x - (2A)^-1 2A(x-c) = x - (x-c) = c.
        # So, if we can estimate H and G accurately, one Newton step finds the exact minimum!
        # This is the key insight.
        
        # So the strategy is:
        # 1. Pick a starting point x0.
        # 2. Estimate Hessian H and Gradient G at x0 using finite differences.
        # 3. Compute x1 = x0 - H^-1 G.
        # 4. Evaluate x1.
        # 5. If x1 is better, set x0 = x1 and repeat (though for Sphere, one step should be nearly exact if FD is accurate).
        # 6. Use remaining budget to polish or verify.
        
        # How to estimate H and G?
        # Central differences for Gradient:
        # G_i = (f(x + h e_i) - f(x - h e_i)) / (2h)
        # This requires 2*dim calls.
        
        # Central differences for Hessian:
        # H_ii = (f(x + h e_i) - 2f(x) + f(x - h e_i)) / h^2
        # H_ij = (f(x + h(e_i+e_j)) - f(x + h(e_i-e_j)) - f(x - h(e_i+e_j)) + f(x - h(e_i-e_j))) / (4h^2)
        # This is expensive. 2*dim + dim*(dim-1)/2 * 4 calls?
        # For dim=5:
        # Gradient: 10 calls.
        # Hessian diagonal: 10 calls (can reuse from Gradient? No, Gradient uses f(x+he) and f(x-he). Hessian diagonal uses the same points!
        # So we can compute both G and H_ii from the same 2*dim calls.
        # Hessian off-diagonal: 5*4/2 = 10 pairs. Each pair needs 4 calls. Total 40 calls.
        # Total calls for one full Newton step setup: 1 (center) + 10 (grad/diag) + 40 (off-diag) = 51 calls.
        
        # We have 500 calls. We can afford ~9 such steps.
        # This is very robust.
        
        # Let's refine the plan:
        # 1. Start at x = np.zeros(dim).
        # 2. Loop:
        #    a. If budget < 55, break.
        #    b. Estimate G and H at x.
        #    c. Compute Newton step.
        #    d. Evaluate new x.
        #    e. If new x is not better, break (or try a smaller step?).
        #    f. Update x.
        
        # We need to choose h.
        # For Sphere, h can be relatively large, but not too large to leave the quadratic region.
        # Since the whole function IS quadratic, h can be large.
        # However, numerical stability of FD:
        # h^2 in denominator for Hessian. If h is too small, noise dominates.
        # If h is too large, we are fine for Sphere, but let's keep it moderate.
        # Let's try h = 0.5.
        
        # Let's implement this.
        
        h = 0.5
        
        # We will use the budget to perform these steps.
        # Each step costs ~51 calls.
        # 500 / 51 = 9 steps.
        
        # To ensure we don't waste budget if we converge early, we check the improvement.
        
        # Also, we should handle the case where H is not positive definite (shouldn't happen for Sphere, but just in case).
        # We can add a small ridge to H: H + lambda I.
        
        for _ in range(10): # Max 10 steps
            if calls + 55 > self.budget:
                break
            if time.time() >= deadline:
                break
                
            # Estimate Gradient and Hessian
            # f0 = fx (already evaluated)
            
            # 1. Gradient and Diagonal Hessian
            # Points: x +/- h e_i
            # We need to store f(x+he_i) and f(x-he_i)
            f_plus = np.zeros(dim)
            f_minus = np.zeros(dim)
            
            for i in range(dim):
                if time.time() >= deadline: break
                x_plus = x.copy()
                x_plus[i] += h
                x_minus = x.copy()
                x_minus[i] -= h
                
                f_plus[i] = evaluate(x_plus)
                f_minus[i] = evaluate(x_minus)
                
            # Check if we ran out of budget mid-way
            if calls > self.budget - 1: break
            
            # Compute Gradient
            G = (f_plus - f_minus) / (2 * h)
            
            # Compute Diagonal Hessian
            H = np.zeros((dim, dim))
            for i in range(dim):
                H[i, i] = (f_plus[i] - 2 * fx + f_minus[i]) / (h * h)
            
            # 2. Off-diagonal Hessian
            # Pairs (i, j) with i < j
            # H_ij = (f(x + h(e_i+e_j)) - f(x + h(e_i-e_j)) - f(x - h(e_i+e_j)) + f(x - h(e_i-e_j))) / (4h^2)
            
            for i in range(dim):
                for j in range(i+1, dim):
                    if time.time() >= deadline: break
                    if calls + 4 > self.budget: break
                    
                    x1 = x.copy(); x1[i] += h; x1[j] += h
                    x2 = x.copy(); x2[i] += h; x2[j] -= h
                    x3 = x.copy(); x3[i] -= h; x3[j] += h
                    x4 = x.copy(); x4[i] -= h; x4[j] -= h
                    
                    f1 = evaluate(x1)
                    f2 = evaluate(x2)
                    f3 = evaluate(x3)
                    f4 = evaluate(x4)
                    
                    val = (f1 - f2 - f3 + f4) / (4 * h * h)
                    H[i, j] = val
                    H[j, i] = val
            
            if calls > self.budget - 1: break
            if time.time() >= deadline: break
            
            # Solve H dx = -G
            # Add ridge for stability
            H_reg = H + 1e-8 * np.eye(dim)
            
            try:
                dx = np.linalg.solve(H_reg, -G)
            except np.linalg.LinAlgError:
                break
            
            # Newton step
            x_new = x + dx
            
            # Clip to box
            x_new = np.clip(x_new, -5, 5)
            
            # Evaluate
            fx_new = evaluate(x_new)
            
            # Check improvement
            if fx_new < fx:
                x = x_new
                fx = fx_new
            else:
                # If no improvement, we might be at the minimum or H is bad.
                # For Sphere, this should rarely happen unless we are very close.
                break

        # Use remaining budget for a simple (1+1)-ES polish if we have calls left
        # This ensures we squeeze out the last bit of precision.
        if calls < self.budget:
            sigma = 0.1
            for _ in range(self.budget - calls):
                if time.time() >= deadline:
                    break
                if calls >= self.budget:
                    break
                    
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                    
                if sigma < 1e-6:
                    break

        if best_x is None:
            best_x = np.zeros(dim)
            best_fx = func(best_x)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
