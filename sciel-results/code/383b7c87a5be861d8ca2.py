"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def objective(h):
    """
    Compute the evaluation value:
    value = max(np.correlate(h, 1 - h, mode="full")) / len(h) * 2
    """
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0

def construct():
    """
    Uses Gradient-Based Optimization (L-BFGS-B) to minimize the maximum correlation.
    Strategy:
    1. The objective function f(h) = max_k (corr_k) is convex because it is the maximum of linear functions (each correlation term is linear in h).
    2. We minimize this convex function subject to box constraints [0, 1] and a linear equality constraint sum(h) = N/2.
    3. We use multiple diverse initializations to ensure we find the global minimum (which is likely unique or close to it due to convexity, but numerical precision might vary).
    """
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # Initial best is the constant 0.5 solution
    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Constraints for scipy.optimize.minimize
    # Equality constraint: sum(h) - N/2 = 0
    def sum_constraint(h):
        return np.sum(h) - N / 2.0
    
    constraints = [{'type': 'eq', 'fun': sum_constraint}]
    
    # Bounds for each variable
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    # Objective function for scipy (must return scalar)
    def obj_func(h):
        # Ensure h is 1D array
        h = np.asarray(h).flatten()
        return objective(h)
    
    # Define several diverse initializations
    inits = []
    
    # 1. Constant 0.5
    inits.append(np.full(N, 0.5))
    
    # 2. Alternating 0, 1
    h_alt = np.zeros(N)
    h_alt[1::2] = 1.0
    inits.append(h_alt)
    
    # 3. Step function
    h_step = np.zeros(N)
    h_step[N//2:] = 1.0
    inits.append(h_step)
    
    # 4. Random uniform projected to sum=50
    np.random.seed(42)
    h_rand = np.random.uniform(0, 1, N)
    h_rand += (N/2.0 - np.sum(h_rand)) / N
    h_rand = np.clip(h_rand, 0, 1)
    # Re-project roughly
    for _ in range(20):
        h_rand = np.clip(h_rand, 0, 1)
        h_rand += (N/2.0 - np.sum(h_rand)) / N
    inits.append(h_rand)
    
    # 5. Multi-frequency sinusoidal
    x = np.linspace(0, 2 * np.pi, N)
    h_sin = 0.5 + 0.4 * np.sin(x) + 0.1 * np.sin(3*x)
    h_sin += (N/2.0 - np.sum(h_sin)) / N
    h_sin = np.clip(h_sin, 0, 1)
    for _ in range(20):
        h_sin = np.clip(h_sin, 0, 1)
        h_sin += (N/2.0 - np.sum(h_sin)) / N
    inits.append(h_sin)
    
    # Run optimization for each initialization
    for i, h_init in enumerate(inits):
        if time.time() - start_time > time_limit:
            break
        
        # Check if we already have a very good solution
        if best_val < 0.4: # Theoretical lower bound is likely around 0.5? Actually 0.5 is the seed. 
                           # If we drop below 0.5, we are improving.
                           # Wait, seed is 0.5. Score 1.0 corresponds to value 0.5.
                           # Current best score 1.2930 implies value ~ 0.386.
                           # So we want value < 0.386.
                           # If we get very low, we can stop.
                           # Let's just run all inits if time permits.
                           pass
        
        try:
            # Use L-BFGS-B with constraints
            res = minimize(
                obj_func, 
                h_init, 
                method='L-BFGS-B', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success:
                h_opt = res.x
                # Repair any slight numerical violations of the sum constraint
                # L-BFGS-B with equality constraints should handle it, but just in case
                diff = N/2.0 - np.sum(h_opt)
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0, 1)
                
                # Final projection to ensure sum is exactly N/2 within tolerance
                # Simple iterative projection
                for _ in range(10):
                    h_opt = np.clip(h_opt, 0, 1)
                    diff = N/2.0 - np.sum(h_opt)
                    h_opt += diff / N
                
                val_opt = objective(h_opt)
                
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt.copy()
                    
        except Exception as e:
            # If optimization fails, ignore and try next init
            continue

    # Final check
    if not np.all(np.isfinite(best_h)) or np.abs(np.sum(best_h) - N/2.0) > 1e-9 * N:
        return np.full(N, 0.5)
        
    return best_h

# EVOLVE-BLOCK-END
