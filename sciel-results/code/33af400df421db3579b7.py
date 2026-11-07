import numpy as np
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Define the objective function to minimize (negative of the value)
    def objective(h):
        h = np.asarray(h)
        # Ensure non-negative for safety during optimization steps, though bounds handle it
        h = np.maximum(h, 0)
        
        g = np.convolve(h, h)
        m = len(g)
        
        # Construct y
        y = np.zeros(m + 2)
        y[1:m+1] = g
        
        # Calculate l2
        # Sum_{i=0}^m (y[i]^2 + y[i]y[i+1] + y[i+1]^2)
        # = 2 * sum(y^2) + sum(y[:-1] * y[1:])
        # Note: y[0]=0, y[m+1]=0
        s1 = 2.0 * np.sum(y**2)
        s2 = np.sum(y[:-1] * y[1:])
        sum_val = s1 + s2
        
        w = 1.0 / (m + 1) # 0.01
        l2 = (w / 3.0) * sum_val
        
        norm_1 = np.sum(np.abs(g)) / (m + 1)
        
        # Smooth max for differentiability
        p = 20
        if np.max(g) < 1e-10:
            return 0.0
        norm_inf_smooth = np.sum(g**p) ** (1.0 / p)
        
        denom = norm_1 * norm_inf_smooth
        if denom < 1e-10:
            return 0.0
            
        val = l2 / denom
        return -val

    # Constraints
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0})
    bounds = [(0.0, None) for _ in range(N)]
    
    # Initial guesses
    guesses = []
    
    # 1. Uniform
    guesses.append(np.ones(N) / N)
    
    # 2. Gaussian (narrow)
    x = np.linspace(-1, 1, N)
    g1 = np.exp(-x**2 * 10)
    guesses.append(g1 / np.sum(g1))
    
    # 3. Gaussian (wide)
    g2 = np.exp(-x**2 * 2)
    guesses.append(g2 / np.sum(g2))
    
    # 4. Random
    np.random.seed(42)
    r = np.random.rand(N)
    guesses.append(r / np.sum(r))
    
    best_val = -np.inf
    best_h = guesses[0]
    
    for h0 in guesses:
        try:
            res = minimize(objective, h0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_h = res.x
        except Exception:
            continue
            
    # Final cleanup
    best_h = np.maximum(best_h, 0)
    best_h /= np.sum(best_h)
    
    return list(best_h)
# EVOLVE-BLOCK-END
