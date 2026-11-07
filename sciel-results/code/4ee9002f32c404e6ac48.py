"""
Strategy: Global optimization via multi-start Nelder-Mead.
The objective function is scale-invariant. We maximize value = l2 / (norm_1 * norm_inf).
We use scipy.optimize.minimize with Nelder-Mead method, which is robust for non-smooth
functions (due to max norm) and handles the 50-dimensional space effectively.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

def compute_value(h):
    """Compute the value metric for given heights."""
    h = np.array(h, dtype=float)
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    
    # Ensure non-negative for validity, though optimizer might explore negatives
    # The problem states heights must be non-negative.
    # We will handle this in the objective function by penalizing or clipping.
    # For speed in compute_value used for final check, we assume valid input.
    
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    w = 1.0 / (m + 1)
    
    y_sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(y_sq)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return 0.0
    
    value = l2 / (norm_1 * norm_inf)
    return value

def objective(h):
    """
    Objective function for minimization (we minimize -value).
    Handles non-negativity constraint by returning a large penalty if any h < 0.
    """
    if np.any(h < 0):
        return 1e6 + np.sum(h[h < 0] ** 2)
    
    # Scale invariance: we can normalize h to prevent overflow/underflow issues
    # in the optimizer, though Nelder-Mead is scale-aware.
    # However, if h is all zeros, value is 0.
    if np.all(h == 0):
        return 0.0
        
    val = compute_value(h)
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # 2 second safety margin
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Generate diverse initial points for multi-start
    initial_points = []
    
    # 1. Seed
    initial_points.append(np.ones(N))
    
    # 2. Random uniform
    for _ in range(5):
        initial_points.append(np.random.rand(N))
        
    # 3. Sparse spikes
    for _ in range(5):
        h = np.zeros(N)
        idx = np.random.choice(N, size=5, replace=False)
        h[idx] = 1.0
        initial_points.append(h)
        
    # 4. Blocky patterns
    for _ in range(5):
        h = np.zeros(N)
        start = np.random.randint(0, N - 10)
        length = np.random.randint(5, 20)
        h[start:start+length] = 1.0
        initial_points.append(h)
        
    # 5. Gaussian bumps
    for _ in range(5):
        x = np.linspace(-1, 1, N)
        center = np.random.uniform(-0.5, 0.5)
        width = np.random.uniform(0.1, 0.5)
        h = np.exp(-((x - center)**2) / (2 * width**2))
        initial_points.append(h)

    # Shuffle initial points to ensure diverse exploration order
    np.random.shuffle(initial_points)
    
    # Nelder-Mead options
    # maxiter: limit iterations per start to ensure we finish in time
    # maxfev: limit function evaluations
    # We have ~148 seconds. Each evaluation is fast (convolve of 50).
    # Let's allow up to 5000 evaluations per start, but check time.
    
    options = {
        'maxiter': 2000,
        'maxfev': 2000,
        'xatol': 1e-5,
        'fatol': 1e-5,
        'disp': False
    }
    
    for i, h0 in enumerate(initial_points):
        if time.time() > deadline - 5:
            break
            
        # Ensure h0 is non-negative
        h0 = np.maximum(0, h0)
        
        try:
            # Run Nelder-Mead
            res = minimize(
                objective, 
                h0, 
                method='Nelder-Mead', 
                options=options
            )
            
            # Check if result is valid (non-negative)
            if res.success or res.fun < 0: # If we found a negative objective (positive value)
                h_opt = res.x
                if np.any(h_opt < 0):
                    # Clip small negatives due to numerical noise
                    h_opt = np.maximum(0, h_opt)
                
                val = compute_value(h_opt)
                if val > best_val:
                    best_val = val
                    best_h = h_opt
                    # Print progress (optional, but helps debug)
                    # print(f"Start {i}: New best {best_val:.4f}")
                    
        except Exception as e:
            # If optimization fails, ignore and try next start
            pass
            
    # Final local polish on the best found solution
    # Sometimes Nelder-Mead stops a bit early. A quick coordinate ascent can help.
    if time.time() < deadline - 2:
        h_cur = best_h.copy()
        step = 0.1
        for _ in range(1000):
            if time.time() > deadline - 2:
                break
            improved = False
            idx = np.random.randint(0, N)
            # Try a few steps in + and - direction
            for delta in [step, -step, step*0.1, -step*0.1]:
                new_h = h_cur.copy()
                new_h[idx] = max(0, new_h[idx] + delta)
                val = compute_value(new_h)
                if val > best_val:
                    best_val = val
                    best_h = new_h
                    h_cur = new_h
                    improved = True
                    break
            if not improved:
                step *= 0.95
                if step < 1e-6:
                    break

    return best_h.tolist()

# Keep the function name and signature as required
# The prompt asks to keep function names and signatures of the seed program.
# Seed program had: construct() -> sequence of 50 nonnegative step heights.
# It also had compute_value defined.
