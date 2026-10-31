import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # Objective: Max overlap of h with shifted 1-h
    def objective(h):
        # h is flat array
        # Calculate L_k for all k
        # L_k = sum_{i=0}^{N-k-1} h[i] * (1 - h[i+k])
        max_overlap = 0.0
        # Optimization: use convolution?
        # h * (1-reversed_h)? No, shifts are different.
        # Loop is fine for N=100
        for k in range(N):
            # Slice lengths
            len_slice = N - k
            if len_slice <= 0:
                break
            # h[0:len_slice] and h[k:N]
            # Product sum
            overlap = np.dot(h[:len_slice], 1.0 - h[k:])
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    # Constraints
    # Sum h = N/2
    # 0 <= h <= 1
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N / 2.0}]
    bounds = [(0.0, 1.0)] * N

    best_h = np.full(N, 0.5)
    best_val = objective(best_h)
    
    # Strategy: Multiple random restarts
    # Use binary-like starts to encourage binary solutions
    num_restarts = 50  # Adjust if slow
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Random binary start
        x0 = np.random.randint(0, 2, N).astype(float)
        # Ensure sum is close to N/2? 
        # SLSQP will fix it, but a closer start might be better.
        # Let's just let SLSQP handle the constraint.
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints, 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success:
                h_res = res.x
                # Validate
                if np.all(h_res >= -1e-5) and np.all(h_res <= 1.0 + 1e-5):
                    # Clip
                    h_res = np.clip(h_res, 0.0, 1.0)
                    val = objective(h_res)
                    if val < best_val:
                        best_val = val
                        best_h = h_res.copy()
        except Exception:
            pass

    return best_h
