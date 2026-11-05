import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    best_h = np.ones(N)
    best_val = 2.0  # Seed value
    
    # Objective function: maximize sum=1, minimize max(conv)
    # Actually, we minimize f(h) = max(conv(h,h)) subject to sum(h)=1
    # The value reported is 2*N*f(h)
    
    def objective_and_grad(h):
        # h is flattened
        h = np.reshape(h, (N,))
        conv = np.convolve(h, h)
        max_val = np.max(conv)
        
        # Gradient of max(conv)
        # Active indices
        # Use a small tolerance to handle numerical noise
        tol = 1e-8 * max(1.0, abs(max_val))
        active = np.where(np.abs(conv - max_val) <= tol)[0]
        
        grad = np.zeros(N)
        for k in active:
            # grad_j += 2 * h[k-j] if 0 <= k-j < N
            # Vectorized:
            # j in 0..N-1
            # idx = k - np.arange(N)
            # valid = (idx >= 0) & (idx < N)
            # grad[valid] += 2 * h[idx[valid]]
            idx = k - np.arange(N)
            valid = (idx >= 0) & (idx < N)
            grad[valid] += 2 * h[idx[valid]]
            
        # Average over active constraints for smoothness (subgradient)
        if len(active) > 0:
            grad /= len(active)
            
        return max_val, grad

    def objective(h):
        val, _ = objective_and_grad(h)
        return val

    def gradient(h):
        _, g = objective_and_grad(h)
        return g

    constraints = [
        {'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}
    ]

    # Try a few random starts
    np.random.seed(42)
    num_starts = 10
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Random initialization
        h0 = np.random.randn(N)
        # Normalize to sum 1
        s = np.sum(h0)
        if s == 0:
            h0 += 1e-6
            s = np.sum(h0)
        h0 = h0 / s
        
        try:
            res = minimize(
                objective, 
                h0, 
                jac=gradient, 
                method='L-BFGS-B', 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            h_opt = res.x
            val_opt, _ = objective_and_grad(h_opt)
            
            # Calculate actual score value
            conv_opt = np.convolve(h_opt, h_opt)
            max_conv = np.max(conv_opt)
            sum_h = np.sum(h_opt)
            
            if sum_h == 0:
                continue
                
            value = abs(2 * N * max_conv / (sum_h ** 2))
            
            if value < best_val:
                best_val = value
                best_h = h_opt.copy()
                
        except Exception as e:
            pass

    # Ensure the result is valid
    if np.sum(best_h) == 0:
        best_h = np.ones(N)
        
    return best_h
