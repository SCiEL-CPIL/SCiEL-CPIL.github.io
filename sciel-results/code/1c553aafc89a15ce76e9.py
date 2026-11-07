import numpy as np
import time
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Strategy: Start with uniform 0.5, then apply gradient descent or simple perturbations
    # to minimize the max correlation.
    # The seed is 0.5.
    # A sinusoidal perturbation reduces C_0 significantly.
    
    # Try a few specific structured patterns
    best_h = np.full(N, 0.5)
    best_val = 0.5
    
    def evaluate(h):
        # h is array of 100
        # Check constraints
        if np.any(h < -1e-9) or np.any(h > 1 + 1e-9):
            return float('inf')
        if abs(np.sum(h) - 50.0) > 1e-6 * N:
            return float('inf')
        
        # Clip to [0, 1] for safety in calculation, though we should maintain feasibility
        h_safe = np.clip(h, 0.0, 1.0)
        
        # Calculate value
        # correlate(h, 1-h, 'full')
        # Note: np.correlate in 'full' mode returns length 2N-1
        # The max is taken over all shifts.
        corr = np.correlate(h_safe, 1.0 - h_safe, mode='full')
        max_corr = np.max(corr)
        return max_corr / N * 2.0

    # Test seed
    val_seed = evaluate(best_h)
    
    # Test sinusoidal patterns
    # h_i = 0.5 + A * cos(2*pi*k*i/N + phase)
    for A in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for k in [1, 2, 3, 4, 5]:
            for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
                if time.time() > deadline:
                    break
                h_test = 0.5 + A * np.cos(2 * np.pi * k * np.arange(N) / N + phase)
                # Ensure sum is 50
                # Sum of cos is 0 if k is integer and N is multiple of period?
                # Sum_{i=0}^{N-1} cos(2*pi*k*i/N + phase) = 0 for integer k >= 1.
                # So sum is 50.
                # Check bounds
                if np.min(h_test) < -1e-9 or np.max(h_test) > 1 + 1e-9:
                    continue
                
                val_test = evaluate(h_test)
                if val_test < best_val:
                    best_val = val_test
                    best_h = h_test.copy()
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Use scipy.optimize.minimize with a soft-max objective
    # Objective: soft_max(corr) + penalty * (sum(h) - 50)^2
    # We optimize h in [0, 1]
    
    def soft_max(c, beta):
        # Numerically stable soft max
        c_max = np.max(c)
        return (c_max + np.log(np.sum(np.exp(beta * (c - c_max))))) / beta

    def objective(h):
        h = np.clip(h, 0.0, 1.0)
        corr = np.correlate(h, 1.0 - h, mode='full')
        # Use a moderate beta for smoothness
        beta = 10.0
        obj = soft_max(corr, beta)
        # Penalty for sum constraint
        sum_err = np.sum(h) - 50.0
        penalty = 100.0 * sum_err**2
        return obj / N * 2.0 + penalty

    # Gradient for objective (approximate or analytical?)
    # L-BFGS-B can use numerical gradients, which is slower but easier.
    # Let's try with numerical gradients first, as it's robust.
    # If time permits, we could implement analytical gradient, but N=100 is small.
    
    if time.time() < deadline - 10.0:
        bounds = [(0.0, 1.0) for _ in range(N)]
        # Start from best_h
        res = minimize(objective, best_h, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False})
        
        if res.success or res.fun < best_val:
            h_opt = res.x
            # Project to satisfy sum exactly
            h_opt = np.clip(h_opt, 0.0, 1.0)
            sum_err = np.sum(h_opt) - 50.0
            # Simple correction: adjust all elements slightly, then re-clip
            # This might break bounds, so we iterate
            for _ in range(10):
                h_opt -= sum_err / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
                sum_err = np.sum(h_opt) - 50.0
                if abs(sum_err) < 1e-9:
                    break
            
            if abs(np.sum(h_opt) - 50.0) < 1e-6 * N:
                val_opt = evaluate(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt

    return best_h
