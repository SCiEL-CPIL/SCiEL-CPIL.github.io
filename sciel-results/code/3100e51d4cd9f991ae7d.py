import numpy as np
import time
from scipy.optimize import minimize

N = 400

def construct():
    # Deadline guard: 150s limit, aim to finish in 140s
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: Random symmetric-ish or just random
    # The knowledge says non-symmetric is better. 
    # Let's try a random start with mean 1.
    np.random.seed(42)
    h0 = np.random.randn(N) + 1.0
    # Ensure sum is not zero
    if np.sum(h0) == 0:
        h0 += 1.0

    # We minimize: 2 * N * max(conv) / sum(h)^2
    # To make it smooth, use log-sum-exp for max.
    # f(h) = 2 * N * (1/beta) * log(sum(exp(beta * conv))) / sum(h)^2
    # We can scale h to keep sum(h) = 1 to simplify? 
    # No, the ratio is invariant to scaling. 
    # Let's fix sum(h) = 1 by normalizing h inside the objective?
    # If we normalize, we reduce the dimensionality effectively.
    # Let g = h / sum(h). Then sum(g)=1.
    # Value = 2 * N * max(conv(g))
    # This removes the denominator and the non-zero sum constraint (as long as sum(h) != 0).
    # This is a better formulation.
    
    def objective_smooth(h, beta):
        # Normalize h to sum to 1
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10  # Penalty
        g = h / s
        
        # Autoconvolution
        conv = np.convolve(g, g)
        
        # Smooth max
        # Use log-sum-exp trick for stability
        max_conv = np.max(conv)
        # Subtract max for numerical stability
        exp_conv = np.exp(beta * (conv - max_conv))
        log_sum = np.log(np.sum(exp_conv))
        smooth_max = max_conv + (1.0 / beta) * log_sum
        
        return 2.0 * N * smooth_max

    best_h = h0.copy()
    best_val = objective_smooth(h0, 10.0)
    
    # Annealing schedule for beta
    # Start low to escape local minima, increase to sharpen
    betas = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    current_h = h0.copy()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        def obj_func(x):
            return objective_smooth(x, beta)
        
        try:
            res = minimize(obj_func, current_h, method='L-BFGS-B', options={'maxiter': 500, 'maxfun': 1000})
            if res.success or res.fun < best_val:
                current_h = res.x
                val = objective_smooth(current_h, beta)
                if val < best_val:
                    best_val = val
                    best_h = current_h.copy()
        except Exception:
            pass
        
        # If time is running out, stop
        if time.time() > deadline - 10.0:
            break

    # Final check: ensure the returned h is valid
    # The evaluator computes: abs(2 * N * np.max(np.convolve(h, h)) / np.sum(h) ** 2)
    # Our optimization minimized the smoothed version of this.
    # We should return best_h.
    
    # Optional: A quick local polish with a very high beta (approximating true max)
    if time.time() < deadline - 5.0:
        # Use a high beta for final polish
        beta_final = 1000.0
        def obj_func_final(x):
            return objective_smooth(x, beta_final)
        try:
            res = minimize(obj_func_final, best_h, method='L-BFGS-B', options={'maxiter': 100})
            if res.success:
                # Verify improvement with exact max
                h_cand = res.x
                s = np.sum(h_cand)
                if abs(s) > 1e-12:
                    g = h_cand / s
                    conv = np.convolve(g, g)
                    exact_val = 2.0 * N * np.max(conv)
                    # Compare with current best exact val
                    s_best = np.sum(best_h)
                    g_best = best_h / s_best
                    conv_best = np.convolve(g_best, g_best)
                    best_exact_val = 2.0 * N * np.max(conv_best)
                    
                    if exact_val < best_exact_val:
                        best_h = h_cand
        except Exception:
            pass

    return best_h

# EVOLVE-BLOCK-END
