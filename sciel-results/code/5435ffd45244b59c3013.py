import numpy as np
import time
from scipy.optimize import minimize

N = 100

def compute_value(h):
    # h is an array of length N
    # Compute full correlation of h and 1-h
    # np.correlate(h, 1-h, mode='full')
    # The max is taken over the full correlation
    # Value = max / N * 2
    corr = np.correlate(h, 1 - h, mode="full")
    max_corr = np.max(corr)
    return max_corr / N * 2

def smooth_value(h, beta=1000.0):
    """
    Smooth approximation of the max correlation using log-sum-exp.
    value = (1/N) * 2 * max_k corr_k
    approx = (1/N) * 2 * (1/beta) * log(sum(exp(beta * corr_k)))
    """
    corr = np.correlate(h, 1 - h, mode="full")
    # Shift to avoid overflow in exp if corr is large, though corr is bounded by N
    # max possible corr is N (if h=1, 1-h=0? No, h in [0,1], 1-h in [0,1]. Max dot product is N*1*1=N)
    # Actually max corr is sum(h_i * (1-h_{i+k})). If h_i=1, 1-h=0. If h_i=0.5, 1-h=0.5.
    # Max value of corr is around N/4 * 2? No.
    # Let's just use stable log-sum-exp.
    m = np.max(corr)
    # If m is very negative, exp(beta*corr) might underflow, but log-sum-exp handles it.
    # log(sum(exp(beta*(corr - m)))) + beta*m
    # scaled_corr = beta * (corr - m)
    # log_sum = np.log(np.sum(np.exp(scaled_corr)))
    # result = (1/N) * 2 * (m + log_sum/beta)
    
    # For numerical stability, if beta is large, exp(beta*(corr-m)) is 0 for corr < m, 1 for corr=m.
    # This approximates the max.
    scaled_corr = beta * (corr - m)
    # Clip to avoid overflow in exp? exp(0)=1, exp(negative) < 1. So no overflow.
    # Underflow to 0 is fine.
    log_sum = np.log(np.sum(np.exp(scaled_corr)))
    return (m + log_sum / beta) / N * 2

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_val = compute_value(best_h)
    
    # Parameters for smooth objective
    beta = 500.0
    
    # Objective function for optimizer (smooth)
    def objective_smooth(h_vec):
        return smooth_value(h_vec, beta)
        
    # Exact objective for final evaluation
    def objective_exact(h_vec):
        return compute_value(h_vec)

    # Constraints
    def constraint_sum(h_vec):
        return np.sum(h_vec) - N/2
        
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    bounds = [(0, 1)] * N
    
    # Generate seeds
    # We want diverse structures. Sinusoids with integer frequencies are good.
    # Also random noise.
    seeds = []
    
    # 1. Uniform
    seeds.append(np.full(N, 0.5))
    
    # 2. Sinusoids with integer frequencies 1..5
    x = np.arange(N)
    for f_int in range(1, 6):
        for A in [0.3, 0.4, 0.45]:
            for phi in np.linspace(0, 2*np.pi, 4, endpoint=False):
                h_seed = 0.5 + A * np.sin(2 * np.pi * f_int * x / N + phi)
                # Adjust sum
                sum_err = np.sum(h_seed) - N/2
                h_seed -= sum_err / N
                # Clip
                h_seed = np.clip(h_seed, 0, 1)
                # Re-adjust sum if clipping changed it significantly?
                # If clipping happens, sum might be off.
                # For A<=0.45, clipping is unlikely unless phase aligns with 0 or 1.
                # If clipping occurs, we project back.
                if not (np.all(h_seed >= 0) and np.all(h_seed <= 1)):
                     # Simple projection: mean shift
                     h_seed -= (np.sum(h_seed) - N/2) / N
                     h_seed = np.clip(h_seed, 0, 1)
                seeds.append(h_seed)
    
    # 3. Random noise
    np.random.seed(42)
    for _ in range(5):
        h_noise = np.random.uniform(0, 1, N)
        h_noise -= (np.sum(h_noise) - N/2) / N
        h_noise = np.clip(h_noise, 0, 1)
        # Re-adjust
        h_noise -= (np.sum(h_noise) - N/2) / N
        seeds.append(h_noise)

    # Optimize each seed
    for h_seed in seeds:
        if time.time() > deadline:
            break
            
        # Check initial value
        val_seed = compute_value(h_seed)
        if val_seed < best_val:
            best_val = val_seed
            best_h = h_seed.copy()
            
        try:
            # Optimize using smooth objective
            res = minimize(objective_smooth, h_seed, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 100, 'ftol': 1e-6})
            
            if res.success:
                h_opt = res.x
                # Ensure validity
                if np.all(h_opt >= 0) and np.all(h_opt <= 1) and abs(np.sum(h_opt) - N/2) < 1e-7:
                    val_opt = compute_value(h_opt)
                    if val_opt < best_val:
                        best_val = val_opt
                        best_h = h_opt.copy()
                        
        except Exception:
            pass

    # Final Polish
    if time.time() < deadline - 5:
        # Try to polish the best_h with exact objective and tighter tolerance
        try:
            res = minimize(objective_exact, best_h, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 200, 'ftol': 1e-8})
            if res.success:
                h_polished = res.x
                if np.all(h_polished >= 0) and np.all(h_polished <= 1) and abs(np.sum(h_polished) - N/2) < 1e-9:
                    val_polished = compute_value(h_polished)
                    if val_polished < best_val:
                        best_val = val_polished
                        best_h = h_polished.copy()
        except Exception:
            pass

    return best_h

# Test locally if needed
if __name__ == "__main__":
    h = construct()
    print("Sum:", np.sum(h))
    print("Min:", np.min(h))
    print("Max:", np.max(h))
    print("Value:", compute_value(h))
