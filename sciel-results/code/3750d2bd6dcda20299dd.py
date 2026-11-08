import numpy as np
from scipy.optimize import minimize
import time

N = 600
DELTA = 1.0 / 1200.0 # Interval width

def objective(h):
    # h is a vector of length N
    # Compute discrete convolution
    # Using np.convolve with mode='full' gives length 2N-1
    # The "max" should probably be over the central part or the whole?
    # The problem says "on [-1/4, 1/4]".
    # If the function is supported there, the convolution is supported on [-1/2, 1/2].
    # The evaluator likely computes the convolution of the step function.
    # I will compute the full convolution and take the max.
    conv = np.convolve(h, h, mode='full')
    # Scale by delta?
    # If h are heights, the integral is sum(h)*delta.
    # The convolution value is sum(h_i h_j) * delta.
    # So max_conv = np.max(conv) * DELTA
    # Norm_sq = sum(h^2) * DELTA
    # Ratio = (max_conv) / (norm_sq) = (max(conv)*DELTA) / (sum(h^2)*DELTA) = max(conv) / sum(h^2)
    # The DELTA cancels out!
    # So we just minimize max(conv) / sum(h^2)
    
    ss = np.sum(h**2)
    if ss < 1e-10:
        return 1e10
    return np.max(conv) / ss

def construct():
    start_time = time.time()
    deadline = start_time + 100.0 # 100s guard
    
    # Initial guesses
    best_score = 1.0
    best_h = np.ones(N)
    
    # Try uniform
    h0 = np.ones(N)
    score = objective(h0)
    if score < best_score:
        best_score = score
        best_h = h0.copy()
        
    # Try random
    np.random.seed(42)
    for i in range(5):
        h_rand = np.random.rand(N)
        # Normalize to avoid overflow/underflow issues in optimizer?
        # L-BFGS-B handles bounds, but objective is scale invariant.
        # To help convergence, we can fix sum(h^2)=1?
        # No, L-BFGS-B is good for this.
        
        # Try a "two peak" guess
        h_two = np.zeros(N)
        h_two[0] = 1.0
        h_two[N-1] = 1.0
        
        guesses = [h0, h_rand, h_two]
        
        for g in guesses:
            if time.time() > deadline:
                break
            try:
                res = minimize(objective, g, method='L-BFGS-B', 
                               bounds=[(0, None)]*N, 
                               options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8})
                if res.success:
                    s = res.fun
                    if s < best_score:
                        best_score = s
                        best_h = res.x.copy()
            except Exception:
                pass

    return list(best_h)
