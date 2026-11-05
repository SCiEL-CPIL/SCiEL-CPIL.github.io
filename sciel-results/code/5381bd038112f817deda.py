import time
import numpy as np
from math import gcd, lcm
import random
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s limit
    
    n = 30
    # LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Integer weights: w_i = L // (i+1) for coefficient c_i multiplying 1/(i+1)
    # Sum = sum(c_i * L/(i+1)) / L
    # Minimize |sum(c_i * w_i)| where w_i = L//(i+1)
    weights = [L // (i + 1) for i in range(n)]
    
    def compute_value(coeffs):
        """Compute the integer numerator K = sum(c_i * w_i)"""
        return sum(c * w for c, w in zip(coeffs, weights))
    
    def value_from_K(K):
        """Compute |K/L| as a float for comparison"""
        if K == 0:
            return float('inf')
        return abs(K) / L
    
    # Simulated Annealing
    random.seed(42)
    
    # Initial solution: random coefficients
    best_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
    best_K = compute_value(best_coeffs)
    best_val = value_from_K(best_K)
    
    # Make sure initial is nonzero
    while best_K == 0:
        best_coeffs = [random.choice([-1, 0, 1]) for _ in range(n)]
        best_K = compute_value(best_coeffs)
        best_val = value_from_K(best_K)
    
    current_coeffs = best_coeffs[:]
    current_K = best_K
    current_val = best_val
    
    T = 1.0  # Initial temperature
    T_min = 1e-15
    cooling = 0.9999
    
    max_iter = 5000000  # Max iterations
    
    iter_count = 0
    while time.time() < deadline - 2.0 and iter_count < max_iter:
        # Random move: flip one coefficient
        i = random.randint(0, n - 1)
        old_val = current_coeffs[i]
        new_val = random.choice([-1, 0, 1])
        while new_val == old_val:
            new_val = random.choice([-1, 0, 1])
        
        # Compute delta in K
        delta_K = (new_val - old_val) * weights[i]
        new_K = current_K + delta_K
        new_val = value_from_K(new_K)
        
        # Accept if better or with probability exp(-delta/T)
        if new_val < current_val:
            # Better, always accept
            current_coeffs[i] = new_val
            current_K = new_K
            current_val = new_val
            
            if new_val < best_val:
                best_val = new_val
                best_K = new_K
                best_coeffs = current_coeffs[:]
        else:
            # Worse, accept with probability
            delta = new_val - current_val
            if delta < T:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_coeffs[i] = new_val
                    current_K = new_K
                    current_val = new_val
        
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
        
        iter_count += 1
    
    # Verify best solution
    final_K = compute_value(best_coeffs)
    if final_K == 0:
        # Fallback: try a simple nonzero solution
        fallback = [0] * 28 + [1, -1]
        return fallback
    
    return best_coeffs
