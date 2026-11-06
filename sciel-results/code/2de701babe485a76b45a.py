import numpy as np
from math import gcd
from functools import reduce
import time

def construct():
    n = 30
    def lcm(a, b):
        return a * b // gcd(a, b)
    L = reduce(lcm, range(1, n+1))
    
    # Values: L/i for i=1..n
    vals = np.array([L // i for i in range(1, n+1)], dtype=np.int64)
    
    # Simulated Annealing
    # State: coeffs (array of -1, 0, 1)
    # Sum: integer
    
    # Initialize
    # Greedy init: try to keep sum small
    coeffs = np.zeros(n, dtype=np.int8)
    current_sum = 0
    for i in range(n):
        # Try -1, 0, 1
        best_c = 0
        best_abs = np.inf
        for c in [-1, 0, 1]:
            s = current_sum + c * vals[i]
            if s == 0:
                # Avoid 0 if possible, but keep as candidate if no other option
                # Actually, we want non-zero final. 
                # In SA, we can allow 0 intermediate.
                pass
            abs_s = abs(s)
            if abs_s < best_abs:
                best_abs = abs_s
                best_c = c
        coeffs[i] = best_c
        current_sum += best_c * vals[i]
        
    if current_sum == 0:
        # Force non-zero
        coeffs[-1] = 1
        current_sum += vals[-1]
        
    best_sum = current_sum
    best_coeffs = coeffs.copy()
    best_abs = abs(current_sum)
    
    start_time = time.time()
    timeout = 55.0
    
    T = 1e6 # Initial temperature (scale for sum)
    # Scale T based on initial sum
    if best_abs > 0:
        T = best_abs / 10.0
    else:
        T = 1e6
        
    alpha = 0.99
    
    while time.time() - start_time < timeout:
        # Pick random index
        i = np.random.randint(n)
        old_c = coeffs[i]
        
        # Pick new coefficient
        options = [-1, 0, 1]
        options.remove(old_c)
        new_c = options[np.random.randint(2)]
        
        delta = (new_c - old_c) * vals[i]
        new_sum = current_sum + delta
        new_abs = abs(new_sum)
        
        # If new_sum is 0, we must reject it if we have a non-zero best?
        # The task requires the FINAL sum to be non-zero.
        # Intermediate can be 0.
        
        if new_sum == 0:
            # Reject 0 if we already have a non-zero best?
            # Or accept with low prob?
            # Let's reject 0 to steer away.
            accept = False
        else:
            if new_abs < best_abs:
                accept = True
            elif new_abs < best_abs * 1.1: # Slight improvement
                accept = True
            else:
                # Metropolis criterion
                # If new_abs > current_abs, accept with prob exp(-(new_abs - current_abs)/T)
                # We want to minimize abs sum.
                # Delta E = new_abs - abs(current_sum)
                delta_e = new_abs - abs(current_sum)
                if delta_e < 0:
                    accept = True
                else:
                    if T > 0:
                        prob = np.exp(-delta_e / T)
                        if np.random.rand() < prob:
                            accept = True
                        else:
                            accept = False
                    else:
                        accept = False
        
        if accept:
            coeffs[i] = new_c
            current_sum = new_sum
            if new_abs < best_abs and new_sum != 0:
                best_abs = new_abs
                best_sum = new_sum
                best_coeffs = coeffs.copy()
                
        T *= alpha
        
    # Return best_coeffs
    # Verify
    total = 0
    for i, c in enumerate(best_coeffs):
        if c != 0:
            total += c * (L // (i + 1))
            
    if total == 0:
        # Fallback
        return [0] * 28 + [1, -1]
        
    return list(best_coeffs)
