import random
import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin for 90s limit
    
    n = 30
    # Precompute LCM of 1..30
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    D = 1
    for i in range(1, n + 1):
        D = lcm(D, i)
        
    # Precompute weights w[i] = D // (i+1) for i in 0..n-1
    # Sum = sum(c[i] * w[i]) / D
    weights = [D // (i + 1) for i in range(n)]
    
    best_S = None
    best_coeff = [0] * n
    
    # Helper to update best
    def update_best(S, coeff):
        nonlocal best_S, best_coeff
        if S == 0:
            return
        if best_S is None or abs(S) < abs(best_S):
            best_S = S
            best_coeff = coeff[:]

    # Initial random search to get a starting point
    # We want a small nonzero sum. 
    # Random might give large sums. 
    # Let's try a few random restarts and then local search.
    
    num_restarts = 1000
    max_local_steps = 100
    
    while time.time() < deadline:
        # Random initialization
        # Bias towards more non-zeros? Or just uniform?
        # Uniform is fine, local search will fix it.
        coeff = [random.choice([-1, 0, 1]) for _ in range(n)]
        
        # Calculate initial S
        S = 0
        for i in range(n):
            if coeff[i] != 0:
                S += coeff[i] * weights[i]
                
        if S == 0:
            # If zero, flip a random bit to make it nonzero
            idx = random.randint(0, n - 1)
            coeff[idx] = 1 if coeff[idx] != 1 else -1
            S += coeff[idx] * weights[idx]
            
        update_best(S, coeff)
        
        # Local Search
        improved = True
        steps = 0
        while improved and steps < max_local_steps and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            for i in indices:
                current_val = coeff[i]
                # Try all other values in {-1, 0, 1}
                for new_val in [-1, 0, 1]:
                    if new_val == current_val:
                        continue
                    delta = (new_val - current_val) * weights[i]
                    new_S = S + delta
                    if new_S != 0 and abs(new_S) < abs(S):
                        # Accept
                        coeff[i] = new_val
                        S = new_S
                        improved = True
                        update_best(S, coeff)
                        break # Move to next index after an improvement
            steps += 1

    # Convert best_coeff to return
    # The task asks for exactly 30 integer coefficients.
    # We already have that.
    return best_coeff

# EVOLVE-BLOCK-END
