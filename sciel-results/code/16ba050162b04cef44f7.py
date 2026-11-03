import numpy as np
import time
import random

N = 400

def calculate_max_autocorr(a):
    """
    Calculates the maximum absolute value of the autocorrelation
    for lags k = 1 to N-1.
    """
    # a is a numpy array of +/- 1
    # Autocorrelation R(k) = sum_{i=0}^{N-1-k} a[i] * a[i+k]
    # We want max |R(k)| for k > 0
    
    # Efficient calculation using FFT is overkill for N=400, 
    # but let's use a simple loop or vectorized sum for speed in the search loop.
    # Actually, for N=400, a pure Python loop over k and inner sum is slow.
    # Vectorized: 
    # R(k) = np.dot(a[:-k], a[k:])
    
    max_val = 0
    for k in range(1, N):
        # Compute correlation at lag k
        # a[:-k] and a[k:] have length N-k
        r_k = np.dot(a[:-k], a[k:])
        if abs(r_k) > max_val:
            max_val = abs(r_k)
    return max_val

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    # Initialize with a random sequence
    # Or a Mersenne sequence? Random is easier to implement correctly.
    # Mersenne sequences have good autocorrelation properties.
    # Let's try random first, then optimize.
    
    best_a = np.random.choice([-1, 1], size=N)
    best_score = calculate_max_autocorr(best_a)
    
    # Simulated Annealing
    T = 10.0
    T_min = 0.1
    cooling = 0.999
    
    current_a = best_a.copy()
    current_score = best_score
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick a random position to flip
        i = random.randint(0, N - 1)
        
        # Flip
        current_a[i] *= -1
        
        # Calculate new score
        new_score = calculate_max_autocorr(current_a)
        
        delta = new_score - current_score
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_a = current_a.copy()
        else:
            if random.random() < np.exp(-delta / T):
                current_score = new_score
        
        T *= cooling
        
    return best_a

# EVOLVE-BLOCK-START
# The code above is the implementation.
# We need to wrap it in the required format.
# The prompt asks to keep function names.
# The seed had N=400 and construct().

# Let's refine the search to be more effective.
# Simple hill climbing gets stuck quickly.
# Simulated annealing is better.
# We can also try multiple restarts if time permits.

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_a = None
    best_score = float('inf')
    
    # Number of restarts
    max_restarts = 20
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize
        # Try different initializations: random, or based on previous best?
        # Random is good for diversity.
        a = np.random.choice([-1, 1], size=N)
        
        # Local search / Simulated Annealing
        T = 5.0
        T_min = 0.5
        cooling = 0.9995
        
        current_score = calculate_max_autocorr(a)
        
        # Run SA for a fixed number of steps or time
        steps = 0
        max_steps = 50000 # Adjust based on speed
        
        while steps < max_steps:
            if time.time() - start_time > time_limit:
                break
                
            i = random.randint(0, N - 1)
            a[i] *= -1
            new_score = calculate_max_autocorr(a)
            delta = new_score - current_score
            
            if delta < 0:
                current_score = new_score
            else:
                if T > T_min and random.random() < np.exp(-delta / T):
                    current_score = new_score
                else:
                    a[i] *= -1 # Revert
                    T *= cooling
                    steps += 1
                    continue
            
            T *= cooling
            steps += 1
            
        if current_score < best_score:
            best_score = current_score
            best_a = a.copy()
            
    if best_a is None:
        # Fallback to seed if something went wrong
        return np.ones(N)
        
    return best_a

# EVOLVE-BLOCK-END
