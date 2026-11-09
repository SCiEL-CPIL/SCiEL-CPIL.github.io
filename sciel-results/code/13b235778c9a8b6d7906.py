import time
import numpy as np

N = 5000
DEADLINE = 295.0 # Safety margin
START_TIME = time.time()

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

PRIMES = _primes(N + 1)

def construct():
    global START_TIME
    if time.time() - START_TIME > DEADLINE:
        return list(range(N + 1))
    
    best_size = 0
    best_set = []
    
    # Define strategies
    # 0: min index
    # 1: random
    # 2: min std (remove concentrated)
    # 3: max std (remove spread)
    
    strategies = ['min', 'random', 'min_std', 'max_std']
    
    # Run multiple restarts
    # Each run is fast (N=5000, ~669 primes, vectorized ops)
    # Estimate: 1 run < 0.1s. We can do ~2000 runs in 300s?
    # Let's aim for 500-1000 runs to be safe.
    
    runs = 0
    max_runs = 1000
    
    while time.time() - START_TIME < DEADLINE and runs < max_runs:
        for strat in strategies:
            if time.time() - START_TIME > DEADLINE:
                break
            
            mask = np.ones(N + 1, dtype=bool)
            current_size = N + 1
            indices = np.arange(N + 1)
            
            for p in PRIMES:
                if p > current_size:
                    break
                
                # Get active values
                active_vals = indices[mask]
                if len(active_vals) == 0:
                    break
                
                residues = active_vals % p
                counts = np.bincount(residues, minlength=p)
                
                if counts.min() > 0:
                    min_count = counts.min()
                    candidates = np.where(counts == min_count)[0]
                    
                    if strat == 'min':
                        chosen = candidates[0]
                    elif strat == 'random':
                        chosen = np.random.choice(candidates)
                    elif strat == 'min_std':
                        # Find candidate with smallest std
                        best_c = candidates[0]
                        best_std = float('inf')
                        for c in candidates:
                            vals = active_vals[residues == c]
                            if len(vals) > 1:
                                std = np.std(vals)
                            else:
                                std = 0
                            if std < best_std:
                                best_std = std
                                best_c = c
                        chosen = best_c
                    elif strat == 'max_std':
                        best_c = candidates[0]
                        best_std = -1
                        for c in candidates:
                            vals = active_vals[residues == c]
                            if len(vals) > 1:
                                std = np.std(vals)
                            else:
                                std = 0
                            if std > best_std:
                                best_std = std
                                best_c = c
                        chosen = best_c
                    else:
                        chosen = candidates[0]
                        
                    # Remove
                    mask[indices % p == chosen] = False
                    current_size = mask.sum()
            
            if current_size > best_size:
                best_size = current_size
                best_set = indices[mask].tolist()
        
        runs += 1
        
    return best_set
