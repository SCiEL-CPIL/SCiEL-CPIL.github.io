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
    
    # Strategies:
    # 'min_idx': min index among min-count candidates
    # 'random': random among min-count candidates
    # 'min_var': min variance among min-count candidates
    # 'max_var': max variance among min-count candidates
    # 'min_range': min range among min-count candidates
    # 'max_range': max range among min-count candidates
    
    strategies = ['min_idx', 'random', 'min_var', 'max_var', 'min_range', 'max_range']
    
    runs = 0
    # We can afford more runs if each is faster
    max_runs = 3000 
    
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
                
                active_vals = indices[mask]
                if len(active_vals) == 0:
                    break
                
                residues = active_vals % p
                counts = np.bincount(residues, minlength=p)
                
                if counts.min() > 0:
                    min_count = counts.min()
                    candidates = np.where(counts == min_count)[0]
                    
                    if strat == 'min_idx':
                        chosen = candidates[0]
                    elif strat == 'random':
                        chosen = np.random.choice(candidates)
                    elif strat in ['min_var', 'max_var']:
                        # Vectorized variance calculation
                        # Only compute for candidates to save time?
                        # Or compute for all and pick best candidate.
                        # Computing for all is O(p + N_active).
                        # N_active is ~5000. p is small.
                        # Let's compute for all residues.
                        
                        sum_vals = np.bincount(residues, weights=active_vals, minlength=p)
                        sum_sq_vals = np.bincount(residues, weights=active_vals**2, minlength=p)
                        
                        # Avoid division by zero for empty classes (though we know counts>0 for candidates)
                        # But some classes might have count 0.
                        with np.errstate(divide='ignore', invalid='ignore'):
                            means = np.where(counts > 0, sum_vals / counts, 0)
                            variances = np.where(counts > 0, (sum_sq_vals / counts) - means**2, 0)
                        
                        # Only consider candidates
                        cand_variances = variances[candidates]
                        
                        if strat == 'min_var':
                            best_idx = np.argmin(cand_variances)
                        else: # max_var
                            best_idx = np.argmax(cand_variances)
                        
                        chosen = candidates[best_idx]
                        
                    elif strat in ['min_range', 'max_range']:
                        # Compute min and max for each residue
                        # This is harder to vectorize simply without loops or advanced bincount tricks.
                        # For now, let's use a simpler proxy or skip if too slow.
                        # Let's try a quick loop for candidates only, as candidates are usually few.
                        best_c = candidates[0]
                        best_range = float('inf') if strat == 'min_range' else -1
                        
                        for c in candidates:
                            vals = active_vals[residues == c]
                            if len(vals) > 0:
                                r = vals.max() - vals.min()
                                if strat == 'min_range':
                                    if r < best_range:
                                        best_range = r
                                        best_c = c
                                else:
                                    if r > best_range:
                                        best_range = r
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
