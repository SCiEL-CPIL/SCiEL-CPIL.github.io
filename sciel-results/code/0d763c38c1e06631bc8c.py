import numpy as np

def construct():
    # Generate primes up to 1000
    primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
    n_primes = len(primes)
    
    MAX_L = 3000
    x = np.arange(1, MAX_L + 1)
    
    coverage = np.zeros(MAX_L, dtype=np.int32)
    residues = np.zeros(n_primes, dtype=np.int64)
    
    def compute_L(cov):
        bad_indices = np.nonzero(cov < 2)[0]
        if len(bad_indices) == 0:
            return MAX_L
        return bad_indices[0]

    # Phase 1: Greedy Assignment targeting the bottleneck
    for i in range(n_primes):
        p = primes[i]
        
        # Target the first uncovered integer
        current_L = compute_L(coverage)
        
        if current_L >= MAX_L:
            # Already covered up to MAX_L, pick arbitrary
            r_best = 0
        else:
            x_fail = current_L + 1
            r_best = x_fail % p
            
        residues[i] = r_best
        mask = (x % p) == r_best
        coverage[mask] += 1
        
    # Phase 2: Targeted Local Search
    L = compute_L(coverage)
    
    # We will try to extend L by fixing the bottleneck x_fail = L + 1
    # We look for a prime to change.
    
    for iteration in range(500):
        if L >= MAX_L:
            break
            
        x_fail = L + 1
        improved = False
        
        # Iterate through primes. 
        # We can prioritize primes that currently cover x_fail? 
        # No, we want to CHANGE a prime to cover x_fail.
        # So we look for primes where r_old != x_fail % p.
        
        # To improve speed, we can shuffle primes or use a fixed order.
        # Let's use a fixed order but maybe reverse (larger primes first) 
        # because they have fewer points in [1, L] and are "easier" to change safely?
        # Actually, smaller primes have more points, so harder to be safe.
        # Larger primes are safer to change.
        
        for i in range(n_primes - 1, -1, -1):
            p = primes[i]
            r_old = residues[i]
            r_new = x_fail % p
            
            if r_new == r_old:
                continue
                
            # Check safety: 
            # For all x in [1, L] with x % p == r_old, coverage[x] must be >= 3.
            # mask_old: x in 1..MAX_L where x % p == r_old
            # We only care about x <= L.
            
            mask_old = (x % p) == r_old
            mask_L = x <= L
            mask_check = mask_old & mask_L
            
            # If any coverage in mask_check is 2, it's unsafe.
            if np.any(coverage[mask_check] == 2):
                continue
            
            # Safe. Apply change.
            residues[i] = r_new
            mask_new = (x % p) == r_new
            
            coverage[mask_old] -= 1
            coverage[mask_new] += 1
            
            new_L = compute_L(coverage)
            
            if new_L > L:
                L = new_L
                improved = True
                break # Restart search with new L
            else:
                # Revert
                coverage[mask_old] += 1
                coverage[mask_new] -= 1
                residues[i] = r_old
        
        if not improved:
            break
            
    return [int(r) for r in residues]
