import numpy as np
import time

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = 280.0 # 20s margin
    
    all_primes = _primes(N)
    small_primes = [2, 3, 5, 7, 11]
    
    best_S = []
    best_size = -1
    
    # Pre-compute indices for all numbers
    indices = np.arange(N + 1)
    
    # Iterate over all combinations of missing residues for small primes
    # 2*3*5*7*11 = 2310 combinations
    for r2 in range(2):
        if time.time() > deadline: break
        for r3 in range(3):
            if time.time() > deadline: break
            for r5 in range(5):
                if time.time() > deadline: break
                for r7 in range(7):
                    if time.time() > deadline: break
                    for r11 in range(11):
                        if time.time() > deadline: break
                        
                        # Create mask
                        mask = np.ones(N + 1, dtype=bool)
                        
                        # Apply constraints for small primes
                        # We want x NOT in r_p mod p
                        mask[indices % 2 == r2] = False
                        mask[indices % 3 == r3] = False
                        mask[indices % 5 == r5] = False
                        mask[indices % 7 == r7] = False
                        mask[indices % 11 == r11] = False
                        
                        S = indices[mask]
                        
                        if len(S) <= best_size:
                            continue # Cannot beat best, skip repair
                            
                        # Repair for larger primes
                        # We need to ensure for all p <= len(S), some residue is missed.
                        # We already handled p in small_primes.
                        # We need to handle p in all_primes where p > 11 and p <= len(S).
                        
                        changed = True
                        while changed:
                            if time.time() > deadline:
                                break
                                
                            K = len(S)
                            if K <= 11:
                                break # No primes > 11 to check. Set is admissible.
                                
                            # Find relevant primes > 11 and <= K
                            active_primes = [p for p in all_primes if 11 < p <= K]
                            
                            if not active_primes:
                                break
                                
                            found_bad = False
                            for p in active_primes:
                                residues = S % p
                                counts = np.bincount(residues, minlength=p)
                                
                                if np.all(counts > 0):
                                    # This prime p is "bad" (all residues covered)
                                    # Remove the residue with min count
                                    min_r = np.argmin(counts)
                                    mask_remove = (S % p) != min_r
                                    S = S[mask_remove]
                                    found_bad = True
                                    break # Restart while loop with new S
                            
                            if not found_bad:
                                changed = False
                                
                        if len(S) > best_size:
                            best_size = len(S)
                            best_S = S.tolist()
                            
    return best_S
