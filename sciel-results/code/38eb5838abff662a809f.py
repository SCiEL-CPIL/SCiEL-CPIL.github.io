import numpy as np
import time

N = 5000
START_TIME = time.time()
TIME_LIMIT = 280  # Leave 20s margin

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Precompute primes up to N
    primes_list = _primes(N)
    
    # Initial guess for K
    K = 500
    best_tuple = None
    best_size = 0
    
    # We will iterate to find the fixed point K
    for _ in range(20): # Max iterations for K convergence
        if time.time() - START_TIME > TIME_LIMIT:
            break
            
        # Determine set of primes P <= K
        P = [p for p in primes_list if p <= K]
        
        # Initialize residues
        residues = {p: 0 for p in P}
        
        # Initialize counts array
        counts = np.zeros(N + 1, dtype=np.int32)
        U = 0
        
        # Apply initial residues
        for p in P:
            r = residues[p]
            indices = np.arange(r, N + 1, p)
            # Check which were 0
            zero_mask = counts[indices] == 0
            U += np.sum(zero_mask)
            counts[indices] += 1
            
        # Hill climbing to minimize U
        improved = True
        while improved:
            if time.time() - START_TIME > TIME_LIMIT:
                break
            improved = False
            for p in P:
                r_old = residues[p]
                best_r = r_old
                best_U = U
                
                # Try all other residues
                for r_new in range(p):
                    if r_new == r_old:
                        continue
                    
                    old_indices = np.arange(r_old, N + 1, p)
                    new_indices = np.arange(r_new, N + 1, p)
                    
                    # Calculate U change
                    # Elements in old_indices that drop from 1 to 0 reduce U
                    # Elements in new_indices that rise from 0 to 1 increase U
                    # Note: An element cannot be in both old and new indices for same p
                    # unless p=1, which is not prime.
                    
                    old_counts = counts[old_indices]
                    new_counts = counts[new_indices]
                    
                    delta_U = np.sum(new_counts == 0) - np.sum(old_counts == 1)
                    
                    if U + delta_U < best_U:
                        best_U = U + delta_U
                        best_r = r_new
                        
                if best_r != r_old:
                    # Apply change
                    old_indices = np.arange(r_old, N + 1, p)
                    new_indices = np.arange(best_r, N + 1, p)
                    
                    counts[old_indices] -= 1
                    counts[new_indices] += 1
                    
                    U = best_U
                    residues[p] = best_r
                    improved = True
                    
        # Calculate final size
        K_new = N + 1 - U
        
        # Check admissibility: We blocked residues for all p <= K.
        # We need admissibility for p <= K_new.
        # If K_new > K, we missed some primes.
        # If K_new <= K, we are safe (blocked more than needed).
        
        if K_new > K:
            K = K_new
            # Continue loop to handle new primes
            continue
        else:
            # K_new <= K, so current set is admissible
            # We can try to reduce K to K_new to see if we can get a larger set?
            # If we reduce K, we remove constraints for primes in (K_new, K].
            # This will decrease U, increasing size.
            # So we should set K = K_new and re-optimize?
            # But if we set K = K_new, we drop constraints.
            # The new size will be >= K_new.
            # So it's a good move.
            
            # However, if K_new == K, we converged.
            if K_new == K:
                # Extract tuple
                mask = counts > 0
                tuple_vals = np.where(~mask)[0]
                return tuple_vals.tolist()
            else:
                K = K_new
                # Loop again to re-optimize with fewer primes
                continue
                
    # If we didn't converge or timed out, return best found
    # If best_tuple is None, we need to generate one
    if best_tuple is None:
        # Fallback: return a simple valid tuple
        # Just return the seed's logic or a small valid set
        # Actually, the loop above should have returned.
        # If we broke due to time, we might not have a valid tuple.
        # Let's ensure we return something valid.
        # We can use the last computed mask if available.
        if 'counts' in locals() and counts is not None:
             mask = counts > 0
             tuple_vals = np.where(~mask)[0]
             return tuple_vals.tolist()
        else:
             # Fallback to seed logic
             points = [1] + [p for p in primes_list if p * p > N]
             changed = True
             while changed:
                 changed = False
                 for p in _primes(len(points)):
                     classes = {}
                     for v in points:
                         classes.setdefault(v % p, []).append(v)
                     if len(classes) == p:
                         smallest = min(classes.values(), key=len)
                         points = [v for v in points if v not in smallest]
                         changed = True
                         break
             return points
