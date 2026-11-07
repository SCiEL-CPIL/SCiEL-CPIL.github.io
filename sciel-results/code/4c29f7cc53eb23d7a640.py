"""Combined program: vectorized sigma sieve + collision grouping."""
import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    # Known seed solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Strategy: 
    # 1. Use a vectorized sieve to compute sigma(k) for k = 1..K
    # 2. Compute M_k = k * sigma(k) for all k
    # 3. Group by M_k value and find the M with most preimages
    # 4. Verify k <= sqrt(M) constraint (should hold for k > 1 since sigma(k) >= k+1)
    
    # We'll try to push K as high as possible within the time budget.
    # The sieve is O(K log K), grouping is O(K).
    # Memory: sigma array of size K+1 (int64), M_vals of size K (int64)
    
    # Try a sequence of increasing K values, stopping when we run out of time
    # or when we've used too much memory
    
    K_values = [10_000_000, 20_000_000, 30_000_000, 50_000_000, 80_000_000, 120_000_000]
    
    for K in K_values:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:  # Need at least 30s for sieve + grouping
            break
        
        try:
            # Sieve for sigma using NumPy
            sigma = np.zeros(K + 1, dtype=np.int64)
            for j in range(1, K + 1):
                if time.time() - start_time > deadline - 15:
                    break
                sigma[j::j] += j
            
            # Check if we completed the sieve
            if time.time() - start_time > deadline - 15:
                break
            
            # Compute M_k = k * sigma[k]
            k_vals = np.arange(1, K + 1, dtype=np.int64)
            M_vals = k_vals * sigma[1:]
            
            # Filter: M <= 10^18
            valid_mask = M_vals <= 10**18
            M_valid = M_vals[valid_mask]
            k_valid = k_vals[valid_mask]
            
            # Use np.unique for efficient grouping (much faster than argsort+diff)
            if len(M_valid) > 0:
                M_unique, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
                
                # Find the maximum count
                max_count = int(counts.max())
                if max_count <= best_count:
                    continue
                
                # Find all M values that achieve the max count
                best_indices = np.where(counts == max_count)[0]
                
                # For each best M, extract the k values and check k <= sqrt(M)
                for idx in best_indices:
                    M_val = int(M_unique[idx])
                    sqrt_M = math.isqrt(M_val)
                    
                    # Get all k values that map to this M
                    mask = (inverse == idx)
                    ks = k_valid[mask]
                    
                    # Filter k <= sqrt(M)
                    valid_ks = ks[ks <= sqrt_M]
                    
                    if len(valid_ks) > best_count:
                        best_count = len(valid_ks)
                        best_M = M_val
                        best_k = [int(k) for k in valid_ks]
                    
                    # Check time
                    if time.time() - start_time > deadline - 10:
                        break
                
        except MemoryError:
            break
        except Exception:
            break
    
    # Final return
    return {'M': int(best_M), 'k': [int(k) for k in best_k]}
