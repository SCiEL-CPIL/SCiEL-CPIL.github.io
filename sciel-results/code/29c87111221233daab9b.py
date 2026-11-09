import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    try:
        # Try increasing N until time runs out or we find a very good solution
        # Start with a moderate N to ensure we get *some* result quickly
        Ns = [5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]
        
        for N in Ns:
            if time.time() > deadline - 20: # Need time for processing
                break
                
            # Compute sigma using NumPy
            # sigma = np.ones(N+1, dtype=np.int64)
            # for d in range(1, N+1):
            #     sigma[d::d] += d
            
            # This loop is O(N log N) but with high constant in Python?
            # No, the inner op is vectorized.
            # However, the loop over d is Python.
            # For N=1e7, 1e7 iterations. ~1s?
            # Let's estimate.
            
            sigma = np.ones(N + 1, dtype=np.int64)
            for d in range(1, N + 1):
                sigma[d::d] += d
                
            if time.time() > deadline - 10:
                break
            
            # Compute M = k * sigma(k)
            ks = np.arange(1, N + 1, dtype=np.int64)
            Ms = ks * sigma[1:]
            
            # Filter M <= 10^18
            # Since k <= N, sigma(k) <= ~k log log k.
            # For N=1e7, M ~ 1e7 * 1e7 * 20 = 2e15 < 1e18.
            # For N=1e8, M ~ 1e8 * 1e8 * 25 = 2.5e17 < 1e18.
            # So filter is mostly for safety.
            valid_mask = Ms <= 10**18
            Ms_valid = Ms[valid_mask]
            ks_valid = ks[valid_mask]
            
            if len(Ms_valid) == 0:
                continue
                
            # Find most frequent M
            # np.unique is fast
            unique_Ms, counts = np.unique(Ms_valid, return_counts=True)
            
            max_idx = np.argmax(counts)
            max_count = counts[max_idx]
            
            if max_count > best_count:
                best_M = int(unique_Ms[max_idx])
                # Find all k that map to best_M
                best_k = ks_valid[Ms_valid == best_M].tolist()
                # Ensure distinct (they are from unique ks)
                # Convert to Python ints
                best_k = [int(k) for k in best_k]
                best_count = max_count
                # If we found a very high count, we might stop early?
                # But larger N might find even higher.
                
            # Check time
            if time.time() > deadline - 5:
                break
                
    except Exception:
        pass
        
    return {'M': int(best_M), 'k': best_k}
