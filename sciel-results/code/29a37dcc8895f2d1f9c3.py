import math
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback: Known 3-way collision
    best_M = 333312
    best_k = [336, 372, 434]

    # Strategy:
    # 1. Use a NumPy-based sieve to compute k * sigma(k) for k up to N.
    # 2. Find the M value that occurs most frequently in the resulting array.
    # 3. Verify that M <= 10^18 and all corresponding k <= sqrt(M).
    # 4. If a better collision is found (more k's), update the best result.
    
    # We try a few sizes of N to balance memory and time.
    # N=10^7 takes ~1-2s and ~80MB.
    # N=2*10^7 takes ~2-4s and ~160MB.
    # N=5*10^7 takes ~5-10s and ~400MB.
    
    # Let's try up to 20 million first, as it's a safe bet for time and memory.
    # If time remains, we could try larger, but let's stick to one robust pass 
    # to avoid complexity and potential timeouts from repeated large allocations.
    
    N = 20_000_000
    
    try:
        # Allocate sigma array
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        # sigma[i::i] += i is a vectorized operation
        for i in range(1, N + 1):
            if time.time() > deadline - 10.0:
                raise TimeoutError()
            sigma[i::i] += i
            
        # Compute M values: k * sigma(k)
        # k is 1..N
        ks = np.arange(1, N + 1, dtype=np.int64)
        Ms = ks * sigma
        
        # Free memory
        del sigma
        del ks
        
        if time.time() > deadline - 5.0:
            raise TimeoutError()
            
        # Find the most frequent M
        # np.unique is efficient enough for this size
        vals, counts, _ = np.unique(Ms, return_counts=True, return_index=True)
        
        max_count = np.max(counts)
        
        # Only update if we found strictly more solutions than the current best
        if max_count > len(best_k):
            max_val_idx = np.argmax(counts)
            new_M = int(vals[max_val_idx])
            
            # Check M constraint
            if new_M <= 10**18:
                # Find all k that produced this M
                mask = Ms == new_M
                new_ks = np.where(mask)[0] + 1
                
                # Free memory
                del Ms
                del vals
                del counts
                del mask
                
                # Filter k <= sqrt(M)
                sqrt_M = math.isqrt(new_M)
                valid_ks = [int(k) for k in new_ks if k <= sqrt_M]
                
                # We need to ensure distinctness (should be distinct by index, but just in case)
                # And verify exactness? The sieve is exact for integers within range.
                # int64 can hold up to ~9e18, so k*sigma(k) for k=20M (sigma ~ 20M*6/2=60M? no sigma is smaller)
                # Max sigma(k) for k=20M is roughly k * log(log(k)) ~ 20M * 2.5 = 50M.
                # k * sigma(k) ~ 10^15, well within int64.
                
                if len(valid_ks) > len(best_k):
                    best_M = new_M
                    best_k = valid_ks
                    
    except Exception:
        pass

    # Final check and cleanup
    if best_M > 10**18:
        best_M = 333312
        best_k = [336, 372, 434]
        
    sqrt_M = math.isqrt(best_M)
    valid_k = [k for k in best_k if 1 <= k <= sqrt_M]
    
    if len(valid_k) < len(best_k):
        best_k = valid_k
        
    if len(best_k) == 0:
        best_M = 333312
        best_k = [336, 372, 434]

    # Ensure types are exact Python ints
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
