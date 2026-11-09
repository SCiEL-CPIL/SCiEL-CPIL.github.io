import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin from 120s limit
    
    # Seed solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Try increasing sizes
    # 50 million should be feasible in memory and time for NumPy
    limits = [10_000_000, 30_000_000, 50_000_000]
    
    for N in limits:
        if time.time() - start_time > deadline - 30.0:
            break
            
        try:
            # Sieve for sigma
            # Using int64 to hold sigma values.
            sigma = np.zeros(N + 1, dtype=np.int64)
            for i in range(1, N + 1):
                sigma[i::i] += i
                
            if time.time() - start_time > deadline - 15.0:
                continue

            k_arr = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_arr * sigma[1:]
            
            # Filter M <= 10^18
            mask = M_vals <= 10**18
            if not np.any(mask):
                continue
                
            M_valid = M_vals[mask]
            k_valid = k_arr[mask]
            
            # Find unique M values and their counts
            unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
            
            # Find the M with the maximum count
            max_idx = np.argmax(counts)
            current_max_count = counts[max_idx]
            
            if current_max_count > best_count:
                best_M = int(unique_M[max_idx])
                # Extract k values corresponding to this M
                mask_best = (inverse == max_idx)
                best_k = k_valid[mask_best].tolist()
                best_count = current_max_count
                
        except MemoryError:
            # If memory runs out, break and return best so far
            break
        except Exception:
            # If other errors occur, break and return best so far
            break

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    return {'M': best_M, 'k': best_k}
