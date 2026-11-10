import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit, 20s buffer
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    # Try increasing N
    # We need to be careful with memory.
    # N=10^7 int64 is 80MB.
    # N=10^8 int64 is 800MB. Might be too much.
    
    target_Ns = [10**6, 2*10**6, 5*10**6, 10**7]
    
    for N in target_Ns:
        if time.time() > deadline - 10:
            break
            
        # Compute sigma
        sigma = np.ones(N+1, dtype=np.int64)
        # Optimization: Only loop up to N? Yes.
        # The loop `for d in range(1, N+1): sigma[d::d] += d`
        
        # This loop is the bottleneck.
        # Let's try to run it.
        
        try:
            for d in range(1, N+1):
                sigma[d::d] += d
                
            ks = np.arange(1, N+1, dtype=np.int64)
            Ms = ks * sigma[1:]
            
            # Filter M <= 10^18
            max_M = 10**18
            valid_mask = Ms <= max_M
            if not np.any(valid_mask):
                continue
                
            valid_ks = ks[valid_mask]
            valid_Ms = Ms[valid_mask]
            
            # Find most frequent M
            # np.unique is O(N log N)
            unique_Ms, counts = np.unique(valid_Ms, return_counts=True)
            
            if len(counts) == 0:
                continue
                
            max_count_idx = np.argmax(counts)
            current_count = counts[max_count_idx]
            
            if current_count > best_count:
                target_M = unique_Ms[max_count_idx]
                # Find all k for this M
                # valid_Ms == target_M is O(N)
                target_ks = valid_ks[valid_Ms == target_M]
                
                best_M = int(target_M)
                best_ks = [int(k) for k in target_ks]
                best_count = int(current_count)
                
        except MemoryError:
            break
        except Exception:
            break

    return {'M': best_M, 'k': best_ks}
