import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 110  # Safety margin before 120s limit
    
    # Initial known solution
    best_M = 333312
    best_k = [336, 372, 434]
    best_score = 3
    
    # Try to find better solution with sieve up to N
    # N=10^7 is a good balance of time and coverage
    N = 10_000_000
    
    try:
        # Use int64 to avoid overflow during intermediate steps
        # sigma(k) for k=10^7 is approx 10^8, k*sigma(k) is 10^15, fits in int64
        sigma = np.zeros(N + 1, dtype=np.int64)
        
        # Sieve for sigma
        # This loop performs N vector additions.
        # Total operations ~ N log N
        for i in range(1, N + 1):
            if time.time() > deadline:
                break
            sigma[i::i] += i
        
        if time.time() > deadline:
            return {'M': int(best_M), 'k': best_k}
            
        k_vals = np.arange(1, N + 1, dtype=np.int64)
        M_vals = k_vals * sigma
        
        # Filter M <= 10^18
        # All M_vals should be < 10^18 for N=10^7
        # max M ~ 10^7 * 10^8 = 10^15
        mask = M_vals <= 10**18
        M_vals = M_vals[mask]
        k_vals = k_vals[mask]
        
        if len(M_vals) == 0:
            return {'M': int(best_M), 'k': best_k}
            
        # Find most frequent M
        unique_M, counts = np.unique(M_vals, return_counts=True)
        
        max_count = np.max(counts)
        
        if max_count > best_score:
            # Find the M with max count
            # If multiple M have same max count, pick one
            best_M_idx = np.argmax(counts)
            best_M = int(unique_M[best_M_idx])
            best_k = k_vals[M_vals == best_M].tolist()
            best_score = int(max_count)
            
        # Optional: If time permits, try larger N?
        # For now, stick to one run to ensure safety.
        # If we had 150s, we could try N=2*10^7 if first run took < 30s.
        
    except Exception:
        pass

    return {'M': int(best_M), 'k': best_k}
