import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    # Use a smaller N to ensure the sieve finishes quickly and leaves time for search
    N = 2000000
    
    # Compute sigma(n) for n = 1..N using additive sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        if d % 100000 == 0:
            if time.time() - start_time > 45:
                break
        sigma[d::d] += d
    
    elapsed = time.time() - start_time
    if elapsed > 50:
        return [11, 6]
        
    # Build dictionary of fibers: sigma_value -> list of indices
    fibers = {}
    for i in range(1, N + 1):
        s = int(sigma[i])
        if s in fibers:
            fibers[s].append(i)
        else:
            fibers[s] = [i]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Search for best pair in each fiber
    for s, indices in fibers.items():
        # indices is sorted
        for b in indices:
            target = 1.5 * b
            # Find position in indices closest to target
            idx = bisect.bisect_left(indices, int(target))
            
            candidates = set()
            if idx < len(indices):
                candidates.add(indices[idx])
            if idx > 0:
                candidates.add(indices[idx-1])
            if idx + 1 < len(indices):
                candidates.add(indices[idx+1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = num / den
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
