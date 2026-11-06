import numpy as np

def construct():
    # Limit for powerful numbers
    # We want to maximize the number of representations.
    # Larger M -> more powerful numbers -> more pairs -> potentially higher max count.
    # Memory constraint is the main factor.
    # Let's try M = 2 * 10^8.
    M = 2 * 10**8
    
    # Generate powerful numbers up to M
    # Powerful number: u^2 * v^3
    powerful_set = set()
    # v^3 <= M => v <= M^(1/3)
    max_v = int(M ** (1/3)) + 2
    
    for v in range(1, max_v + 1):
        v3 = v ** 3
        # u^2 <= M / v^3 => u <= sqrt(M / v^3)
        max_u = int((M // v3) ** 0.5) + 1
        for u in range(1, max_u + 1):
            p = (u * u) * v3
            if p <= M:
                powerful_set.add(p)
            else:
                break
                
    powerful_list = sorted(powerful_set)
    P = np.array(powerful_list, dtype=np.int32)
    
    # If P is empty or too small, handle edge case
    if len(P) < 2:
        # Fallback to small search
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Compute all sums
    # To save memory, we can compute in chunks or just use the full matrix if it fits.
    # K = len(P). K^2 elements.
    # If K=10000, K^2=10^8. int32 is 4 bytes. 400MB.
    # Let's try.
    
    try:
        # Broadcasting to get all sums
        # This creates a K x K array
        sums = P[:, None] + P[None, :]
        
        # Flatten and count
        # minlength ensures the array covers up to 2*M
        counts = np.bincount(sums.ravel(), minlength=2 * M + 1)
        
        # Find the N with max representations
        # We must ensure N >= 2
        if len(counts) > 1:
            counts[0] = 0
            counts[1] = 0
            
        max_idx = np.argmax(counts)
        N = int(max_idx)
        
        # Extract pairs
        pairs = []
        P_set = set(powerful_list)
        for a in powerful_list:
            b = N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
                
        return {'N': N, 'pairs': pairs}
        
    except MemoryError:
        # Fallback to smaller M
        M = 10**8
        powerful_set = set()
        max_v = int(M ** (1/3)) + 2
        for v in range(1, max_v + 1):
            v3 = v ** 3
            max_u = int((M // v3) ** 0.5) + 1
            for u in range(1, max_u + 1):
                p = (u * u) * v3
                if p <= M:
                    powerful_set.add(p)
                else:
                    break
        powerful_list = sorted(powerful_set)
        P = np.array(powerful_list, dtype=np.int32)
        sums = P[:, None] + P[None, :]
        counts = np.bincount(sums.ravel(), minlength=2 * M + 1)
        counts[0] = 0
        counts[1] = 0
        max_idx = np.argmax(counts)
        N = int(max_idx)
        pairs = []
        P_set = set(powerful_list)
        for a in powerful_list:
            b = N - a
            if b >= a and b in P_set:
                pairs.append([a, b])
        return {'N': N, 'pairs': pairs}
