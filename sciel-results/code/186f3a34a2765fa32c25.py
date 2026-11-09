import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # 1. Generate powerful numbers up to 10^12
    L = 10**12
    powerful_set = set()
    max_u = int(L**0.5)
    for u in range(1, max_u + 1):
        u2 = u * u
        if u2 > L:
            break
        # v^3 <= L / u2
        # v <= (L / u2)^(1/3)
        max_v = int((L // u2) ** (1/3))
        # Adjust for floating point errors
        while (max_v + 1)**3 * u2 <= L:
            max_v += 1
        while max_v**3 * u2 > L:
            max_v -= 1
        for v in range(1, max_v + 1):
            powerful_set.add(u2 * (v ** 3))
    
    P_array = np.array(sorted(powerful_set), dtype=np.int64)
    
    # 2. Select candidates for N
    N_min = 10**11
    N_max = 10**12
    
    candidates = set()
    
    # Random sampling
    np.random.seed(42)
    # Bias towards larger N
    for _ in range(2000):
        # Uniform in [N_min, N_max]
        n = np.random.randint(N_min, N_max)
        candidates.add(n)
        
    # Structured sampling: Multiples of 60?
    # Let's add some specific high-value candidates
    # E.g., N = 10^12, 10^12 - 1, etc.
    for i in range(100):
        candidates.add(N_max - i)
        candidates.add(N_min + i)
        
    # 3. Evaluate candidates
    best_count = 0
    best_N = 2
    best_pairs = []
    
    # Sort candidates to process larger ones first? 
    # Larger N has higher expected count.
    candidates = sorted(candidates, reverse=True)
    
    for N in candidates:
        if time.time() > deadline:
            break
            
        half_N = N // 2
        idx = np.searchsorted(P_array, half_N)
        a_candidates = P_array[:idx]
        
        if len(a_candidates) == 0:
            continue
            
        b_candidates = N - a_candidates
        
        # Check if b_candidates are in P_array
        matches = np.in1d(b_candidates, P_array)
        
        count = np.sum(matches)
        
        if count > best_count:
            best_count = count
            best_N = N
            a_vals = a_candidates[matches]
            b_vals = b_candidates[matches]
            pairs = sorted(zip(a_vals.tolist(), b_vals.tolist()))
            best_pairs = pairs
            
            # Optional: Early stop if count is very high?
            # No, keep searching.
            
    if not best_pairs:
         best_N = 2
         best_pairs = [[1, 1]]
         best_count = 1

    return {'N': best_N, 'pairs': best_pairs}
