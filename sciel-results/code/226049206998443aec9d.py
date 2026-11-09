import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    # Generate powerful numbers up to L
    L = 200_000_000
    powerful = set()
    max_y = int(round(L ** (1/3))) + 1
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > L:
            break
        max_x = int(round((L // y3) ** 0.5)) + 1
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= L:
                powerful.add(val)
    
    P = sorted(powerful)
    K = len(P)
    P_arr = np.array(P, dtype=np.int64)
    P_set = set(P)
    
    best_N = 2
    best_count = 0
    best_pairs = [[1, 1]]
    
    def count_representations_vectorized(N):
        """Count representations of N as sum of two powerful numbers using numpy."""
        # a must be <= N/2, and a in P, and N-a in P
        # Find all a in P with a <= N/2
        mask = P_arr <= N // 2
        a_candidates = P_arr[mask]
        if len(a_candidates) == 0:
            return 0, []
        b_candidates = N - a_candidates
        # Check which b_candidates are in P
        # Use searchsorted for efficiency
        idx = np.searchsorted(P_arr, b_candidates)
        # Check bounds and actual values
        valid = (idx < len(P_arr)) & (P_arr[idx] == b_candidates)
        count = int(np.sum(valid))
        pairs = []
        for i in range(len(a_candidates)):
            if valid[i]:
                pairs.append([int(a_candidates[i]), int(b_candidates[i])])
        return count, pairs
    
    # Phase 1: Coarse search
    N_min = 2
    N_max = 2 * L
    step_coarse = 1000000
    
    for N in range(N_min, N_max + 1, step_coarse):
        if time.time() > deadline - 20:
            break
        count, pairs = count_representations_vectorized(N)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    # Phase 2: Fine search around best
    if best_N > 2:
        search_range = 200000
        N_low = max(2, best_N - search_range)
        N_high = min(N_max, best_N + search_range)
        
        for N in range(N_low, N_high + 1, 100):
            if time.time() > deadline - 5:
                break
            count, pairs = count_representations_vectorized(N)
            if count > best_count:
                best_count = count
                best_N = N
                best_pairs = pairs
    
    # Test specific N values
    test_Ns = [2, 4, 8, 16, 32, 64, 100, 1000, 10000, 100000, 
               L // 2, 3 * L // 4, L, 5 * L // 4, 7 * L // 4, 2 * L - 1]
    for N in test_Ns:
        if N < 2 or N > N_max:
            continue
        if time.time() > deadline - 5:
            break
        count, pairs = count_representations_vectorized(N)
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = pairs
    
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': best_pairs}
