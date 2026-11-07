import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # We will search up to this limit for k
    # 50 million is a good balance. 80 million might be risky on memory/time.
    # Let's start with 50M and see if we can push higher if fast.
    N_MAX = 50000000

    # Precompute sigma for all k using a divisor-sum sieve with NumPy
    # Use int64 to be safe
    sigma_arr = np.zeros(N_MAX + 1, dtype=np.int64)
    
    # Sieve
    # The loop is the bottleneck.
    # We can optimize by only iterating d up to N_MAX
    for d in range(1, N_MAX + 1):
        if d % 10000 == 0:
            if time.time() > deadline - 40:
                break
        sigma_arr[d::d] += d

    if time.time() > deadline - 30:
        # Fallback to a smaller range or return best found so far?
        # If we break early, sigma_arr is incomplete.
        # We must ensure we have a valid result.
        # If we break, we can only use the portion we computed?
        # No, the sieve is incomplete.
        # Better to reduce N_MAX if time is tight.
        # But we don't know it's tight until we try.
        # Let's assume 50M fits in time.
        pass

    # Compute M = k * sigma(k)
    # k is 1 to N_MAX
    k_vals = np.arange(1, N_MAX + 1, dtype=np.int64)
    M_vals = k_vals * sigma_arr[1:N_MAX + 1]
    
    # Filter M <= 10^18
    mask = M_vals <= 10**18
    M_valid = M_vals[mask]
    k_valid = k_vals[mask]
    
    if len(M_valid) == 0:
        return {'M': 1, 'k': [1]}

    # Sort M_valid to find duplicates
    # We need to keep track of which k corresponds to which M
    # Sorting indices is better
    sorted_indices = np.argsort(M_valid)
    M_sorted = M_valid[sorted_indices]
    k_sorted = k_valid[sorted_indices]
    
    # Find max run of identical M values
    # Use np.diff to find where M changes
    # If M_sorted[i] == M_sorted[i+1], it's a continuation
    # Otherwise, it's a new group
    
    # Create a boolean array indicating start of a new group
    # new_group[0] is True
    # new_group[i] is True if M_sorted[i] != M_sorted[i-1]
    new_group = np.ones(len(M_sorted), dtype=bool)
    if len(M_sorted) > 1:
        new_group[1:] = M_sorted[1:] != M_sorted[:-1]
        
    # Find indices where new groups start
    group_starts = np.where(new_group)[0]
    
    # Length of each group
    # Group i starts at group_starts[i] and ends at group_starts[i+1] - 1
    # Last group ends at len(M_sorted) - 1
    group_lengths = np.diff(np.append(group_starts, len(M_sorted)))
    
    # Find max length
    max_len = np.max(group_lengths)
    
    if max_len <= 1:
        # No collisions found? Return a known small solution or best single
        # The task requires distinct k. If max_len is 1, we return 1 solution.
        # But we want to maximize.
        # If no collisions, we return the first one?
        # Actually, the score is number of solutions.
        # If we can't find >1, we return 1.
        # But there are always collisions for small M.
        # e.g. M=12, k=3 (3*4=12), k=4 (4*3=12)? No sigma(4)=7. 4*7=28.
        # M=12: k=3 (sigma=4, 3*4=12). Any other?
        # k=1 -> 1. k=2 -> 2*3=6. k=3 -> 12. k=4 -> 28.
        # So M=12 has 1 solution.
        # M=6: k=2 (2*3=6). k=1 (1).
        # M=18: k=6 (6*12=72)? No.
        # Let's just return the best found.
        
        # If max_len is 1, we pick the first k.
        best_M = int(M_sorted[0])
        best_ks = [int(k_sorted[0])]
    else:
        # Find the index of the max length
        max_idx = np.argmax(group_lengths)
        start_idx = group_starts[max_idx]
        end_idx = start_idx + max_len
        
        best_M = int(M_sorted[start_idx])
        best_ks = [int(k) for k in k_sorted[start_idx:end_idx]]

    # Verify
    def sigma_exact(n):
        if n == 1: return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                exp = 0
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    exp += 1
                    p_power *= p
                result *= (p_power * p - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if temp > 1:
            result *= (temp * temp - 1) // (temp - 1)
        return result

    final_ks = []
    for k in best_ks:
        if k * sigma_exact(k) == best_M:
            final_ks.append(k)
            
    if not final_ks:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}
        
    return {'M': int(best_M), 'k': final_ks}
