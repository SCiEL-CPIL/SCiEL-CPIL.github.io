"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    p = 1009
    N = p - 1  # 1008
    
    # Precompute modular inverses using Fermat's little theorem (p is prime)
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each index i in [1, N], define R[i] as the maximum allowed right endpoint
    # if i is in the interval. If inv[i] > i, then having i forces e < inv[i], so R[i] = inv[i] - 1.
    # If inv[i] < i, then the constraint is already handled by the element inv[i] (which has R[inv[i]] = i - 1).
    # If inv[i] == i (self-inverse: i=1 or i=1008), then i cannot be in any valid interval, so R[i] = i - 1.
    INF = float('inf')
    R = [INF] * (p + 2)  # index 1..1008
    
    for i in range(1, p):
        j = inv[i]
        if j == i:
            # Self-inverse: i cannot appear in any valid interval
            R[i] = i - 1
        elif j > i:
            # If i is in interval, e must be < j
            R[i] = j - 1
        else:
            # j < i: constraint is handled by element j (R[j] = i - 1)
            # R[i] remains INF (no additional constraint from i's perspective)
            pass
    
    # For a given start s, the maximum valid end is:
    #   max_e(s) = min{ R[k] | k in [s, N] }
    # because for every k >= s that is in the interval, e must be <= R[k].
    
    # Compute suffix minimum: suffix_min[s] = min(R[s], suffix_min[s+1])
    suffix_min = [INF] * (p + 2)
    suffix_min[p] = INF  # no elements beyond N
    for s in range(p - 1, 0, -1):
        suffix_min[s] = min(R[s], suffix_min[s + 1])
    
    # Now find the longest valid interval
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        if time.time() > deadline - 2.0:
            break
        
        # Maximum valid end for start s
        max_e = suffix_min[s]
        if max_e > N:
            max_e = N
        
        if max_e >= s:
            length = max_e - s + 1
            if length > best_len:
                best_len = length
                best_start = s
    
    # Verify the found interval is valid (sanity check)
    s_final = best_start
    e_final = s_final + best_len - 1
    for x in range(s_final, e_final + 1):
        for y in range(x, e_final + 1):
            if (x * y) % p == 1:
                # Fallback: do a full brute-force search
                best_len_bf = 0
                best_start_bf = 1
                for s2 in range(1, p):
                    for e2 in range(s2, p):
                        valid = True
                        for x2 in range(s2, e2 + 1):
                            for y2 in range(x2, e2 + 1):
                                if (x2 * y2) % p == 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            len2 = e2 - s2 + 1
                            if len2 > best_len_bf:
                                best_len_bf = len2
                                best_start_bf = s2
                return [best_start_bf, best_len_bf]
    
    return [best_start, best_len]

# EVOLVE-BLOCK-END
