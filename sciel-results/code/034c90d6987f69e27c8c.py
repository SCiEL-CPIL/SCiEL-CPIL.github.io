import time
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Precompute integer weights w_i = L / i
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves for Meet-in-the-Middle
    half_size = n // 2
    w1 = weights[:half_size]
    w2 = weights[half_size:]
    
    # Generate all possible sums AND store the best coefficient tuple for each sum
    # Using a dict: sum -> coeffs_tuple (first found is fine, any valid coeffs work)
    def generate_sums_and_coeffs(w_list):
        # Start with {0: ()}
        state = {0: ()}
        for w in w_list:
            new_state = {}
            for s, coeffs in state.items():
                # c = -1
                ns = s - w
                if ns not in new_state:
                    new_state[ns] = coeffs + (-1,)
                # c = 0
                if s not in new_state:
                    new_state[s] = coeffs + (0,)
                # c = 1
                ns = s + w
                if ns not in new_state:
                    new_state[ns] = coeffs + (1,)
            state = new_state
        return state

    # Generate sums and coefficients for both halves
    t0 = time.time()
    map1 = generate_sums_and_coeffs(w1)
    map2 = generate_sums_and_coeffs(w2)
    
    # Get sorted list of sums2 for binary search
    sums2_sorted = sorted(map2.keys())
    
    best_abs_int_diff = float('inf')
    best_s1 = 0
    best_s2 = 0
    
    len2 = len(sums2_sorted)
    
    # Iterate through sums1
    for s1 in map1.keys():
        if time.time() - start_time > deadline - 5:
            break
            
        target = -s1
        import bisect
        idx = bisect.bisect_left(sums2_sorted, target)
        
        # Check idx and idx-1 (and possibly idx+1 for safety)
        candidates = []
        if idx < len2:
            candidates.append(sums2_sorted[idx])
        if idx > 0:
            candidates.append(sums2_sorted[idx - 1])
        if idx + 1 < len2:
            candidates.append(sums2_sorted[idx + 1])
            
        for s2 in candidates:
            diff = s1 + s2
            if diff != 0:
                abs_diff = abs(diff)
                if abs_diff < best_abs_int_diff:
                    best_abs_int_diff = abs_diff
                    best_s1 = s1
                    best_s2 = s2
                    if best_abs_int_diff == 1:
                        break
        if best_abs_int_diff == 1:
            break
            
    if best_abs_int_diff == float('inf'):
        # Fallback: c_29=1, c_30=-1 gives 1/29 - 1/30 = 1/870
        return [0] * 28 + [1, -1]
        
    # Reconstruct coefficients directly from the maps
    c1 = list(map1[best_s1])
    c2 = list(map2[best_s2])
    result = c1 + c2
    
    # Verify
    total = sum(c * w for c, w in zip(result, weights))
    if total == 0:
        return [0] * 28 + [1, -1]
        
    return result

# EVOLVE-BLOCK-END
