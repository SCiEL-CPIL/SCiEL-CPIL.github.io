def construct():
    from fractions import Fraction
    import time
    import math
    
    start_time = time.time()
    deadline = start_time + 45
    
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    float_fracs = [1.0 / d for d in denominators]
    
    # Meet in the middle
    mid = n // 2
    nl = mid
    nr = n - mid
    
    left_fracs_float = float_fracs[:nl]
    right_fracs_float = float_fracs[nl:]
    
    # Enumerate left subset sums (float for speed)
    left_sums = []  # (float_sum, mask)
    for mask in range(1 << nl):
        s = 0.0
        for j in range(nl):
            if mask & (1 << j):
                s += left_fracs_float[j]
        if s < 1.0:
            left_sums.append((s, mask))
    
    left_sums.sort(key=lambda x: x[0])
    
    # Binary search helper
    def find_best_left(target_float):
        lo, hi = 0, len(left_sums)
        while lo < hi:
            m = (lo + hi) // 2
            if left_sums[m][0] < target_float:
                lo = m + 1
            else:
                hi = m
        if lo == 0:
            return None
        return left_sums[lo - 1]
    
    best_deficit = Fraction(1)
    best_left_mask = 0
    best_right_mask = 0
    
    # Enumerate right subsets
    for mask in range(1 << nr):
        if time.time() > deadline:
            break
        s_r = 0.0
        for j in range(nr):
            if mask & (1 << j):
                s_r += right_fracs_float[j]
        if s_r >= 1.0:
            continue
        
        target = 1.0 - s_r
        best_l = find_best_left(target)
        if best_l is None:
            continue
        
        s_l_float, lmask = best_l
        # Verify with exact arithmetic
        s_l_exact = Fraction(0)
        for j in range(nl):
            if lmask & (1 << j):
                s_l_exact += fracs[j]
        s_r_exact = Fraction(0)
        for j in range(nr):
            if mask & (1 << j):
                s_r_exact += fracs[nl + j]
        
        total = s_l_exact + s_r_exact
        if total < 1:
            deficit = 1 - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lmask
                best_right_mask = mask
    
    result = []
    for j in range(nl):
        if best_left_mask & (1 << j):
            result.append(j)
    for j in range(nr):
        if best_right_mask & (1 << j):
            result.append(nl + j)
    
    if not result:
        result = [0]
    
    return result
