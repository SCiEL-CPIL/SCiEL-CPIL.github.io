import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    def search(k, min_d, target, current):
        if time.time() > deadline:
            return None

        if k == 0:
            if target[0] == 0:
                return current
            return None

        num, den = target
        
        # 1/d <= num/den  => d >= den/num
        # If num == 0, handled by base case.
        if num == 0:
            return None # Can't have positive denominators sum to 0
            
        start_d = max(min_d, den // num + 1)
        
        d = start_d
        while d <= 10000000:
            if time.time() > deadline:
                return None

            # 1/d <= num/den
            # den <= d * num
            if den > d * num:
                d += 1
                continue

            # Calculate remainder
            rem_num = num * d - den
            rem_den = den * d
            
            if rem_num == 0:
                if k == 1:
                    return current + [d]
                else:
                    d += 1
                    continue
            
            # rem_num > 0
            
            # Pruning: Max sum of k-1 terms starting from d+1
            # Sum < (k-1)/(d+1)
            # If (k-1)/(d+1) < rem_num/rem_den, break
            # (k-1)*rem_den < rem_num*(d+1)
            if (k - 1) * rem_den < rem_num * (d + 1):
                break

            res = search(k - 1, d + 1, (rem_num, rem_den), current + [d])
            if res:
                return res
            
            d += 1
        
        return None

    # Minimum count is 8
    result = search(8, 5, (1, 1), [])
    if result:
        return result

    # Fallback for larger counts if 8 fails
    for k in range(9, 20):
        if time.time() > deadline:
            break
        result = search(k, 5, (1, 1), [])
        if result:
            return result

    # If still nothing, return a hardcoded valid set if possible, or raise
    # A valid set of 8:
    # 5, 6, 7, 8, 9, 10, 11, 12?
    # Sum: 1/5+1/6+1/7+1/8+1/9+1/10+1/11+1/12
    # LCM 27720
    # 5544 + 4620 + 3960 + 3465 + 3080 + 2772 + 2520 + 2310 = 28271 != 27720
    # So that specific set doesn't work.
    
    # But the search should find one.
    raise RuntimeError("No solution found within time")

