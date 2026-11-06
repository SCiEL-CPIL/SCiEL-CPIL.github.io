import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    def max_sum(r, d_min):
        s = 0.0
        for i in range(r):
            s += 1.0 / (d_min + i)
        return s

    def search(k, d_min, current_sum, current_denoms):
        if time.time() > deadline:
            return None
        
        if k == 0:
            if abs(current_sum - 1.0) < 1e-9:
                return current_denoms
            return None
        
        target = 1.0 - current_sum
        
        if target < -1e-9:
            return None

        # Pruning: Max possible sum with k terms starting from d_min
        m = max_sum(k, d_min)
        if m < target - 1e-9:
            return None
            
        # Estimate max denominator for this step
        # 1/d + (k-1)/(d+1) >= target
        # Approx 1/d + (k-1)/d >= target => k/d >= target => d <= k/target
        # Use a slightly larger bound to be safe
        if target > 0:
            max_d = int(k / target) + 20
        else:
            max_d = 10**7
            
        # Cap at 10^7
        max_d = min(max_d, 10**7)
        
        for d in range(d_min, max_d + 1):
            if time.time() > deadline:
                return None
            
            new_sum = current_sum + 1.0 / d
            if new_sum > 1.0 + 1e-9:
                continue
                
            rem_target = 1.0 - new_sum
            if rem_target < -1e-9:
                continue
            
            if k - 1 == 0:
                if abs(rem_target) < 1e-9:
                    return current_denoms + [d]
            else:
                # Check if remaining sum is possible
                # Max sum of remaining k-1 terms starting from d+1
                if max_sum(k-1, d+1) < rem_target - 1e-9:
                    continue
                
                res = search(k-1, d+1, new_sum, current_denoms + [d])
                if res is not None:
                    return res
                    
        return None

    # Minimum k is 8
    for k in range(8, 15):
        res = search(k, 5, 0.0, [])
        if res is not None:
            return res
            
    # Fallback: Return a known valid set if search fails (unlikely)
    # This set sums to 1? 
    # 1/2+1/3+1/7+1/42 = 1. 
    # Split 1/2 = 1/3+1/6 (3<5 no). 
    # Split 1/2 = 1/5+1/10. 
    # 1/3 = 1/6+1/12? No. 1/3 = 1/4+1/12? No.
    # Let's just return the first found or a hardcoded one if we had one.
    # Since k=8 should be found, this shouldn't be reached.
    raise RuntimeError("Search failed to find a solution within time limit")
