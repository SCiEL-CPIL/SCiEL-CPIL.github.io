import time
import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    
    # Calculate LCM of divisors
    lcm_val = 1
    for d in divisors:
        lcm_val = lcm_val * d // math.gcd(lcm_val, d)
        
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit

    # Pre-compute sorted indices for MRV heuristic? 
    # The MRV order depends on the window, so we compute it per window.
    
    # We search for the minimum span.
    # Theoretical minimum span is 9 (10 distinct integers).
    # We iterate span upwards.
    
    for span in range(9, lcm_val + 1):
        if time.time() > deadline:
            break
            
        for L in range(1, lcm_val + 1):
            if time.time() > deadline:
                break
                
            # 1. Check existence and build masks
            # Values in window: L, L+1, ..., L+span
            # Index of value v is v - L
            
            masks = []
            counts = []
            valid_window = True
            
            for d in divisors:
                m = 0
                cnt = 0
                # Find multiples of d in [L, L+span]
                # First multiple >= L
                first_k = (L + d - 1) // d
                val = first_k * d
                while val <= L + span:
                    idx = val - L
                    m |= (1 << idx)
                    cnt += 1
                    val += d
                
                if cnt == 0:
                    valid_window = False
                    break
                masks.append(m)
                counts.append(cnt)
                
            if not valid_window:
                continue
                
            # 2. Order divisors by number of candidates (MRV)
            # We need to keep track of which mask corresponds to which original divisor
            # to reconstruct the answer.
            
            order = list(range(n))
            order.sort(key=lambda i: counts[i])
            
            ordered_masks = [masks[i] for i in order]
            
            # 3. DFS to find a valid assignment
            # State: idx in order, used_mask
            # We want to assign a bit from ordered_masks[idx] that is not in used_mask
            
            solution = [0] * n  # Stores value for each original divisor index
            found = False
            
            def dfs(idx, used_mask):
                nonlocal found
                if idx == n:
                    found = True
                    return True
                    
                m = ordered_masks[idx]
                available = m & ~used_mask
                
                if available == 0:
                    return False
                
                # Iterate through set bits in available
                # Extract bits
                a = available
                while a:
                    # Get lowest set bit
                    bit = a & -a
                    a -= bit
                    # Index of this bit is the value offset
                    offset = bit.bit_length() - 1
                    value = L + offset
                    
                    # Assign this value to the divisor at original index order[idx]
                    solution[order[idx]] = value
                    
                    if dfs(idx + 1, used_mask | bit):
                        return True
                    
                    # Backtrack
                    solution[order[idx]] = 0
                    
                return False
                
            if dfs(0, 0):
                # Ensure types are int
                return [int(x) for x in solution]
                
    # Fallback (should not happen)
    return [0] * n
