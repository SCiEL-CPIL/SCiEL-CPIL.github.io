import time

def construct():
    start_time = time.time()
    deadline = 60.0 # 60s limit, keep 10s margin
    
    selected = []
    
    # Greedy from top
    for x in range(192, 0, -1):
        if time.time() - start_time > deadline:
            break
            
        # Check if x can be added
        valid = True
        
        # 1. Check if x divides sum of 3 distinct from selected
        # Sum of 3 from selected
        # If len(selected) < 3, no sum of 3 exists, so valid for this part
        if len(selected) >= 3:
            # Optimization: min sum of 3 from selected
            s_min = sum(sorted(selected)[:3])
            s_max = sum(sorted(selected)[-3:])
            # If x > s_max, x cannot divide sum (sum < x, sum > 0)
            # If 2*x > s_max and x > s_max, then only multiple is x, which is > s_max.
            # So if x > s_max, valid.
            if x > s_max:
                pass # Valid
            else:
                # Check multiples
                # k*x in [s_min, s_max]
                k_min = (s_min + x - 1) // x
                k_max = s_max // x
                for k in range(k_min, k_max + 1):
                    target = k * x
                    # Check if target is sum of 3 distinct from selected
                    # This is still expensive.
                    # But we can just iterate combinations if selected is small.
                    # Or break early if found.
                    # Given selected size ~48, combinations is ~16k.
                    # This is acceptable.
                    found = False
                    # We can use a set of sums? No, sums change.
                    # Just iterate.
                    from itertools import combinations
                    for c in combinations(selected, 3):
                        if sum(c) == target:
                            found = True
                            break
                    if found:
                        valid = False
                        break
        
        if not valid:
            continue
            
        # 2. Check if any a in selected divides x + b + c
        # b, c distinct from selected, b != c
        # a can be any in selected
        if len(selected) >= 2:
            for a in selected:
                # Check if a divides x + b + c
                # x + b + c range:
                # min: x + sum of 2 smallest in selected
                # max: x + sum of 2 largest in selected
                s2_min = sum(sorted(selected)[:2])
                s2_max = sum(sorted(selected)[-2:])
                min_sum = x + s2_min
                max_sum = x + s2_max
                
                if a > max_sum:
                    continue # a cannot divide sum
                
                k_min = (min_sum + a - 1) // a
                k_max = max_sum // a
                
                conflict_found = False
                for k in range(k_min, k_max + 1):
                    target = k * a
                    # Check if target - x is sum of 2 distinct from selected
                    rem = target - x
                    # rem = b + c
                    # b, c in selected, b != c
                    # We can check if rem can be formed by 2 distinct elements.
                    # Precompute pair sums?
                    # For now, iterate pairs.
                    # This is O(N^2) per a.
                    # Total O(N^3) per step.
                    # 48^3 = 110k.
                    # 192 steps * 110k = 21M. Fast enough.
                    
                    # To speed up, we can just check if rem is in pair_sums
                    # But we need distinct b,c.
                    # Let's just iterate pairs for simplicity in this thought process,
                    # but in code we should optimize.
                    
                    # Actually, we can just check if there exist b,c in selected such that b+c=rem and b!=c.
                    # We can use a set of elements.
                    # For each b in selected:
                    #   if rem - b in selected and rem - b != b:
                    #       conflict
                    
                    found_pair = False
                    # Optimize: if rem < 2*min(selected) or rem > 2*max(selected), skip
                    if rem < 2 * min(selected) or rem > 2 * max(selected):
                        continue
                    for b in selected:
                        c = rem - b
                        if c in selected and c != b:
                            found_pair = True
                            break
                    if found_pair:
                        conflict_found = True
                        break
                if conflict_found:
                    valid = False
                    break
        
        if valid:
            selected.append(x)
            
    return selected
