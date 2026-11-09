def construct():
    import time
    import random

    t0 = time.time()
    DEADLINE = t0 + 55.0
    n = 128

    # Precompute all 4-term arithmetic progressions
    aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    m = len(aps)
    if m == 0:
        return [0] * n

    # Map position to list of AP indices it belongs to
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)

    def try_find_k(k, time_budget):
        end_time = t0 + time_budget
        
        while time.time() < end_time:
            # Random initialization
            colors = [random.randrange(k) for _ in range(n)]
            
            # distinct_counts[i] = number of distinct colors in AP i
            # color_counts[i][c] = count of color c in AP i
            distinct_counts = [0] * m
            color_counts = [[0] * k for _ in range(m)]
            violations = 0
            
            for i in range(m):
                ap = aps[i]
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                cc = color_counts[i]
                cc[c0] += 1
                cc[c1] += 1
                cc[c2] += 1
                cc[c3] += 1
                d = sum(1 for c in range(k) if cc[c] > 0)
                distinct_counts[i] = d
                if d < 3:
                    violations += 1
            
            if violations == 0:
                return colors
            
            # Local Search
            stall = 0
            while time.time() < end_time and violations > 0:
                # Pick a random violating AP
                found_violation = False
                for _ in range(5): # Limit search for violation to keep it fast
                    i = random.randrange(m)
                    if distinct_counts[i] < 3:
                        found_violation = True
                        break
                
                if not found_violation:
                    break
                    
                ap = aps[i]
                p = random.choice(ap)
                old_c = colors[p]
                
                best_new_c = None
                best_delta = 10**9
                
                for c in range(k):
                    if c == old_c:
                        continue
                    
                    delta = 0
                    for j in pos_aps[p]:
                        aj = aps[j]
                        old_dist = distinct_counts[j]
                        was_viol = old_dist < 3
                        
                        # Simulate change
                        v0 = colors[aj[0]] if aj[0] != p else c
                        v1 = colors[aj[1]] if aj[1] != p else c
                        v2 = colors[aj[2]] if aj[2] != p else c
                        v3 = colors[aj[3]] if aj[3] != p else c
                        
                        new_dist = len({v0, v1, v2, v3})
                        now_viol = new_dist < 3
                        
                        if was_viol and not now_viol:
                            delta -= 1
                        elif (not was_viol) and now_viol:
                            delta += 1
                            
                    if delta < best_delta:
                        best_delta = delta
                        best_new_c = c
                        
                if best_new_c is None:
                    best_new_c = random.randrange(k)
                    
                # Apply change
                colors[p] = best_new_c
                for j in pos_aps[p]:
                    aj = aps[j]
                    old_dist = distinct_counts[j]
                    was_viol = old_dist < 3
                    
                    cc_j = color_counts[j]
                    cc_j[old_c] -= 1
                    cc_j[best_new_c] += 1
                    
                    v0, v1, v2, v3 = colors[aj[0]], colors[aj[1]], colors[aj[2]], colors[aj[3]]
                    new_dist = len({v0, v1, v2, v3})
                    distinct_counts[j] = new_dist
                    now_viol = new_dist < 3
                    
                    if was_viol and not now_viol:
                        violations -= 1
                    elif (not was_viol) and now_viol:
                        violations += 1
                        
                if best_delta < 0:
                    stall = 0
                else:
                    stall += 1
                    if stall > 500:
                        # Kick
                        p2 = random.randrange(n)
                        c2 = random.randrange(k)
                        if c2 != colors[p2]:
                            old_c2 = colors[p2]
                            colors[p2] = c2
                            for j in pos_aps[p2]:
                                aj = aps[j]
                                old_dist = distinct_counts[j]
                                was_viol = old_dist < 3
                                
                                cc_j = color_counts[j]
                                cc_j[old_c2] -= 1
                                cc_j[c2] += 1
                                
                                v0, v1, v2, v3 = colors[aj[0]], colors[aj[1]], colors[aj[2]], colors[aj[3]]
                                new_dist = len({v0, v1, v2, v3})
                                distinct_counts[j] = new_dist
                                now_viol = new_dist < 3
                                
                                if was_viol and not now_viol:
                                    violations -= 1
                                elif (not was_viol) and now_viol:
                                    violations += 1
                            stall = 0

            if violations == 0:
                return colors

        return None

    # Try k=3 first
    result = try_find_k(3, 45.0)
    if result is not None:
        return result

    # If 3 fails, try 4
    if time.time() < DEADLINE - 5:
        result = try_find_k(4, 5.0)
        if result is not None:
            return result

    # Fallback: Greedy (should not be reached)
    colors = [0] * n
    return colors
