import random
import time

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    
    # Map index to list of AP indices containing it
    aps_for_index = [[] for _ in range(N)]
    for i, ap in enumerate(aps):
        for idx in ap:
            aps_for_index[idx].append(i)

    def is_violation(ap, colors):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) < 3

    def find_coloring_local_search(K, time_limit):
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        # Initialize violation state
        is_violating = [False] * num_aps
        violation_count = 0
        for i, ap in enumerate(aps):
            if is_violation(ap, colors):
                is_violating[i] = True
                violation_count += 1
        
        if violation_count == 0:
            return colors

        start_time = time.time()
        
        while violation_count > 0 and time.time() - start_time < time_limit:
            # Pick a random violating AP
            # We can pick a random index from 0 to num_aps-1 and check if it's violating
            # Or maintain a list. For simplicity and speed, let's pick random AP index.
            # If it's not violating, try again.
            # To avoid infinite loop if few violations, we can just pick a random one.
            ap_idx = random.randint(0, num_aps - 1)
            if not is_violating[ap_idx]:
                continue
                
            ap = aps[ap_idx]
            # Pick a random position in the AP to change
            pos_in_ap = random.randint(0, 3)
            idx = ap[pos_in_ap]
            
            # Change color
            old_color = colors[idx]
            new_color = random.randint(0, K - 1)
            if old_color == new_color:
                continue
            colors[idx] = new_color
            
            # Update violations for all APs involving idx
            for ap_idx_2 in aps_for_index[idx]:
                # Check if this AP is currently violating
                was_violating = is_violating[ap_idx_2]
                now_violating = is_violation(aps[ap_idx_2], colors)
                
                if was_violating and not now_violating:
                    violation_count -= 1
                    is_violating[ap_idx_2] = False
                elif not was_violating and now_violating:
                    violation_count += 1
                    is_violating[ap_idx_2] = True

        if violation_count == 0:
            return colors
        return None

    # Try increasing K
    for K in [3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 128]:
        # Try a few restarts for each K
        for _ in range(5):
            sol = find_coloring_local_search(K, 2.0)
            if sol is not None:
                return sol

    # Fallback: Greedy
    colors = [0] * N
    for x in range(N):
        valid_colors = []
        for c in range(128):
            ok = True
            max_d = x // 3
            for d in range(1, max_d + 1):
                idx3 = x - 1 - d
                idx2 = x - 1 - 2 * d
                idx1 = x - 1 - 3 * d
                if idx1 < 0:
                    continue
                c3 = colors[idx3]
                c2 = colors[idx2]
                c1 = colors[idx1]
                if len({c, c1, c2, c3}) < 3:
                    ok = False
                    break
            if ok:
                valid_colors.append(c)
        
        if valid_colors:
            used = set(colors[:x])
            preferred = [c for c in valid_colors if c in used]
            if preferred:
                colors[x] = random.choice(preferred)
            else:
                colors[x] = random.choice(valid_colors)
        else:
            colors[x] = random.randint(0, 127)
            
    return colors
