import random
import time

def construct():
    N = 128
    aps = []
    for a in range(1, N + 1):
        for d in range(1, (N - a) // 3 + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    ap_count = len(aps)
    # Map position to list of AP indices
    pos_to_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d) in enumerate(aps):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d].append(idx)
        
    def ap_violates(colors, a, b, c, d):
        return len({colors[a], colors[b], colors[c], colors[d]}) < 3

    start_time = time.time()
    deadline = start_time + 40.0
    
    best_cost = float('inf')
    best_colors = None
    
    for restart in range(50):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 2) for _ in range(N)]
        current_cost = 0
        # Precompute violation status for each AP
        ap_status = [False] * ap_count
        for idx, (a, b, c, d) in enumerate(aps):
            if ap_violates(colors, a, b, c, d):
                ap_status[idx] = True
                current_cost += 1
                
        if current_cost == 0:
            return colors
            
        T = 1.0
        while time.time() < deadline:
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = random.randint(0, 2)
            if new_c == old_c:
                continue
                
            # Calculate delta cost
            delta = 0
            affected_aps = pos_to_aps[i]
            for idx in affected_aps:
                a, b, c, d = aps[idx]
                was_violating = ap_status[idx]
                # Update color temporarily for check
                # We need to check if this AP violates with new_c at i
                # Get the other 3 colors
                others = []
                for pos in (a, b, c, d):
                    if pos != i:
                        others.append(colors[pos])
                    else:
                        others.append(new_c)
                now_violating = len(set(others)) < 3
                
                if was_violating and not now_violating:
                    delta -= 1
                elif not was_violating and now_violating:
                    delta += 1
                    
            if delta < 0 or random.random() < pow(2.718, -delta / T):
                # Accept
                colors[i] = new_c
                # Update ap_status and current_cost
                current_cost += delta
                for idx in affected_aps:
                    a, b, c, d = aps[idx]
                    # Recalculate status
                    ap_status[idx] = len({colors[a], colors[b], colors[c], colors[d]}) < 3
                    
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
                    
                if current_cost == 0:
                    return colors
            else:
                # Reject, colors[i] remains old_c, no state change needed
                pass
                
            T *= 0.99
            if T < 0.05:
                break
                
    return best_colors
