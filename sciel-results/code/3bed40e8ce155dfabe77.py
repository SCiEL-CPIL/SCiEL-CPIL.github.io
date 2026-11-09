import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 128
    aps = []
    ap_indices = [[] for _ in range(n)]
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            idx = len(aps)
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            aps.append((a, b, c, dd))
            ap_indices[a].append(idx)
            ap_indices[b].append(idx)
            ap_indices[c].append(idx)
            ap_indices[dd].append(idx)
    
    num_aps = len(aps)
    
    def is_bad(c, ap_idx):
        a, b, c_idx, d_idx = aps[ap_idx]
        ca, cb, cc, cd = c[a], c[b], c[c_idx], c[d_idx]
        return len({ca, cb, cc, cd}) < 3

    best_solution = None
    best_energy = float('inf')
    
    for K in range(3, 8):
        if time.time() > deadline:
            break
            
        for restart in range(20):
            if time.time() > deadline:
                break
                
            c = [random.randint(0, K - 1) for _ in range(n)]
            
            energy = 0
            ap_bad = [False] * num_aps
            for i in range(num_aps):
                if is_bad(c, i):
                    ap_bad[i] = True
                    energy += 1
            
            if energy == 0:
                return c
            
            T = 1.0
            cool_rate = 0.9995
            min_T = 0.01
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                i = random.randint(0, n - 1)
                old_color = c[i]
                new_color = random.randint(0, K - 1)
                if new_color == old_color:
                    continue
                
                affected_aps = ap_indices[i]
                
                # Calculate delta
                delta = 0
                # Store old states
                old_states = []
                for ap_idx in affected_aps:
                    was_bad = ap_bad[ap_idx]
                    old_states.append((ap_idx, was_bad))
                    
                    # Temporarily set new color
                    c[i] = new_color
                    is_bad_now = is_bad(c, ap_idx)
                    # Revert temporarily? No, we'll revert all at once if rejected.
                    # But we need to calculate delta for ALL affected APs.
                    # So we must calculate is_bad_now for each.
                    
                    if was_bad and not is_bad_now:
                        delta -= 1
                    elif not was_bad and is_bad_now:
                        delta += 1
                    # Note: c[i] is currently new_color.
                    # We don't update ap_bad yet.
                
                # Decide accept/reject
                if delta < 0 or random.random() < math.exp(-delta / T):
                    # Accept
                    # Update ap_bad
                    for ap_idx, was_bad in old_states:
                        # c[i] is already new_color
                        ap_bad[ap_idx] = is_bad(c, ap_idx)
                    energy += delta
                else:
                    # Reject
                    c[i] = old_color
                    # ap_bad remains unchanged
                
                T *= cool_rate
                if T < min_T:
                    break
            
            # Check final
            final_energy = 0
            for i in range(num_aps):
                if is_bad(c, i):
                    final_energy += 1
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_solution = c[:]
                
            if best_energy == 0:
                return best_solution
                
    # Fallback Greedy
    colors = []
    for x in range(128):
        found = False
        for color in range(128):
            valid = True
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 0: continue
                b = a + d
                c_idx = a + 2 * d
                if len({colors[a], colors[b], colors[c_idx], color}) < 3:
                    valid = False
                    break
            if valid:
                for d in range(1, (x - 1) // 2 + 1):
                    if x - 1 - 2 * d < 0: continue
                    if colors[x - 1 - d] == color and colors[x - 1 - 2 * d] == color:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
        if not found:
            colors.append(0)
    return colors
