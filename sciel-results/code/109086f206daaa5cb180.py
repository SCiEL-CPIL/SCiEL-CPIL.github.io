import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    
    aps = []
    ap_indices = [[] for _ in range(N)]
    
    for a in range(N):
        max_d = (N - 1 - a) // 3
        for d in range(1, max_d + 1):
            idx = len(aps)
            aps.append((a, a+d, a+2*d, a+3*d))
            for pos in aps[-1]:
                ap_indices[pos].append(idx)
                
    num_aps = len(aps)
    
    def is_valid_4ap_colors(c0, c1, c2, c3):
        # Returns True if >= 3 distinct colors
        # Logic:
        # 1 distinct: c0==c1==c2==c3
        # 2 distinct: 
        #   - 3 of one, 1 of other
        #   - 2 of one, 2 of other
        #   - 2 of one, 1 of other, 1 of other? No, that's 3 distinct.
        # So 2 distinct means the set size is 2.
        
        # Quick checks
        if c0 == c1:
            if c1 == c2:
                if c2 == c3:
                    return False # 1
                else:
                    return False # A A A B (2)
            else:
                if c2 == c3:
                    return False # A A B B (2)
                else:
                    return True # A A B C (3)
        else:
            if c1 == c2:
                if c2 == c3:
                    return False # A B B B (2)
                else:
                    return True # A B B C (3)
            else:
                if c2 == c3:
                    return True # A B C C (3)
                else:
                    # c0!=c1, c1!=c2, c2!=c3
                    if c0 == c2:
                        return True # A B A C (3)
                    elif c0 == c3:
                        return True # A B C A (3)
                    else:
                        return True # A B C D (4) or similar, always >=3

    for k in range(3, 11):
        if time.time() > deadline:
            break
            
        max_restarts = 1000
        found = False
        
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            colors = np.random.randint(0, k, N)
            
            # Initialize validity
            ap_is_valid = np.zeros(num_aps, dtype=bool)
            viol_set = set()
            
            for i in range(num_aps):
                ap = aps[i]
                if is_valid_4ap_colors(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]):
                    ap_is_valid[i] = True
                else:
                    ap_is_valid[i] = False
                    viol_set.add(i)
                    
            if len(viol_set) == 0:
                return colors.tolist()
                
            iterations = 0
            max_iterations = 5000
            stuck_count = 0
            last_viol_count = len(viol_set)
            
            while iterations < max_iterations:
                if time.time() > deadline:
                    break
                    
                if len(viol_set) == 0:
                    found = True
                    break
                    
                if len(viol_set) < last_viol_count:
                    last_viol_count = len(viol_set)
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count > 50:
                        break
                
                # Pick a random violating AP
                # Convert to list if needed, or use random choice on set?
                # random.choice requires sequence.
                # For small sets, list() is fine.
                viol_list = list(viol_set)
                if not viol_list:
                    break
                viol_ap_idx = random.choice(viol_list)
                pos = random.choice(aps[viol_ap_idx])
                
                best_color = colors[pos]
                best_viol_count = len(viol_set)
                
                for c in range(k):
                    if c == colors[pos]:
                        continue
                    
                    old_c = colors[pos]
                    colors[pos] = c
                    
                    # Update validity for APs containing pos
                    # We need to track changes to viol_set
                    # To avoid modifying viol_set multiple times if we revert,
                    # we can simulate or just update and revert.
                    # Updating and reverting is safer.
                    
                    # Save old states of affected APs
                    affected_aps = ap_indices[pos]
                    old_states = {}
                    for ap_idx in affected_aps:
                        old_states[ap_idx] = ap_is_valid[ap_idx]
                        
                    # Apply new states
                    new_viol_count_delta = 0
                    # We can't easily get delta without updating set.
                    # Let's just update the set and count.
                    
                    # This is getting complex. 
                    # Simpler: Just recalculate the count for affected APs 
                    # and estimate the new total count.
                    
                    # Actually, let's just update `ap_is_valid` and `viol_set` 
                    # and if this color is not the best, we revert.
                    
                    # To revert, we need to restore `ap_is_valid` and `viol_set`.
                    # This is expensive if we do it for every color.
                    
                    # Alternative: 
                    # Calculate the "new violation count" for affected APs 
                    # without modifying the global set.
                    
                    # Let's do that.
                    new_viol_count = len(viol_set)
                    for ap_idx in affected_aps:
                        was_valid = ap_is_valid[ap_idx]
                        ap = aps[ap_idx]
                        is_valid_now = is_valid_4ap_colors(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                        
                        if was_valid and not is_valid_now:
                            new_viol_count += 1
                        elif not was_valid and is_valid_now:
                            new_viol_count -= 1
                        # else: no change
                    
                    if new_viol_count < best_viol_count:
                        best_viol_count = new_viol_count
                        best_color = c
                        
                    # Revert colors[pos]
                    colors[pos] = old_c
                    
                # Apply best color
                if best_color != colors[pos]:
                    colors[pos] = best_color
                    # Update global state
                    affected_aps = ap_indices[pos]
                    for ap_idx in affected_aps:
                        was_valid = ap_is_valid[ap_idx]
                        ap = aps[ap_idx]
                        is_valid_now = is_valid_4ap_colors(colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
                        ap_is_valid[ap_idx] = is_valid_now
                        
                        if was_valid and not is_valid_now:
                            viol_set.add(ap_idx)
                        elif not was_valid and is_valid_now:
                            viol_set.discard(ap_idx)
                            
                iterations += 1
                
            if found:
                return colors.tolist()

    # Fallback
    colors = []
    for x in range(1, 129):
        for color in range(128):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x-3*d < 1: break
                c1 = colors[x-1-3*d]
                c2 = colors[x-1-2*d]
                c3 = colors[x-1-d]
                if len({c1, c2, c3, color}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
    return colors
