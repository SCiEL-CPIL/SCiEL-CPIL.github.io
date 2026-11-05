import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    n = 512

    # Precompute all 4-APs
    ap_list = []
    ap_masks = []
    elem_to_ap_indices = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if e > n:
                continue
            ap = (a, b, c, e)
            ap_idx = len(ap_list)
            ap_list.append(ap)
            mask = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)) | (1 << (e - 1))
            ap_masks.append(mask)
            elem_to_ap_indices[a].append(ap_idx)
            elem_to_ap_indices[b].append(ap_idx)
            elem_to_ap_indices[c].append(ap_idx)
            elem_to_ap_indices[e].append(ap_idx)
            
    num_aps = len(ap_list)
    
    # Precompute blockers: for each element x, list of (ap_idx, mask_of_other_3)
    elem_blockers = [[] for _ in range(n + 1)]
    for ap_idx, ap in enumerate(ap_list):
        mask = ap_masks[ap_idx]
        for x in ap:
            x_bit = 1 << (x - 1)
            other_mask = mask & ~x_bit
            elem_blockers[x].append((ap_idx, other_mask))
    
    def can_add(x, S_mask):
        for _, other_mask in elem_blockers[x]:
            if (S_mask & other_mask) == other_mask:
                return False
        return True

    def get_blocking_aps(S_mask):
        blocking = []
        for i in range(num_aps):
            m = ap_masks[i]
            if bin(S_mask & m).count('1') == 3:
                blocking.append(i)
        return blocking

    best_size = 0
    best_mask = 0
    best_set = set()

    restart_count = 0
    while time.time() < deadline:
        restart_count += 1
        if time.time() > deadline:
            break
            
        # 1. Greedy Construction with multiple orderings
        S_mask = 0
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S_mask):
                S_mask |= (1 << (x - 1))
                
        current_size = bin(S_mask).count('1')
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            best_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
            
        # 2. Local Search Phase with improved swap strategy
        local_steps = 0
        max_local_steps = 30000
        no_improve_counter = 0
        max_no_improve = 800

        while local_steps < max_local_steps and time.time() < deadline:
            local_steps += 1
            if time.time() > deadline:
                break
                
            improved = False
            
            # A. Try to add any element (iterate in random order for diversity)
            added_any = False
            add_candidates = [x for x in range(1, n + 1) if not (S_mask & (1 << (x - 1)))]
            random.shuffle(add_candidates)
            for x in add_candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    current_size += 1
                    improved = True
                    added_any = True
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = S_mask
                        best_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
                        
            if added_any:
                no_improve_counter = 0
                continue
                
            # B. No additions possible. Try swaps.
            blocking_aps = get_blocking_aps(S_mask)
            
            if not blocking_aps:
                break

            # Try multiple random blocking APs before giving up
            found_swap = False
            for _ in range(10):
                b_idx = random.choice(blocking_aps)
                ap = ap_list[b_idx]
                
                in_ap = []
                for x in ap:
                    if S_mask & (1 << (x - 1)):
                        in_ap.append(x)
                        
                if len(in_ap) != 3:
                    continue
                    
                # Try removing each element in the AP and see if we can add something
                for y in in_ap:
                    y_bit = 1 << (y - 1)
                    S_mask_temp = S_mask & ~y_bit
                    
                    added_x = None
                    for x in range(1, n + 1):
                        if not (S_mask_temp & (1 << (x - 1))):
                            if can_add(x, S_mask_temp):
                                added_x = x
                                break
                                
                    if added_x is not None:
                        S_mask = S_mask_temp | (1 << (added_x - 1))
                        improved = True
                        no_improve_counter = 0
                        found_swap = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = S_mask
                            best_set = {i+1 for i in range(n) if (S_mask >> i) & 1}
                        break
                        
                if found_swap:
                    break
                    
            if not found_swap:
                no_improve_counter += 1
                if no_improve_counter > max_no_improve:
                    break

    return sorted(best_set)
