import time
import random
import math

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
    
    # Precompute: for each element, the list of APs it belongs to
    elem_aps = [[] for _ in range(n + 1)]
    for ap_idx, ap in enumerate(ap_list):
        for x in ap:
            elem_aps[x].append(ap_idx)
    
    # Precompute for each AP, the 4 element bits
    ap_elems = [[x - 1 for x in ap] for ap in ap_list]  # 0-indexed bit positions
    
    def can_add(x, S_mask):
        for _, other_mask in elem_blockers[x]:
            if (S_mask & other_mask) == other_mask:
                return False
        return True
    
    def count_blocking_aps(S_mask):
        """Count how many APs have exactly 3 elements in S_mask."""
        count = 0
        for i in range(num_aps):
            m = ap_masks[i]
            # Fast popcount using int.bit_count() (Python 3.10+)
            if (S_mask & m).bit_count() == 3:
                count += 1
        return count
    
    def get_blocking_aps_fast(S_mask, ap_count_tracker):
        """Get list of AP indices that have exactly 3 elements in S_mask.
        Uses the tracker array where ap_count_tracker[i] = number of elements of AP i in S_mask."""
        blocking = []
        for i in range(num_aps):
            if ap_count_tracker[i] == 3:
                blocking.append(i)
        return blocking
    
    def build_ap_count_tracker(S_mask):
        """Build array where tracker[i] = count of elements of AP i in S_mask."""
        tracker = [0] * num_aps
        for x in range(1, n + 1):
            if S_mask & (1 << (x - 1)):
                for ap_idx in elem_aps[x]:
                    tracker[ap_idx] += 1
        return tracker
    
    def update_tracker_remove(tracker, x):
        """Update tracker after removing element x from S."""
        for ap_idx in elem_aps[x]:
            tracker[ap_idx] -= 1
    
    def update_tracker_add(tracker, x):
        """Update tracker after adding element x to S."""
        for ap_idx in elem_aps[x]:
            tracker[ap_idx] += 1
    
    best_size = 0
    best_mask = 0
    best_set = set()
    
    def update_best(S_mask, current_size):
        nonlocal best_size, best_mask, best_set
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
            best_set = {i + 1 for i in range(n) if (S_mask >> i) & 1}
    
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
                
        current_size = S_mask.bit_count()
        update_best(S_mask, current_size)
        
        # Build tracker for local search
        tracker = build_ap_count_tracker(S_mask)
        
        # 2. Local Search Phase with simulated annealing + swap strategy
        local_steps = 0
        max_local_steps = 50000
        no_improve_counter = 0
        max_no_improve = 2000
        
        # Simulated annealing parameters
        temperature = 1.0
        cooling = 0.9995
        min_temp = 0.01
        
        while local_steps < max_local_steps and time.time() < deadline:
            local_steps += 1
            if time.time() > deadline:
                break
            
            improved = False
            
            # A. Try to add any element
            added_any = False
            add_candidates = [x for x in range(1, n + 1) if not (S_mask & (1 << (x - 1)))]
            random.shuffle(add_candidates)
            for x in add_candidates:
                if can_add(x, S_mask):
                    S_mask |= (1 << (x - 1))
                    update_tracker_add(tracker, x)
                    current_size += 1
                    improved = True
                    added_any = True
                    update_best(S_mask, current_size)
            
            if added_any:
                no_improve_counter = 0
                continue
            
            # B. No additions possible. Try swaps and removals.
            blocking_aps = get_blocking_aps_fast(S_mask, tracker)
            
            if not blocking_aps:
                break
            
            found_move = False
            
            # Strategy 1: Try swaps (remove one from blocking AP, add another)
            for _ in range(20):
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
                    
                    # Check if removing y actually helps (reduces blocking)
                    # Find a candidate to add
                    removed_x = None
                    for x in range(1, n + 1):
                        if not (S_mask_temp & (1 << (x - 1))):
                            if can_add(x, S_mask_temp):
                                removed_x = x
                                break
                                
                    if removed_x is not None:
                        # This swap maintains size, try it
                        S_mask = S_mask_temp | (1 << (removed_x - 1))
                        update_tracker_remove(tracker, y)
                        update_tracker_add(tracker, removed_x)
                        improved = True
                        no_improve_counter = 0
                        found_move = True
                        break
                        
                if found_move:
                    break
            
            # Strategy 2: Simulated annealing - try removing an element (accept with probability)
            if not found_move and temperature > min_temp:
                # Pick a random element to remove
                if current_size > 0:
                    # Pick from elements in blocking APs for targeted removal
                    blocking_elements = set()
                    for b_idx in blocking_aps[:50]:  # Limit to first 50 blocking APs
                        for x in ap_list[b_idx]:
                            if S_mask & (1 << (x - 1)):
                                blocking_elements.add(x)
                    
                    if blocking_elements:
                        y = random.choice(list(blocking_elements))
                        y_bit = 1 << (y - 1)
                        S_mask_temp = S_mask & ~y_bit
                        
                        # After removal, check if we can add more elements
                        added_count = 0
                        for x in range(1, n + 1):
                            if not (S_mask_temp & (1 << (x - 1))):
                                if can_add(x, S_mask_temp):
                                    S_mask_temp |= (1 << (x - 1))
                                    added_count += 1
                        
                        net_change = added_count - 1
                        # Accept if net change >= 0, or with probability exp(net_change / temperature) if negative
                        if net_change >= 0:
                            S_mask = S_mask_temp
                            update_tracker_remove(tracker, y)
                            for x in range(1, n + 1):
                                if (S_mask >> (x - 1)) & 1:
                                    pass  # tracker already updated for y
                            # Rebuild tracker for simplicity
                            tracker = build_ap_count_tracker(S_mask)
                            current_size = S_mask.bit_count()
                            improved = True
                            no_improve_counter = 0
                            update_best(S_mask, current_size)
                        else:
                            # Accept with probability exp(net_change / temperature)
                            if random.random() < math.exp(net_change / temperature):
                                S_mask = S_mask_temp
                                tracker = build_ap_count_tracker(S_mask)
                                current_size = S_mask.bit_count()
                                no_improve_counter = 0
            
            if not found_move and not improved:
                no_improve_counter += 1
                temperature *= cooling
                if no_improve_counter > max_no_improve:
                    break
            
            # Periodically rebuild tracker to avoid drift
            if local_steps % 500 == 0:
                tracker = build_ap_count_tracker(S_mask)
                current_size = S_mask.bit_count()

    return sorted(best_set)
