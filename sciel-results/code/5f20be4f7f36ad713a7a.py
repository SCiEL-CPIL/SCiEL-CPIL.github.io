import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    N = 256
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
            
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map elements to APs containing them
    elem_to_3aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(three_aps):
        for x in ap:
            elem_to_3aps[x].append(i)
            
    elem_to_4aps = [[] for _ in range(N + 1)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4aps[x].append(i)

    # Precompute bitmask for each element's position in 3-APs and 4-APs
    # For each element x, precompute the bitmask of other elements in each AP it belongs to
    # This allows fast delta computation using bitwise AND

    def count_3aps_bitmask(mask):
        """Count 3-APs using a bitmask (bit i set means element i+1 is in the set)."""
        count = 0
        for ap in three_aps:
            a, b, c = ap
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                count += 1
        return count

    def count_4aps_bitmask(mask):
        """Count 4-APs using a bitmask."""
        count = 0
        for ap in four_aps:
            a, b, c, d = ap
            if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1 and (mask >> d) & 1:
                count += 1
        return count

    def _exp(x):
        """Safe exponential for simulated annealing."""
        if x < -100:
            return 0.0
        if x > 100:
            return 1e100
        return math.exp(x)

    def is_valid_add_bitmask(mask, x):
        """Check if adding x to mask creates a 4-AP. Returns True if valid."""
        for ap_idx in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if x == a:
                if (mask >> b) & 1 and (mask >> c) & 1 and (mask >> d) & 1:
                    return False
            elif x == b:
                if (mask >> a) & 1 and (mask >> c) & 1 and (mask >> d) & 1:
                    return False
            elif x == c:
                if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> d) & 1:
                    return False
            else:
                if (mask >> a) & 1 and (mask >> b) & 1 and (mask >> c) & 1:
                    return False
        return True

    def delta_add_bitmask(mask, x):
        """Change in 3-AP count if we add x to mask."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if (mask >> b) & 1 and (mask >> c) & 1:
                    delta += 1
            elif x == b:
                if (mask >> a) & 1 and (mask >> c) & 1:
                    delta += 1
            else:
                if (mask >> a) & 1 and (mask >> b) & 1:
                    delta += 1
        return delta

    def delta_remove_bitmask(mask, x):
        """Change in 3-AP count if we remove x from mask."""
        delta = 0
        for ap_idx in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if x == a:
                if (mask >> b) & 1 and (mask >> c) & 1:
                    delta += 1
            elif x == b:
                if (mask >> a) & 1 and (mask >> c) & 1:
                    delta += 1
            else:
                if (mask >> a) & 1 and (mask >> b) & 1:
                    delta += 1
        return -delta

    def is_valid_swap_bitmask(mask, x_out, x_in):
        """Check if swapping x_out for x_in is valid (no 4-APs)."""
        for ap_idx in elem_to_4aps[x_in]:
            a, b, c, d = four_aps[ap_idx]
            if x_in == a:
                others = [b, c, d]
            elif x_in == b:
                others = [a, c, d]
            elif x_in == c:
                others = [a, b, d]
            else:
                others = [a, b, c]
            
            all_in_mask = True
            x_out_in_others = False
            for y in others:
                if not ((mask >> y) & 1):
                    all_in_mask = False
                    break
                if y == x_out:
                    x_out_in_others = True
            
            if all_in_mask and not x_out_in_others:
                return False
        return True

    def greedy_construct_bitmask(random_seed):
        random.seed(random_seed)
        mask = 0
        available = list(range(1, N + 1))
        
        while available:
            candidates = []
            for x in available:
                if is_valid_add_bitmask(mask, x):
                    d = delta_add_bitmask(mask, x)
                    candidates.append((d, x))
            
            if not candidates:
                break
                
            candidates.sort(key=lambda item: item[0], reverse=True)
            
            top_k = min(5, len(candidates))
            if random.random() < 0.8:
                best_x = candidates[0][1]
            else:
                best_x = random.choice([c[1] for c in candidates[:top_k]])
            
            mask |= (1 << best_x)
            available.remove(best_x)
            
        return mask

    def local_search_1move_bitmask(mask):
        """Try add and remove moves to improve 3-AP count."""
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if (mask >> x) & 1:
                    continue
                if not is_valid_add_bitmask(mask, x):
                    continue
                gain = delta_add_bitmask(mask, x)
                if gain > 0:
                    mask |= (1 << x)
                    improved = True
                    break
            if not improved:
                for x in range(1, N + 1):
                    if not ((mask >> x) & 1):
                        continue
                    loss = delta_remove_bitmask(mask, x)
                    if loss >= 0:
                        mask &= ~(1 << x)
                        improved = True
                        break
        return mask

    def local_search_swap_bitmask(mask):
        """Try swap moves to improve 3-AP count."""
        improved = True
        while improved:
            improved = False
            best_gain = 0
            best_swap = None
            for x_out in range(1, N + 1):
                if not ((mask >> x_out) & 1):
                    continue
                loss = -delta_remove_bitmask(mask, x_out)
                if loss == 0:
                    continue
                for x_in in range(1, N + 1):
                    if (mask >> x_in) & 1:
                        continue
                    if not is_valid_swap_bitmask(mask, x_out, x_in):
                        continue
                    gain = 0
                    for ap_idx in elem_to_3aps[x_in]:
                        a, b, c = three_aps[ap_idx]
                        if x_in == a:
                            if ((mask >> b) & 1) and ((mask >> c) & 1) and b != x_out and c != x_out:
                                gain += 1
                        elif x_in == b:
                            if ((mask >> a) & 1) and ((mask >> c) & 1) and a != x_out and c != x_out:
                                gain += 1
                        else:
                            if ((mask >> a) & 1) and ((mask >> b) & 1) and a != x_out and b != x_out:
                                gain += 1
                    
                    net = gain - loss
                    if net > best_gain:
                        best_gain = net
                        best_swap = (x_out, x_in)
            if best_swap is not None:
                x_out, x_in = best_swap
                mask &= ~(1 << x_out)
                mask |= (1 << x_in)
                improved = True
        return mask

    def repair_bitmask(mask, max_iterations=50):
        """Remove elements to break all 4-APs."""
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            found = False
            for ap in four_aps:
                a, b, c, d = ap
                if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                    # Remove the element with the lowest 3-AP contribution
                    min_loss = None
                    min_elem = None
                    for x in [a, b, c, d]:
                        loss = -delta_remove_bitmask(mask, x)
                        if min_loss is None or loss < min_loss:
                            min_loss = loss
                            min_elem = x
                    mask &= ~(1 << min_elem)
                    found = True
                    break
            if not found:
                break
        return mask

    def simulated_annealing_bitmask(mask, t0=1.0, t1=0.01, steps=2000, penalty=50):
        """Simulated annealing that can temporarily accept 4-AP violations."""
        current_score = count_3aps_bitmask(mask)
        current_4aps = count_4aps_bitmask(mask)
        current_obj = current_score - penalty * current_4aps
        best_mask = mask
        best_obj = current_obj
        
        for step in range(steps):
            if time.time() > deadline:
                break
            T = t0 * (t1 / t0) ** (step / steps)
            
            move_type = random.random()
            
            if move_type < 0.4:
                # Add a random element
                candidates = [x for x in range(1, N + 1) if not ((mask >> x) & 1)]
                if not candidates:
                    continue
                x = random.choice(candidates)
                new_mask = mask | (1 << x)
                delta_3 = delta_add_bitmask(mask, x)
                # Check if this creates a 4-AP
                delta_4 = 0
                for ap_idx in elem_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if x == a:
                        if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 += 1
                    elif x == b:
                        if ((mask >> a) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 += 1
                    elif x == c:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> d) & 1):
                            delta_4 += 1
                    else:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                            delta_4 += 1
                new_score = current_score + delta_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
            elif move_type < 0.7:
                # Remove a random element
                in_set = [x for x in range(1, N + 1) if (mask >> x) & 1]
                if not in_set:
                    continue
                x = random.choice(in_set)
                new_mask = mask & ~(1 << x)
                delta_3 = delta_remove_bitmask(mask, x)
                # Check if this breaks a 4-AP
                delta_4 = 0
                for ap_idx in elem_to_4aps[x]:
                    a, b, c, d = four_aps[ap_idx]
                    if x == a:
                        if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    elif x == b:
                        if ((mask >> a) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    elif x == c:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    else:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                            delta_4 -= 1
                new_score = current_score + delta_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
            else:
                # Swap
                in_set = [x for x in range(1, N + 1) if (mask >> x) & 1]
                if not in_set:
                    continue
                x_out = random.choice(in_set)
                candidates_in = [x for x in range(1, N + 1) if not ((mask >> x) & 1)]
                if not candidates_in:
                    continue
                x_in = random.choice(candidates_in)
                
                # Calculate changes
                loss_3 = -delta_remove_bitmask(mask, x_out)
                gain_3 = 0
                for ap_idx in elem_to_3aps[x_in]:
                    a, b, c = three_aps[ap_idx]
                    if x_in == a:
                        if ((mask >> b) & 1) and ((mask >> c) & 1) and b != x_out and c != x_out:
                            gain_3 += 1
                    elif x_in == b:
                        if ((mask >> a) & 1) and ((mask >> c) & 1) and a != x_out and c != x_out:
                            gain_3 += 1
                    else:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and a != x_out and b != x_out:
                            gain_3 += 1
                
                # Check 4-AP changes
                delta_4 = 0
                # 4-APs broken by removing x_out
                for ap_idx in elem_to_4aps[x_out]:
                    a, b, c, d = four_aps[ap_idx]
                    if x_out == a:
                        if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    elif x_out == b:
                        if ((mask >> a) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    elif x_out == c:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> d) & 1):
                            delta_4 -= 1
                    else:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1):
                            delta_4 -= 1
                # 4-APs created by adding x_in
                for ap_idx in elem_to_4aps[x_in]:
                    a, b, c, d = four_aps[ap_idx]
                    if x_in == a:
                        if ((mask >> b) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1) and b != x_out and c != x_out and d != x_out:
                            delta_4 += 1
                    elif x_in == b:
                        if ((mask >> a) & 1) and ((mask >> c) & 1) and ((mask >> d) & 1) and a != x_out and c != x_out and d != x_out:
                            delta_4 += 1
                    elif x_in == c:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> d) & 1) and a != x_out and b != x_out and d != x_out:
                            delta_4 += 1
                    else:
                        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> c) & 1) and a != x_out and b != x_out and c != x_out:
                            delta_4 += 1
                
                new_score = current_score + gain_3 - loss_3
                new_4aps = current_4aps + delta_4
                new_obj = new_score - penalty * new_4aps
                new_mask = (mask & ~(1 << x_out)) | (1 << x_in)
            
            # Accept or reject
            delta_obj = new_obj - current_obj
            if delta_obj > 0 or random.random() < _exp(delta_obj / T):
                mask = new_mask
                current_score = new_score
                current_4aps = new_4aps
                current_obj = new_obj
                
                # Periodic repair
                if step % 100 == 0 and current_4aps > 0:
                    mask = repair_bitmask(mask)
                    current_score = count_3aps_bitmask(mask)
                    current_4aps = count_4aps_bitmask(mask)
                    current_obj = current_score - penalty * current_4aps
                
                if current_4aps == 0 and current_score > best_obj:
                    best_obj = current_score
                    best_mask = mask
        
        return best_mask

    def mask_to_list(mask):
        return [x for x in range(1, N + 1) if (mask >> x) & 1]

    best_mask = 0
    best_score = -1
    
    num_restarts = 30
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        mask = greedy_construct_bitmask(i)
        score = count_3aps_bitmask(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
        
        mask = local_search_1move_bitmask(mask)
        score = count_3aps_bitmask(mask)
        
        if score > best_score:
            best_score = score
            best_mask = mask
        
        if time.time() < deadline - 5:
            mask = local_search_swap_bitmask(mask)
            score = count_3aps_bitmask(mask)
            
            if score > best_score:
                best_score = score
                best_mask = mask
        
        if time.time() < deadline - 10:
            mask = simulated_annealing_bitmask(mask, t0=0.5, t1=0.01, steps=800, penalty=50)
            score = count_3aps_bitmask(mask)
            if count_4aps_bitmask(mask) == 0:
                if score > best_score:
                    best_score = score
                    best_mask = mask
    
    if best_mask and time.time() < deadline - 5:
        mask = simulated_annealing_bitmask(best_mask, t0=0.3, t1=0.005, steps=2000, penalty=50)
        score = count_3aps_bitmask(mask)
        if count_4aps_bitmask(mask) == 0:
            if score > best_score:
                best_score = score
                best_mask = mask
    
    return mask_to_list(best_mask)
