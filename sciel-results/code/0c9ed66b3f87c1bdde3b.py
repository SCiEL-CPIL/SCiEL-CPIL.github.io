"""SA + greedy repair for 4-AP-free set maximizing 3-AP count."""
import time
import random
import math

N = 256
TIME_LIMIT = 85.0  # safety margin below 90s

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    # Precompute all 4-APs and 3-APs (0-indexed) as bitmasks
    four_ap_masks = []
    four_aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(0, N - 3 * d):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            four_aps.append(ap)
            mask = 0
            for x in ap:
                mask |= (1 << x)
            four_ap_masks.append(mask)
    
    three_ap_masks = []
    three_aps = []
    for d in range(1, (N - 2) // 2 + 1):
        for a in range(0, N - 2 * d):
            ap = (a, a + d, a + 2 * d)
            three_aps.append(ap)
            mask = 0
            for x in ap:
                mask |= (1 << x)
            three_ap_masks.append(mask)
    
    # For each element, which 4-AP indices does it participate in?
    elem_to_4ap_indices = [[] for _ in range(N)]
    for i, ap in enumerate(four_aps):
        for x in ap:
            elem_to_4ap_indices[x].append(i)
            
    # For each element, count how many 3-APs it participates in (weight)
    elem_3ap_count = [0] * N
    for ap in three_aps:
        for x in ap:
            elem_3ap_count[x] += 1
            
    def count_3aps(mask):
        """Count 3-APs in selected set (mask is an integer bitmask)."""
        cnt = 0
        for m in three_ap_masks:
            if (mask & m) == m:
                cnt += 1
        return cnt
        
    def count_4ap_violations(mask):
        """Count 4-APs fully contained in mask."""
        cnt = 0
        for m in four_ap_masks:
            if (mask & m) == m:
                cnt += 1
        return cnt
        
    def creates_4ap(mask, x):
        """Check if adding x to mask creates a 4-AP."""
        new_mask = mask | (1 << x)
        for idx in elem_to_4ap_indices[x]:
            if (new_mask & four_ap_masks[idx]) == four_ap_masks[idx]:
                return True
        return False

    def greedy_construct():
        """Greedy: add elements in order of decreasing 3-AP weight, skip if creates 4-AP."""
        mask = 0
        # Sort by weight descending, break ties by index
        order = sorted(range(N), key=lambda x: (-elem_3ap_count[x], x))
        for x in order:
            if not creates_4ap(mask, x):
                mask |= (1 << x)
        return mask
        
    def greedy_repair(mask):
        """Remove elements to eliminate all 4-APs, keeping as many as possible."""
        mask = mask
        changed = True
        while changed:
            changed = False
            for i, m in enumerate(four_ap_masks):
                if (mask & m) == m:
                    # Find the element in this 4-AP with lowest weight that is in mask
                    ap = four_aps[i]
                    worst = min((x for x in ap if (mask >> x) & 1), key=lambda x: elem_3ap_count[x])
                    mask &= ~(1 << worst)
                    changed = True
                    break
        return mask
        
    def local_search(mask, max_iters=100):
        """Hill-climbing: try adds, removes, swaps to improve 3-AP count while maintaining no 4-APs."""
        current_score = count_3aps(mask)
        
        for _ in range(max_iters):
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            
            improved = False
            
            # Try adding an element
            candidates = [x for x in range(N) if not ((mask >> x) & 1)]
            rng.shuffle(candidates)
            for x in candidates:
                if not creates_4ap(mask, x):
                    new_mask = mask | (1 << x)
                    new_score = count_3aps(new_mask)
                    if new_score > current_score:
                        mask = new_mask
                        current_score = new_score
                        improved = True
                        break
            
            if improved:
                continue
            
            # Try removing an element
            sel_list = [x for x in range(N) if (mask >> x) & 1]
            rng.shuffle(sel_list)
            for x in sel_list:
                new_mask = mask & ~(1 << x)
                new_score = count_3aps(new_mask)
                if new_score > current_score:
                    mask = new_mask
                    current_score = new_score
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try swap: remove one, add another
            sel_list = [x for x in range(N) if (mask >> x) & 1]
            rng.shuffle(sel_list)
            for x_remove in sel_list:
                if time.time() - start_time > TIME_LIMIT - 3:
                    return mask, current_score
                candidates = [x for x in range(N) if not ((mask >> x) & 1)]
                rng.shuffle(candidates)
                for x_add in candidates:
                    new_mask = (mask & ~(1 << x_remove)) | (1 << x_add)
                    # Check 4-AP violations only for x_add
                    if creates_4ap(new_mask & ~(1 << x_add), x_add):
                        continue
                    
                    new_score = count_3aps(new_mask)
                    if new_score > current_score:
                        mask = new_mask
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        return mask, current_score
    
    # Phase 1: Greedy construction
    best_mask = greedy_construct()
    best_score = count_3aps(best_mask)
    
    # Phase 2: Local search on greedy solution
    best_mask, best_score = local_search(best_mask, max_iters=200)
    
    # Phase 3: Simulated Annealing with toggle moves
    # SA can temporarily allow 4-AP violations (penalized) to escape local optima
    current_mask = best_mask
    current_score = best_score
    
    T_start = 10.0
    T_end = 0.01
    alpha = 0.97  # cooling rate
    T = T_start
    
    sa_iter = 0
    max_sa_iters = 5000
    
    while T > T_end and sa_iter < max_sa_iters:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        
        sa_iter += 1
        
        # Generate a toggle move: add, remove, or swap
        move_type = rng.random()
        new_mask = current_mask
        
        if move_type < 0.4:
            # Add an element
            x = rng.randint(0, N - 1)
            if not ((current_mask >> x) & 1):
                new_mask |= (1 << x)
        elif move_type < 0.7:
            # Remove an element
            if current_mask:
                # Pick a random set bit
                # Efficient way: pick random index, if not set, retry or pick from list
                # For simplicity in SA, list is fine if size is small, but let's be safe
                sel_list = [x for x in range(N) if (current_mask >> x) & 1]
                if sel_list:
                    x = rng.choice(sel_list)
                    new_mask &= ~(1 << x)
        else:
            # Swap: remove one, add another
            if current_mask:
                sel_list = [x for x in range(N) if (current_mask >> x) & 1]
                if sel_list:
                    x_remove = rng.choice(sel_list)
                    x_add = rng.randint(0, N - 1)
                    if not ((current_mask >> x_add) & 1):
                        new_mask = (current_mask & ~(1 << x_remove)) | (1 << x_add)
        
        # Compute new score with penalty for 4-APs
        new_3ap_count = count_3aps(new_mask)
        new_4ap_count = count_4ap_violations(new_mask)
        new_score = new_3ap_count - 100 * new_4ap_count
        
        old_4ap_count = count_4ap_violations(current_mask)
        old_3ap_count = count_3aps(current_mask)
        old_score_val = old_3ap_count - 100 * old_4ap_count
        
        delta = new_score - old_score_val
        
        if delta >= 0 or rng.random() < math.exp(delta / T):
            current_mask = new_mask
        
        # Periodic greedy repair every 100 iterations to maintain feasibility
        if sa_iter % 100 == 0:
            current_mask = greedy_repair(current_mask)
            
        # Track best feasible solution
        if count_4ap_violations(current_mask) == 0:
            feasible_score = count_3aps(current_mask)
            if feasible_score > best_score:
                best_score = feasible_score
                best_mask = current_mask
        
        T *= alpha
    
    # Final local search on best solution
    if time.time() - start_time < TIME_LIMIT - 3:
        best_mask, best_score = local_search(best_mask, max_iters=200)
    
    # Verify no 4-APs
    assert count_4ap_violations(best_mask) == 0, "4-AP violation found!"
    
    # Convert mask to list of 1-based indices
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return result
