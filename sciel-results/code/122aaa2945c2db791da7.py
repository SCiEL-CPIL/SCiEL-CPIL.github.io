import time
import random

# EVOLVE-BLOCK-START

def construct():
    N = 512
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for evaluator overhead

    # Precompute all 4-APs: list of (a, a+d, a+2d, a+3d)
    # For each element, precompute which APs it participates in
    # Use bitmask: bit i set means element (i+1) is in the set
    
    # Precompute for each position x (0-indexed, so x=0 means element 1):
    # the list of APs containing x, and for each AP, the other 3 positions
    # This allows fast checking: for a candidate x, check all APs containing x,
    # and see if the other 3 are all in the set
    
    # Build AP index
    ap_by_elem = [[] for _ in range(N)]  # ap_by_elem[x] = list of (other1, other2, other3)
    
    for a in range(N):  # a is 0-indexed, element is a+1
        for d in range(1, N // 3 + 1):
            a3 = a + 3 * d
            if a3 >= N:
                break
            # AP: a, a+d, a+2d, a+3d (0-indexed)
            p0, p1, p2, p3 = a, a + d, a + 2 * d, a + 3 * d
            ap_by_elem[p0].append((p1, p2, p3))
            ap_by_elem[p1].append((p0, p2, p3))
            ap_by_elem[p2].append((p0, p1, p3))
            ap_by_elem[p3].append((p0, p1, p2))
    
    def can_add(mask, x):
        """Check if adding element x (0-indexed) to set represented by mask creates a 4-AP."""
        for (o1, o2, o3) in ap_by_elem[x]:
            if (mask >> o1) & 1 and (mask >> o2) & 1 and (mask >> o3) & 1:
                return False
        return True
    
    def greedy_with_order(order):
        """Greedy construction with a given order of elements."""
        mask = 0
        for x in order:
            if can_add(mask, x):
                mask |= (1 << x)
        return mask
    
    def count_bits(mask):
        return bin(mask).count('1')
    
    # Strategy 1: Natural order greedy
    natural_order = list(range(N))
    mask1 = greedy_with_order(natural_order)
    best_mask = mask1
    best_size = count_bits(mask1)
    
    # Strategy 2: Reverse order greedy
    reverse_order = list(range(N - 1, -1, -1))
    mask2 = greedy_with_order(reverse_order)
    if count_bits(mask2) > best_size:
        best_size = count_bits(mask2)
        best_mask = mask2
    
    # Strategy 3: Random orders (multiple restarts)
    random.seed(42)
    
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        order = list(range(N))
        random.shuffle(order)
        m = greedy_with_order(order)
        sz = count_bits(m)
        if sz > best_size:
            best_size = sz
            best_mask = m
    
    # Local search on best_mask: try to add elements not in the set
    # Then try swaps: remove one element, add one or more others
    current_mask = best_mask
    current_size = best_size
    
    # Get list of elements not in current set
    not_in_set = [x for x in range(N) if not (current_mask >> x) & 1]
    
    # Phase 1: Try to add any element from not_in_set
    while time.time() - start_time < time_limit:
        # Try adding elements
        made_progress = False
        random.shuffle(not_in_set)
        
        for x in not_in_set:
            if time.time() - start_time > time_limit:
                break
            if can_add(current_mask, x):
                current_mask |= (1 << x)
                not_in_set.remove(x)
                current_size += 1
                made_progress = True
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
        
        if made_progress:
            continue
        
        # Phase 2: Stuck. Try swaps.
        # Remove a random element, then try to add others
        if current_size == 0:
            break
            
        # Try a few random swaps
        swap_success = False
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Pick random element to remove
            in_set = [x for x in range(N) if (current_mask >> x) & 1]
            if not in_set:
                break
            y = random.choice(in_set)
            
            # Remove y
            new_mask = current_mask & ~(1 << y)
            new_not_in = list(not_in_set) + [y]
            
            # Try to add elements from new_not_in
            added_count = 0
            random.shuffle(new_not_in)
            for z in new_not_in:
                if z == y:
                    continue
                if can_add(new_mask, z):
                    new_mask |= (1 << z)
                    added_count += 1
            
            # Accept if we added at least as many as we removed (1)
            if added_count >= 1:
                current_mask = new_mask
                current_size += added_count - 1
                # Update not_in_set
                not_in_set = [x for x in range(N) if not (current_mask >> x) & 1]
                swap_success = True
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                break
            else:
                # Revert (we didn't actually change anything since we used new_mask)
                pass
        
        if not swap_success:
            # Try a more aggressive move: remove 2 elements, try to add 3
            aggressive_success = False
            in_set = [x for x in range(N) if (current_mask >> x) & 1]
            if len(in_set) >= 2:
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    y1, y2 = random.sample(in_set, 2)
                    new_mask = current_mask & ~(1 << y1) & ~(1 << y2)
                    new_not_in = list(not_in_set) + [y1, y2]
                    
                    added_count = 0
                    random.shuffle(new_not_in)
                    for z in new_not_in:
                        if z in (y1, y2):
                            continue
                        if can_add(new_mask, z):
                            new_mask |= (1 << z)
                            added_count += 1
                    
                    if added_count >= 2:
                        current_mask = new_mask
                        current_size += added_count - 2
                        not_in_set = [x for x in range(N) if not (current_mask >> x) & 1]
                        aggressive_success = True
                        if current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask
                        break
            
            if not aggressive_success and not swap_success:
                # Try a full restart with different random seed
                random.seed(random.randint(0, 2**31))
                order = list(range(N))
                random.shuffle(order)
                m = greedy_with_order(order)
                sz = count_bits(m)
                if sz > current_size:
                    current_mask = m
                    current_size = sz
                    not_in_set = [x for x in range(N) if not (current_mask >> x) & 1]
                    if current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask
                else:
                    # Give up and break to avoid infinite loop
                    break
    
    # Convert best_mask to sorted list
    result = [x + 1 for x in range(N) if (best_mask >> x) & 1]
    return sorted(result)

# EVOLVE-BLOCK-END
