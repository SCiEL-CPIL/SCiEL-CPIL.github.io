"""Max-SAT style stochastic local search for 4-AP-free sets."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin from 90s limit
    n = 512
    
    # Precompute all 4-APs
    # A 4-AP is (a, a+d, a+2d, a+3d)
    # Store as list of tuples
    all_4aps = []
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # For each position, store which 4-APs it participates in
    # pos_to_aps[pos] = list of indices into all_4aps
    pos_to_aps = [[] for _ in range(n + 1)]
    for i, ap in enumerate(all_4aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    # Precompute for each 4-AP, the positions
    # ap_positions[i] = tuple of 4 positions
    ap_positions = [ap for ap in all_4aps]
    num_aps = len(all_4aps)
    
    # State: in_set is a list of bools (index 0 unused)
    # ap_count[i] = number of elements from all_4aps[i] that are in the set
    
    def init_state(in_set):
        """Initialize ap_count from in_set."""
        ap_count = [0] * num_aps
        for i in range(num_aps):
            a, b, c, d = ap_positions[i]
            cnt = 0
            if in_set[a]: cnt += 1
            if in_set[b]: cnt += 1
            if in_set[c]: cnt += 1
            if in_set[d]: cnt += 1
            ap_count[i] = cnt
        return ap_count
    
    def count_violations(ap_count):
        """Count number of violated 4-APs (all 4 elements present)."""
        return sum(1 for c in ap_count if c == 4)
    
    def add_element(x, in_set, ap_count):
        """Add x to set, update ap_count. Returns True if x was added."""
        if in_set[x]:
            return False
        in_set[x] = True
        for ap_idx in pos_to_aps[x]:
            ap_count[ap_idx] += 1
        return True
    
    def remove_element(x, in_set, ap_count):
        """Remove x from set, update ap_count. Returns True if x was removed."""
        if not in_set[x]:
            return False
        in_set[x] = False
        for ap_idx in pos_to_aps[x]:
            ap_count[ap_idx] -= 1
        return True
    
    def get_set_size(in_set):
        return sum(in_set)
    
    # Phase 1: Greedy construction with base-3 ordering
    def base3_sum(x):
        s = 0
        while x > 0:
            s += x % 3
            x //= 3
        return s
    
    order = sorted(range(1, n + 1), key=lambda x: (base3_sum(x), x))
    
    in_set = [False] * (n + 1)
    for x in order:
        if time.time() > deadline - 15:
            break
        # Check if adding x creates a violation
        # A violation occurs if after adding x, some 4-AP has all 4 elements
        # Before adding x, ap_count[ap_idx] for ap_idx in pos_to_aps[x] is at most 3
        # After adding x, it becomes at most 4
        # So we need to check if any ap_idx in pos_to_aps[x] has ap_count[ap_idx] == 3
        can_add = True
        for ap_idx in pos_to_aps[x]:
            if ap_count_check(in_set, ap_positions[ap_idx]):
                can_add = False
                break
        if can_add:
            in_set[x] = True
    
    # Build ap_count for the greedy solution
    ap_count = init_state(in_set)
    
    best_size = get_set_size(in_set)
    best_in_set = list(in_set)
    
    # Phase 2: Stochastic Local Search (WalkSAT-style)
    # We want to maximize |S| subject to no 4-AP violations
    # Energy = -|S| + M * (# violations), where M is a large penalty
    # We minimize energy, or equivalently maximize |S| - M * (# violations)
    
    random.seed(42)
    
    # Start from the greedy solution
    current_in_set = list(in_set)
    current_ap_count = list(ap_count)
    current_size = get_set_size(current_in_set)
    current_violations = count_violations(current_ap_count)
    
    # We'll use a two-phase approach:
    # Phase 2a: Try to add elements (increase size)
    # Phase 2b: If stuck, remove elements to allow adding more
    
    # Track best solution
    best_size = get_set_size(best_in_set)
    
    # Precompute list of all elements
    all_elements = list(range(1, n + 1))
    
    # Local search with random restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        # Perturb: start from best solution and remove some random elements
        if restart == 0:
            current_in_set = list(best_in_set)
        else:
            current_in_set = list(best_in_set)
            # Remove 3-8 random elements
            num_remove = random.randint(3, 8)
            in_set_list = [x for x in range(1, n + 1) if current_in_set[x]]
            random.shuffle(in_set_list)
            for x in in_set_list[:num_remove]:
                current_in_set[x] = False
        
        current_ap_count = init_state(current_in_set)
        current_size = get_set_size(current_in_set)
        current_violations = count_violations(current_ap_count)
        
        # Iterate: try to improve
        for iteration in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Try to add a random element not in set
            # Prefer elements that don't create violations
            missing = [x for x in range(1, n + 1) if not current_in_set[x]]
            if not missing:
                break
            
            # Score each missing element by how many violations it would create
            # and how many new 4-APs it would complete
            best_x = None
            best_score = -1
            
            # Sample up to 200 candidates for speed
            sample_size = min(200, len(missing))
            candidates = random.sample(missing, sample_size)
            
            for x in candidates:
                # Check if adding x creates any violation
                # A violation is created if any 4-AP containing x has 3 other elements in set
                creates_violation = False
                for ap_idx in pos_to_aps[x]:
                    a, b, c, d = ap_positions[ap_idx]
                    # Count how many of the other 3 are in set
                    other_count = 0
                    for pos in (a, b, c, d):
                        if pos != x and current_in_set[pos]:
                            other_count += 1
                    if other_count == 3:
                        creates_violation = True
                        break
                
                if not creates_violation:
                    best_x = x
                    best_score = 1
                    break
                else:
                    # Score based on how close to violation
                    # This is complex; just skip for now
                    pass
            
            if best_x is not None:
                # Add the element
                current_in_set[best_x] = True
                for ap_idx in pos_to_aps[best_x]:
                    current_ap_count[ap_idx] += 1
                current_size += 1
            else:
                # All sampled candidates would create violations
                # Try to remove a random element from the set to unblock
                in_set_list = [x for x in range(1, n + 1) if current_in_set[x]]
                if not in_set_list:
                    break
                x_remove = random.choice(in_set_list)
                current_in_set[x_remove] = False
                for ap_idx in pos_to_aps[x_remove]:
                    current_ap_count[ap_idx] -= 1
                current_size -= 1
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_in_set = list(current_in_set)
    
    # Final: greedy fill from best solution
    # Try to add any remaining elements that don't create violations
    current_in_set = list(best_in_set)
    current_ap_count = init_state(current_in_set)
    
    # Try all missing elements in base-3 order
    for x in order:
        if time.time() > deadline - 1:
            break
        if current_in_set[x]:
            continue
        # Check if adding x creates a violation
        can_add = True
        for ap_idx in pos_to_aps[x]:
            a, b, c, d = ap_positions[ap_idx]
            other_count = 0
            for pos in (a, b, c, d):
                if pos != x and current_in_set[pos]:
                    other_count += 1
            if other_count == 3:
                can_add = False
                break
        if can_add:
            current_in_set[x] = True
            for ap_idx in pos_to_aps[x]:
                current_ap_count[ap_idx] += 1
    
    if get_set_size(current_in_set) > best_size:
        best_size = get_set_size(current_in_set)
        best_in_set = list(current_in_set)
    
    # Convert to sorted list
    result = sorted([x for x in range(1, n + 1) if best_in_set[x]])
    
    # Verify: check no 4-APs
    final_in_set = [False] * (n + 1)
    for x in result:
        final_in_set[x] = True
    
    valid = True
    for i in range(num_aps):
        a, b, c, d = ap_positions[i]
        if final_in_set[a] and final_in_set[b] and final_in_set[c] and final_in_set[d]:
            valid = False
            break
    
    if not valid:
        # Fallback: remove elements from violated 4-APs
        # This shouldn't happen, but just in case
        # Remove one element from each violated 4-AP
        for i in range(num_aps):
            a, b, c, d = ap_positions[i]
            if final_in_set[a] and final_in_set[b] and final_in_set[c] and final_in_set[d]:
                # Remove the largest element
                final_in_set[d] = False
    
        result = sorted([x for x in range(1, n + 1) if final_in_set[x]])
    
    return result

def ap_count_check(in_set, ap):
    """Check if all 4 elements of ap are in in_set."""
    a, b, c, d = ap
    return in_set[a] and in_set[b] and in_set[c] and in_set[d]
# EVOLVE-BLOCK-END
