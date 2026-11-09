import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 118  # Safety margin before 150s limit
    
    n = 16
    all_triples = list(itertools.combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # For each set of 6 points, precompute the 15 ways to partition into 3 pairs
    # and the 8 transversal triples for each such partition
    six_points_configs = {}
    for six in itertools.combinations(range(n), 6):
        p0, p1, p2, p3, p4, p5 = six
        pairings = [
            [(p0,p1),(p2,p3),(p4,p5)],
            [(p0,p1),(p2,p4),(p3,p5)],
            [(p0,p1),(p2,p5),(p3,p4)],
            [(p0,p2),(p1,p3),(p4,p5)],
            [(p0,p2),(p1,p4),(p3,p5)],
            [(p0,p2),(p1,p5),(p3,p4)],
            [(p0,p3),(p1,p2),(p4,p5)],
            [(p0,p3),(p1,p4),(p2,p5)],
            [(p0,p3),(p1,p5),(p2,p4)],
            [(p0,p4),(p1,p2),(p3,p5)],
            [(p0,p4),(p1,p3),(p2,p5)],
            [(p0,p4),(p1,p5),(p2,p3)],
            [(p0,p5),(p1,p2),(p3,p4)],
            [(p0,p5),(p1,p3),(p2,p4)],
            [(p0,p5),(p1,p4),(p2,p3)],
        ]
        configs = []
        for pairs in pairings:
            (a,b),(c,d),(e,f) = pairs
            transversal = [
                tuple(sorted((a,c,e))), tuple(sorted((a,c,f))),
                tuple(sorted((a,d,e))), tuple(sorted((a,d,f))),
                tuple(sorted((b,c,e))), tuple(sorted((b,c,f))),
                tuple(sorted((b,d,e))), tuple(sorted((b,d,f))),
            ]
            configs.append(transversal)
        six_points_configs[six] = configs
    
    # Precompute: for each triple, list of (six_idx, config_idx) where it appears
    # Use indices for faster lookup
    six_list = list(six_points_configs.keys())
    six_to_idx = {six: i for i, six in enumerate(six_list)}
    
    triple_to_constraints = [[] for _ in range(num_triples)]
    for six, configs in six_points_configs.items():
        six_idx = six_to_idx[six]
        for config_idx, transversal in enumerate(configs):
            for t in transversal:
                t_idx = triple_index[t]
                triple_to_constraints[t_idx].append((six_idx, config_idx))
    
    # Count how many constraints each triple is in
    constraint_count = [len(constraints) for constraints in triple_to_constraints]
    
    # Use bitmask for fast membership testing
    # Each triple has an index 0..1715, so we need 1716 bits
    # Python ints can handle arbitrary precision, so we use a single int as bitmask
    
    def is_valid_fast(mask, candidate_idx):
        """Check if adding candidate triple (by index) to mask creates a violation."""
        if (mask >> candidate_idx) & 1:
            return False
        for six_idx, config_idx in triple_to_constraints[candidate_idx]:
            transversal = six_points_configs[six_list[six_idx]][config_idx]
            # Check if all 8 transversal triples are in mask (or is candidate)
            all_present = True
            for t in transversal:
                t_idx = triple_index[t]
                if t_idx == candidate_idx:
                    continue
                if not ((mask >> t_idx) & 1):
                    all_present = False
                    break
            if all_present:
                return False
        return True
    
    def greedy_order(order_indices):
        mask = 0
        count = 0
        for t_idx in order_indices:
            if is_valid_fast(mask, t_idx):
                mask |= (1 << t_idx)
                count += 1
        return mask, count
    
    def mask_to_set(mask):
        return [all_triples[i] for i in range(num_triples) if (mask >> i) & 1]
    
    def local_search_improved(mask, max_steps=10000):
        """Improved local search with add, swap, and 2-swap moves using bitmask."""
        current_mask = mask
        best_mask = mask
        best_size = bin(mask).count('1')
        current_size = best_size
        
        # Precompute list of all triple indices for random selection
        all_indices = list(range(num_triples))
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Try adding a random triple
                t_idx = random.choice(all_indices)
                if not ((current_mask >> t_idx) & 1):
                    if is_valid_fast(current_mask, t_idx):
                        current_mask |= (1 << t_idx)
                        current_size += 1
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size
                        continue
            
            elif move_type < 0.85:
                # Try swapping: remove one, add another
                # Find a random set bit to remove
                if current_size > 0:
                    # Pick random position and find next set bit
                    pos = random.randint(0, current_size - 1)
                    temp = current_mask
                    for _ in range(pos):
                        temp &= temp - 1  # clear lowest set bit
                    remove_idx = (temp & -temp).bit_length() - 1
                    
                    current_mask &= ~(1 << remove_idx)
                    
                    found = False
                    for _ in range(30):
                        add_idx = random.choice(all_indices)
                        if add_idx != remove_idx and not ((current_mask >> add_idx) & 1):
                            if is_valid_fast(current_mask, add_idx):
                                current_mask |= (1 << add_idx)
                                found = True
                                break
                    
                    if found:
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size
                    else:
                        current_mask |= (1 << remove_idx)
            
            else:
                # Try 2-swap: remove two, add two
                if current_size >= 2:
                    # Pick two random set bits
                    pos1 = random.randint(0, current_size - 1)
                    pos2 = random.randint(0, current_size - 1)
                    if pos1 == pos2:
                        pos2 = (pos2 + 1) % current_size
                    
                    temp1 = current_mask
                    for _ in range(pos1):
                        temp1 &= temp1 - 1
                    r1_idx = (temp1 & -temp1).bit_length() - 1
                    
                    temp2 = current_mask & ~(1 << r1_idx)
                    # Adjust pos2 if needed
                    temp2_count = bin(temp2).count('1')
                    if pos2 >= temp2_count:
                        pos2 = random.randint(0, temp2_count - 1)
                    for _ in range(pos2):
                        temp2 &= temp2 - 1
                    r2_idx = (temp2 & -temp2).bit_length() - 1
                    
                    current_mask &= ~(1 << r1_idx)
                    current_mask &= ~(1 << r2_idx)
                    
                    found1 = False
                    found2 = False
                    a1_idx = -1
                    a2_idx = -1
                    
                    for _ in range(30):
                        a1_candidate = random.choice(all_indices)
                        if a1_candidate != r1_idx and a1_candidate != r2_idx and not ((current_mask >> a1_candidate) & 1):
                            if is_valid_fast(current_mask, a1_candidate):
                                current_mask |= (1 << a1_candidate)
                                found1 = True
                                a1_idx = a1_candidate
                                break
                    
                    if found1:
                        for _ in range(30):
                            a2_candidate = random.choice(all_indices)
                            if a2_candidate != r1_idx and a2_candidate != r2_idx and a2_candidate != a1_idx and not ((current_mask >> a2_candidate) & 1):
                                if is_valid_fast(current_mask, a2_candidate):
                                    current_mask |= (1 << a2_candidate)
                                    found2 = True
                                    a2_idx = a2_candidate
                                    break
                    
                    if found1 and found2:
                        if current_size > best_size:
                            best_mask = current_mask
                            best_size = current_size
                    else:
                        # Revert
                        if found1:
                            current_mask &= ~(1 << a1_idx)
                        if found2:
                            current_mask &= ~(1 << a2_idx)
                        current_mask |= (1 << r1_idx)
                        current_mask |= (1 << r2_idx)
        
        return best_mask, best_size
    
    # Run multiple strategies
    best_mask = 0
    best_size = 0
    
    # Strategy 1: Greedy with constraint-count ordering (fewest constraints first)
    order1 = sorted(range(num_triples), key=lambda i: constraint_count[i])
    mask1, size1 = greedy_order(order1)
    if size1 > best_size:
        best_mask = mask1
        best_size = size1
    
    # Strategy 2: Greedy with reverse constraint-count ordering
    order2 = sorted(range(num_triples), key=lambda i: -constraint_count[i])
    mask2, size2 = greedy_order(order2)
    if size2 > best_size:
        best_mask = mask2
        best_size = size2
    
    # Strategy 3: Deterministic sorted order
    order3 = list(range(num_triples))
    mask3, size3 = greedy_order(order3)
    if size3 > best_size:
        best_mask = mask3
        best_size = size3
    
    # Local search on best
    if time.time() < deadline:
        ls_mask, ls_size = local_search_improved(best_mask, max_steps=15000)
        if ls_size > best_size:
            best_mask = ls_mask
            best_size = ls_size
    
    # Multiple random restarts
    random.seed(42)
    
    while time.time() < deadline - 10:
        # Random greedy
        random_order = list(range(num_triples))
        random.shuffle(random_order)
        g_mask, g_size = greedy_order(random_order)
        
        # Local search
        ls_mask, ls_size = local_search_improved(g_mask, max_steps=10000)
        
        if ls_size > best_size:
            best_mask = ls_mask
            best_size = ls_size
    
    # Final polish
    if time.time() < deadline:
        final_mask, final_size = local_search_improved(best_mask, max_steps=15000)
        if final_size > best_size:
            best_mask = final_mask
            best_size = final_size
    
    # Convert to list of lists
    result = mask_to_set(best_mask)
    return [list(t) for t in sorted(result)]
