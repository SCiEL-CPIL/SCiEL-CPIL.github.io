import random
import math
import time

def construct():
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples
    all_possible_triples = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                all_possible_triples.append([i, j, k])
                
    # Helper to check if a triple is valid (linear) with respect to a set of triples
    def is_valid(triple, existing):
        t_set = set(triple)
        for e in existing:
            e_set = set(e)
            if len(t_set & e_set) > 1:
                return False
        return True

    # Helper to calculate independence number restricted to a specific mask
    # triples: list of triples (lists of ints)
    # mask: bitmask of allowed points
    def get_independence_number_restricted(triples, mask):
        # Get points in mask
        pts_in_mask = [i for i in range(n) if (mask >> i) & 1]
        k = len(pts_in_mask)
        
        # Map points in mask to 0..k-1 for faster subset iteration?
        # Or just iterate 2^k subsets and check.
        # k is at most 16, but usually ~8 in 2-coloring.
        # 2^16 is 65536, which is fine.
        
        # Precompute triple masks restricted to pts_in_mask?
        # Actually, we just need to check if a subset S of pts_in_mask contains any triple.
        # A triple t is "in" S if all elements of t are in S.
        
        # Convert triples to sets for fast intersection?
        # Or use bitmasks.
        t_masks = []
        for t in triples:
            tm = 0
            for p in t:
                tm |= (1 << p)
            t_masks.append(tm)
            
        max_size = 0
        # Iterate all subsets of pts_in_mask
        # We can iterate all masks from 0 to 2^n, but only consider those subset of mask?
        # Faster: iterate submasks of mask.
        # submask = mask
        # while submask:
        #   ...
        #   submask = (submask - 1) & mask
        
        submask = mask
        while submask:
            # Check if submask is independent
            is_ind = True
            for tm in t_masks:
                if (submask & tm) == tm:
                    is_ind = False
                    break
            if is_ind:
                size = bin(submask).count("1")
                if size > max_size:
                    max_size = size
            submask = (submask - 1) & mask
        
        # Check empty set (size 0) just in case, though max_size starts at 0
        return max_size

    def get_true_independence_number(triples):
        return get_independence_number_restricted(triples, (1 << n) - 1)

    # Initial State
    # Random greedy initialization
    current_triples = []
    random.shuffle(all_possible_triples)
    for t in all_possible_triples:
        if is_valid(t, current_triples):
            current_triples.append(t)
            if len(current_triples) >= 37:
                break

    # Initial Coloring
    # Random 2-coloring
    current_coloring = [random.randint(0, 1) for _ in range(n)]
    
    def get_masks(coloring):
        mask0 = 0
        mask1 = 0
        for i in range(n):
            if coloring[i] == 0:
                mask0 |= (1 << i)
            else:
                mask1 |= (1 << i)
        return mask0, mask1

    def calculate_cost(triples, coloring):
        mask0, mask1 = get_masks(coloring)
        # Filter triples contained in mask0 and mask1
        t_in_0 = [t for t in triples if ((mask0 >> t[0]) & 1) and ((mask0 >> t[1]) & 1) and ((mask0 >> t[2]) & 1)]
        t_in_1 = [t for t in triples if ((mask1 >> t[0]) & 1) and ((mask1 >> t[1]) & 1) and ((mask1 >> t[2]) & 1)]
        
        alpha0 = get_independence_number_restricted(t_in_0, mask0)
        alpha1 = get_independence_number_restricted(t_in_1, mask1)
        
        return max(alpha0, alpha1)

    best_triples = current_triples[:]
    best_coloring = current_coloring[:]
    best_cost = calculate_cost(best_triples, best_coloring)
    
    start_time = time.time()
    timeout = 110
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() - start_time < timeout:
        move_type = random.choice(['flip_color', 'add_triple', 'remove_triple', 'swap_triple'])
        
        new_triples = current_triples[:]
        new_coloring = current_coloring[:]
        
        if move_type == 'flip_color':
            idx = random.randint(0, n - 1)
            new_coloring[idx] = 1 - new_coloring[idx]
            
        elif move_type == 'add_triple':
            # Try to add a random valid triple
            added = False
            for _ in range(20):
                cand = random.choice(all_possible_triples)
                if cand not in new_triples:
                    if is_valid(cand, new_triples):
                        new_triples.append(cand)
                        added = True
                        break
            if not added:
                continue
                
        elif move_type == 'remove_triple':
            if len(new_triples) > 0:
                idx = random.randint(0, len(new_triples) - 1)
                new_triples.pop(idx)
            else:
                continue
                
        elif move_type == 'swap_triple':
            if len(new_triples) == 0:
                continue
            idx = random.randint(0, len(new_triples) - 1)
            removed = new_triples.pop(idx)
            added = False
            for _ in range(20):
                cand = random.choice(all_possible_triples)
                if cand != removed and cand not in new_triples:
                    if is_valid(cand, new_triples):
                        new_triples.append(cand)
                        added = True
                        break
            if not added:
                new_triples.insert(idx, removed)
                continue

        new_cost = calculate_cost(new_triples, new_coloring)
        
        if new_cost < best_cost:
            best_cost = new_cost
            best_triples = new_triples[:]
            best_coloring = new_coloring[:]
            
        if new_cost < calculate_cost(current_triples, current_coloring):
            current_triples = new_triples
            current_coloring = new_coloring
        else:
            delta = new_cost - calculate_cost(current_triples, current_coloring)
            if delta == 0:
                if random.random() < 0.5:
                    current_triples = new_triples
                    current_coloring = new_coloring
            else:
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_triples = new_triples
                    current_coloring = new_coloring
        
        T *= cooling

    # Return the best triples found
    # The evaluator will check the true independence number
    return best_triples
