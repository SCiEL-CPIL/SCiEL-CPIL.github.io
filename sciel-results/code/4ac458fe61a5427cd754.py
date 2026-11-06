import random
from itertools import combinations

def construct():
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    
    # Precompute pairwise intersections
    # inter[i][j] = all_sets[i] & all_sets[j]
    inter = [[0]*n_sets for _ in range(n_sets)]
    for i in range(n_sets):
        mi = all_sets[i]
        for j in range(i+1, n_sets):
            mj = all_sets[j]
            ij = mi & mj
            inter[i][j] = ij
            inter[j][i] = ij
        inter[i][i] = mi # Self intersection is the set itself, but we only care about distinct sets in the sunflower check, so i!=j. But just in case.

    best_family = []
    best_size = 0
    
    def is_valid_add(c_idx, F_indices):
        # F_indices is a list of indices into all_sets
        c_mask = all_sets[c_idx]
        n_f = len(F_indices)
        for i in range(n_f):
            a_idx = F_indices[i]
            a_mask = all_sets[a_idx]
            ac = a_mask & c_mask
            # Check if there is b in F such that a^b == ac and b^c == ac
            for j in range(i+1, n_f):
                b_idx = F_indices[j]
                if inter[a_idx][b_idx] == ac:
                    b_mask = all_sets[b_idx]
                    if (b_mask & c_mask) == ac:
                        return False
        return True

    # To speed up is_valid_add, we can maintain F as a list of indices.
    # But the current code uses masks. I will switch to indices for faster lookup in inter table.
    
    for seed in range(50):
        random.seed(seed)
        order = list(range(n_sets))
        random.shuffle(order)
        
        family_indices = []
        family_set = set()
        
        for idx in order:
            if is_valid_add(idx, family_indices):
                family_indices.append(idx)
                family_set.add(idx)
        
        current_size = len(family_indices)
        if current_size > best_size:
            best_size = current_size
            best_family = family_indices.copy()
        
        # Local Search
        for _ in range(200):
            # Try to add
            added = False
            # Iterate all candidates not in family
            for c_idx in range(n_sets):
                if c_idx not in family_set:
                    if is_valid_add(c_idx, family_indices):
                        family_indices.append(c_idx)
                        family_set.add(c_idx)
                        current_size += 1
                        added = True
                        break
            
            if added:
                if current_size > best_size:
                    best_size = current_size
                    best_family = family_indices.copy()
                continue
            
            # Try 20 random swaps
            swap_done = False
            for _ in range(20):
                if not family_indices:
                    break
                idx_out_pos = random.randint(0, len(family_indices)-1)
                idx_out = family_indices[idx_out_pos]
                
                # Pick random candidate from complement
                # To avoid O(N) scan, pick random index and check
                c_in = random.randint(0, n_sets-1)
                if c_in in family_set:
                    continue
                
                # Create new family without idx_out
                # Instead of list copy, we can just check validity against F \ {idx_out}
                # We need a helper that skips one index
                def is_valid_add_skip(c_idx, F_indices, skip_pos):
                    c_mask = all_sets[c_idx]
                    n_f = len(F_indices)
                    for i in range(n_f):
                        if i == skip_pos: continue
                        a_idx = F_indices[i]
                        a_mask = all_sets[a_idx]
                        ac = a_mask & c_mask
                        for j in range(i+1, n_f):
                            if j == skip_pos: continue
                            b_idx = F_indices[j]
                            if inter[a_idx][b_idx] == ac:
                                b_mask = all_sets[b_idx]
                                if (b_mask & c_mask) == ac:
                                    return False
                    return True

                if is_valid_add_skip(c_in, family_indices, idx_out_pos):
                    family_indices[idx_out_pos] = c_in
                    family_set.remove(idx_out)
                    family_set.add(c_in)
                    swap_done = True
                    break
            
            if not swap_done and random.random() < 0.1:
                # Perturb: remove one random element
                if family_indices:
                    idx_out_pos = random.randint(0, len(family_indices)-1)
                    idx_out = family_indices[idx_out_pos]
                    family_indices.pop(idx_out_pos)
                    family_set.remove(idx_out)
                    current_size -= 1

    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
