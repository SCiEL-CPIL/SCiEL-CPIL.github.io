import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58

    # Precompute all 4-subsets as bitmasks
    all_sets = []
    for c in combinations(range(16), 4):
        mask = 0
        for i in c:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_total = len(all_sets)  # 1820
    masks = np.array(all_sets, dtype=np.uint16)
    
    # Precompute intersection matrix
    inter = (masks[:, None] & masks[None, :]).astype(np.uint8)
    inter_sizes = np.zeros((n_total, n_total), dtype=np.uint8)
    for i in range(n_total):
        for j in range(n_total):
            inter_sizes[i, j] = bin(inter[i, j]).count('1')
    
    # For each pair (i,j), compute their intersection mask
    # A third set k conflicts with (i,j) if inter_size(k,i) == inter_size(k,j) == inter_size(i,j)
    # Precompute for each pair (i,j) the list of k that would form a sunflower
    # This is expensive, so let's use a different approach
    
    # Better: for a candidate k, it conflicts with existing family if there exist i,j in family
    # such that inter_size(k,i) == inter_size(k,j) == inter_size(i,j)
    
    # For speed, maintain for the family:
    # - For each intersection size s (0,1,2,3), the sets in family grouped by their inter_size with new candidate
    # Actually, let's think differently.
    
    # For candidate k and family F, k is invalid if there exist i,j in F with:
    #   inter_size(k,i) == inter_size(k,j) == inter_size(i,j)
    
    # Let's precompute for each k, and for each possible intersection size s,
    # which sets in the family have inter_size(k,.) == s
    # Then check if any pair among those has inter_size == s
    
    # Maintain family as a list, and for each family member, precompute its inter_size with all
    # Actually, let's just be efficient in the check.
    
    def can_add(family, k):
        """Check if k can be added to family without creating a sunflower."""
        if len(family) < 2:
            return True
        # Group family members by their inter_size with k
        groups = {}
        for i in family:
            s = inter_sizes[k, i]
            if s not in groups:
                groups[s] = []
            groups[s].append(i)
        # For each group with intersection size s, check if any pair has inter_size == s
        for s, members in groups.items():
            if len(members) < 2:
                continue
            for a in range(len(members)):
                for b in range(a+1, len(members)):
                    if inter_sizes[members[a], members[b]] == s:
                        return False
        return True
    
    rng = random.Random(12345)
    best_family = []
    best_size = 0
    
    # Multiple greedy passes with different orderings
    for trial in range(30):
        if time.time() > deadline - 20:
            break
        order = list(range(n_total))
        if trial > 0:
            rng.shuffle(order)
        family = []
        for idx in order:
            if can_add(family, idx):
                family.append(idx)
        if len(family) > best_size:
            best_size = len(family)
            best_family = list(family)
    
    # Try to add remaining sets
    family = list(best_family)
    fam_set = set(family)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for idx in range(n_total):
            if idx in fam_set:
                continue
            if can_add(family, idx):
                family.append(idx)
                fam_set.add(idx)
                improved = True
                if len(family) > best_size:
                    best_size = len(family)
                    best_family = list(family)
                if len(family) >= 250:
                    break
        if len(family) >= 250:
            break
    
    # Local search: remove one, try to add two
    current = list(family)
    cur_set = set(current)
    
    for it in range(50000):
        if time.time() > deadline - 3:
            break
        if not current:
            break
        
        # Remove a random element
        ri = rng.randrange(len(current))
        removed = current[ri]
        current.pop(ri)
        cur_set.discard(removed)
        
        # Try to add two elements
        added_any = False
        for _ in range(2):
            # Try random candidates
            for _ in range(50):
                cand = rng.randrange(n_total)
                if cand in cur_set:
                    continue
                if can_add(current, cand):
                    current.append(cand)
                    cur_set.add(cand)
                    added_any = True
                    break
        
        if added_any and len(current) > best_size:
            best_size = len(current)
            best_family = list(current)
        else:
            # Revert
            current.insert(ri, removed)
            cur_set.add(removed)
        
        if best_size >= 250:
            break
    
    # Convert to list of lists
    result = []
    for idx in best_family:
        mask = all_sets[idx]
        elems = [i+1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elems))
    
    return result
