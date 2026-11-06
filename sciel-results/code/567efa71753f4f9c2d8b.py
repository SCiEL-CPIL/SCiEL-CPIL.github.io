# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    n = 16
    k = 4
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s kill

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_masks = len(all_masks)  # 1820

    # Precompute pairwise intersections
    inter = [[0] * num_masks for _ in range(num_masks)]
    for i in range(num_masks):
        mi = all_masks[i]
        for j in range(i + 1, num_masks):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def check_conflict(candidate_idx, family_indices):
        """Check if adding candidate_idx to family_indices creates a sunflower."""
        cm = all_masks[candidate_idx]
        m = len(family_indices)
        for ai in range(m):
            a_idx = family_indices[ai]
            a_inter_c = inter[a_idx][candidate_idx]
            for bi in range(ai + 1, m):
                b_idx = family_indices[bi]
                if inter[a_idx][b_idx] == a_inter_c and inter[b_idx][candidate_idx] == a_inter_c:
                    return True
        return False

    def greedy(order):
        """Run greedy in given order, return list of indices in family."""
        family = []
        for idx in order:
            if not check_conflict(idx, family):
                family.append(idx)
        return family

    # 1. Deterministic greedy (lexicographic order)
    best_family = greedy(list(range(num_masks)))
    best_size = len(best_family)

    if time.time() - start_time > time_limit:
        return [[i + 1 for i in range(n) if all_masks[best_family[j]] & (1 << i)] 
                for j in range(len(best_family))]

    # 2. Randomized greedy restarts
    random.seed(42)
    for restart in range(50):
        if time.time() - start_time > time_limit - 10:
            break
        order = list(range(num_masks))
        random.shuffle(order)
        family = greedy(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family

    # 3. Hypergraph local search: treat as 3-uniform hypergraph of sunflower triples
    # Each hyperedge is a triple of sets that form a sunflower
    # We want to maximize the number of selected vertices with no hyperedge fully contained
    
    # Build adjacency: for each pair (i,j), count how many k make (i,j,k) a sunflower
    # Actually, let's use a different approach: 
    # For each vertex, compute its "conflict count" with the current family
    # Then do local search by flipping vertices
    
    in_family = set(best_family)
    family_list = list(best_family)
    
    # Precompute for each vertex, which other vertices it forms sunflowers with
    # conflict_count[i] = number of vertices j such that i and j are in a sunflower triple
    # But this is expensive. Instead, let's do a simpler local search.
    
    # Local search: try to swap out a family member for a non-family member
    improved = True
    while improved and time.time() - start_time < time_limit - 5:
        improved = False
        family_set = set(best_family)
        
        # For each candidate not in family, try to add it by removing one family member
        candidates = [c for c in range(num_masks) if c not in family_set]
        random.shuffle(candidates)
        
        for c in candidates:
            if time.time() - start_time > time_limit - 5:
                break
            # Try removing each family member
            for f in list(best_family):
                # Check if (family - {f}) + {c} is valid
                new_family = [x for x in best_family if x != f]
                if not check_conflict(c, new_family):
                    new_family.append(c)
                    # Try to greedily add more
                    added = 0
                    for c2 in range(num_masks):
                        if c2 in set(new_family):
                            continue
                        if not check_conflict(c2, new_family):
                            new_family.append(c2)
                            added += 1
                    if len(new_family) > best_size:
                        best_family = new_family
                        best_size = len(best_family)
                        improved = True
                        break
            if improved:
                break
    
    # Convert to result
    result = []
    for idx in best_family:
        mask = all_masks[idx]
        subset = [i + 1 for i in range(n) if mask & (1 << i)]
        result.append(subset)
    
    return result
# EVOLVE-BLOCK-END
