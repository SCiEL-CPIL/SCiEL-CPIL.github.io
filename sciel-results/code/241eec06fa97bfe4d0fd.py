"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill
    
    # Generate all 4-subsets of [1,16] as bitmasks
    all_masks = []
    for row in combinations(range(1, 17), 4):
        mask = 0
        for x in row:
            mask |= (1 << x)
        all_masks.append(mask)
    
    n = len(all_masks)  # C(16,4) = 1820
    
    # Precompute pairwise intersections as bitmasks
    # inter[i][j] = all_masks[i] & all_masks[j]
    inter = [[0] * n for _ in range(n)]
    for i in range(n):
        mi = all_masks[i]
        for j in range(i + 1, n):
            ij = mi & all_masks[j]
            inter[i][j] = ij
            inter[j][i] = ij
    
    # Precompute sunflower triples: for each pair (i,j), which k form a sunflower with them
    # A triple (i,j,k) is a sunflower if inter[i][j] == inter[i][k] == inter[j][k]
    # For each pair (i,j), the intersection is fixed = inter[i][j]
    # So k must satisfy: inter[i][k] == inter[i][j] AND inter[j][k] == inter[i][j]
    
    # conflict[i] = set of indices j that conflict with i (i.e., there exists k such that (i,j,k) is sunflower)
    # Actually, we need: for a given set S of chosen indices, adding index x is invalid if there exist i,j in S with (x,i,j) forming a sunflower.
    # So for each x, define bad_pairs[x] = set of (i,j) pairs (i<j, i!=x, j!=x) such that (x,i,j) is a sunflower triple.
    
    # More efficiently: for each pair (i,j), compute the set of all k that form a sunflower with i,j.
    # Then when checking if we can add x to chosen set C, we check: for each pair (i,j) in C, is x in sunflower_with[i][j]?
    
    # Let's precompute: for each unordered pair (i,j), the list of k (k!=i,j) such that (i,j,k) is a sunflower.
    # But that's O(n^2 * n) = O(n^3) which is ~6 billion... too much.
    
    # Alternative: for each pair (i,j), the intersection val = inter[i][j].
    # k is a sunflower partner iff inter[i][k] == val AND inter[j][k] == val.
    # We can precompute for each mask value, which indices have that intersection with a given index.
    
    # Better approach: For each index i, precompute a dictionary: inter_val -> list of j where inter[i][j] == inter_val
    # Then for pair (i,j) with val = inter[i][j], the sunflower partners k are those in inter_dict[i][val] ∩ inter_dict[j][val] minus {i,j}.
    
    # Let's use a different strategy for the greedy: 
    # Maintain for each candidate x, a count of how many "conflict pairs" it has with the current chosen set.
    # A conflict pair for x is a pair (i,j) in chosen such that (x,i,j) is a sunflower.
    
    # For efficiency, when we add a new element c to chosen, we need to update the conflict counts for all remaining candidates.
    # For each remaining candidate x, the new conflicts added are: pairs (i, c) where i in chosen_before_c such that (x, i, c) is sunflower.
    # (x, i, c) is sunflower iff inter[x][i] == inter[x][c] == inter[i][c].
    
    # Let me just do a fast check using bitmasks.
    
    def is_sunflower(i, j, k):
        """Check if indices i, j, k form a sunflower triple."""
        return inter[i][j] == inter[i][k] == inter[j][k]
    
    def can_add(chosen_set, x, chosen_list):
        """Check if x can be added to the chosen set."""
        for idx in range(len(chosen_list)):
            i = chosen_list[idx]
            for idx2 in range(idx + 1, len(chosen_list)):
                j = chosen_list[idx2]
                if is_sunflower(x, i, j):
                    return False
        return True
    
    # Greedy with ordering by minimum conflicts (least constraining first)
    def greedy_with_order(order, deadline):
        chosen = []
        chosen_set = set()
        for x in order:
            if time.time() > deadline:
                break
            if x in chosen_set:
                continue
            if can_add(chosen_set, x, chosen):
                chosen.append(x)
                chosen_set.add(x)
        return chosen
    
    # Strategy 1: Greedy with natural order
    best = greedy_with_order(list(range(n)), deadline)
    best_size = len(best)
    
    if time.time() > deadline:
        return [sorted([(1 if (m >> i) & 1 else 0) for i in range(1, 17)]) for m in [all_masks[x] for x in best]]
    
    # Strategy 2: Greedy with random order (multiple seeds)
    for seed in [42, 123, 456, 789, 101]:
        if time.time() > deadline:
            break
        random.seed(seed)
        order = list(range(n))
        random.shuffle(order)
        fam = greedy_with_order(order, deadline)
        if len(fam) > best_size:
            best = fam
            best_size = len(fam)
    
    if time.time() > deadline:
        return [sorted([(1 if (m >> i) & 1 else 0) for i in range(1, 17)]) for m in [all_masks[x] for x in best]]
    
    # Strategy 3: Greedy with "least conflicts first" ordering
    # For each element, count how many sunflower triples it participates in
    # Elements in fewer sunflower triples are "safer" to pick
    triple_count = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            val = inter[i][j]
            # Find all k where inter[i][k] == val and inter[j][k] == val
            # This is expensive; approximate by sampling or just use random orders
    
    # Strategy 4: Local search / improvement
    # Start from best, try to swap elements to improve
    current = list(best)
    current_set = set(best)
    
    # Precompute for each pair in current, the set of elements that conflict with that pair
    # For local search: try removing one element and adding another
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each pair of (remove r, add a) where r in current, a not in current
        # Check if removing r and adding a keeps the set valid
        for r_idx in range(len(current)):
            if time.time() > deadline:
                break
            r = current[r_idx]
            # Try adding each non-current element
            for a in range(n):
                if a in current_set:
                    continue
                # Check: is (current \ {r}) ∪ {a} valid?
                # i.e., no sunflower triple in the new set
                # A sunflower triple in new set either:
                # 1. Was already in current \ {r} (impossible since current is valid)
                # 2. Includes a and two elements from current \ {r}
                # So check: for all pairs (i,j) in current \ {r}, is (a,i,j) a sunflower?
                valid = True
                for i_idx in range(len(current)):
                    if i_idx == r_idx:
                        continue
                    i = current[i_idx]
                    for j_idx in range(i_idx + 1, len(current)):
                        if j_idx == r_idx:
                            continue
                        j = current[j_idx]
                        if is_sunflower(a, i, j):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Perform the swap
                    current[r_idx] = a
                    current_set.discard(r)
                    current_set.add(a)
                    improved = True
                    break
            if improved:
                break
    
    best_size = len(current)
    best = current
    
    # Convert back to sorted lists of elements
    result = []
    for x in best:
        m = all_masks[x]
        elems = []
        for i in range(1, 17):
            if (m >> i) & 1:
                elems.append(i)
        result.append(elems)
    
    return result
# EVOLVE-BLOCK-END
