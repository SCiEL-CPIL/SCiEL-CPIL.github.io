# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s kill
    
    n = 16
    k = 4
    
    # Precompute all 4-subsets as bitmasks and sorted tuples
    all_sets = list(combinations(range(1, n + 1), k))
    # Convert to bitmasks for fast intersection
    set_to_mask = {}
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        set_to_mask[s] = mask
    
    total = len(all_sets)
    
    # Precompute intersection sizes between all pairs
    # This is C(1820,2) ~ 1.65M pairs, feasible
    # Actually let's be smarter: for each set, store its mask
    masks = [set_to_mask[s] for s in all_sets]
    
    def popcount(x):
        return bin(x).count('1')
    
    def can_add(family_masks, family_indices, new_mask, new_idx):
        """Check if adding new_mask creates a sunflower with any two in family."""
        for i in range(len(family_masks)):
            inter_i = popcount(family_masks[i] & new_mask)
            for j in range(i + 1, len(family_masks)):
                inter_j = popcount(family_masks[j] & new_mask)
                if inter_i != inter_j:
                    continue
                # Check if inter_i == inter_j == popcount(family_masks[i] & family_masks[j])
                inter_ij = popcount(family_masks[i] & family_masks[j])
                if inter_i == inter_ij:
                    return False
        return True
    
    def greedy_random(order):
        """Greedy: iterate through sets in given order, add if valid."""
        family_masks = []
        family_indices = []
        for idx in order:
            if time.time() > deadline:
                break
            m = masks[idx]
            if can_add(family_masks, family_indices, m, idx):
                family_masks.append(m)
                family_indices.append(idx)
        return family_indices
    
    best_size = 0
    best_family = []
    
    # Try multiple random orders
    rng = random.Random(42)
    num_tries = 0
    while time.time() < deadline:
        order = list(range(total))
        rng.shuffle(order)
        family = greedy_random(order)
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            # Try to improve with local search
            if time.time() < deadline - 5.0:
                # Local search: try swapping out a set for an unused one
                used_set = set(best_family)
                improved = True
                while improved and time.time() < deadline - 2.0:
                    improved = False
                    for i in range(len(best_family)):
                        if time.time() > deadline - 2.0:
                            break
                        # Try removing best_family[i] and adding an unused set
                        candidate_family = best_family[:i] + best_family[i+1:]
                        candidate_used = set(candidate_family)
                        for j in range(total):
                            if j in candidate_used:
                                continue
                            m = masks[j]
                            # Check if adding m to candidate_family is valid
                            valid = True
                            cm = [masks[x] for x in candidate_family]
                            for a in range(len(cm)):
                                ia = popcount(cm[a] & m)
                                for b in range(a + 1, len(cm)):
                                    ib = popcount(cm[b] & m)
                                    if ia != ib:
                                        continue
                                    iab = popcount(cm[a] & cm[b])
                                    if ia == iab:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_family = candidate_family + [j]
                                # This is same size, so try to see if we can do better
                                # Actually same size, so no improvement in cardinality
                                # Let's break and try different approach
                                break
                        break  # Only try one swap per iteration for speed
                    # If no improvement found in this pass, break
                    if not improved:
                        break
        
        num_tries += 1
        if num_tries >= 200:
            break
    
    # Convert best_family indices back to sorted tuples
    result = [sorted(all_sets[i]) for i in best_family]
    return result
# EVOLVE-BLOCK-END
