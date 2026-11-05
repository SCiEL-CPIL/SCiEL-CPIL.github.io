"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # 1. Generate all 4-subsets of [1, 16] as bitmasks
    all_masks = []
    mask_to_idx = {}
    idx_to_mask = []
    
    # Precompute masks for 4-combinations
    for c in combinations(range(1, 17), 4):
        m = 0
        for x in c:
            m |= (1 << (x - 1))
        all_masks.append(m)
        
    n_total = len(all_masks)
    
    # Map mask to index for quick lookup
    for i, m in enumerate(all_masks):
        mask_to_idx[m] = i
        
    # 2. Precompute pairwise intersections
    # inter[i][j] = all_masks[i] & all_masks[j]
    # To save memory and time, we can store this as a list of lists or a flat array.
    # 1820 * 1820 is ~3.3M entries.
    # We will store it as a list of lists of ints.
    
    inter = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        mi = all_masks[i]
        for j in range(i, n_total):
            mij = mi & all_masks[j]
            inter[i][j] = mij
            inter[j][i] = mij
            
    # 3. Precompute neighbors for faster validity check
    # neighbors[i][V] = list of j such that inter[i][j] == V
    # V is a bitmask (0 to 2^16 - 1).
    # We only care about V that are subsets of all_masks[i].
    # Actually, V can be any intersection.
    # Number of distinct intersections for a fixed i is small.
    # Let's use a dictionary: neighbors[i] = {V: [j1, j2, ...]}
    
    neighbors = [{} for _ in range(n_total)]
    for i in range(n_total):
        for j in range(n_total):
            if i == j:
                continue
            V = inter[i][j]
            if V not in neighbors[i]:
                neighbors[i][V] = []
            neighbors[i][V].append(j)
            
    def is_valid_addition(cand_idx, family_set):
        """
        Check if cand_idx can be added to family_set (set of indices) without creating a sunflower.
        Uses precomputed neighbors.
        """
        for i in family_set:
            # Intersection of i and cand
            V = inter[i][cand_idx]
            # We need to check if there is j in family_set such that:
            # 1. inter[i][j] == V
            # 2. inter[j][cand_idx] == V
            
            # Get all j that intersect i in V
            # If V is not in neighbors[i], there are no such j (other than i itself, but j!=i)
            if V in neighbors[i]:
                for j in neighbors[i][V]:
                    if j in family_set:
                        # Check if j and cand also intersect in V
                        if inter[j][cand_idx] == V:
                            return False
        return True

    best_family = []
    best_size = 0
    rng = random.Random(42)
    
    # Strategy: Multiple greedy passes, then local search
    
    def greedy_fill(family_set, order):
        """
        Try to add sets from 'order' to family_set.
        Returns the updated family_set.
        """
        current_family = set(family_set)
        for idx in order:
            if idx in current_family:
                continue
            if is_valid_addition(idx, current_family):
                current_family.add(idx)
        return current_family

    # Run multiple greedy trials
    num_greedy_trials = 30
    for trial in range(num_greedy_trials):
        if time.time() > deadline - 20:
            break
            
        if trial == 0:
            order = list(range(n_total))
        else:
            order = list(range(n_total))
            rng.shuffle(order)
            
        family_set = greedy_fill(set(), order)
        
        if len(family_set) > best_size:
            best_size = len(family_set)
            best_family = family_set.copy()
            
        if best_size >= 250:
            break
            
    # Local Search on the best family found
    if time.time() < deadline - 5 and best_size > 0:
        family_set = best_family.copy()
        family_size = len(family_set)
        
        # List of indices NOT in family for quick random selection
        # We can just pick random indices and check if they are in family
        
        while time.time() < deadline - 5:
            improved = False
            
            # 1. Try to Add a random set
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                cand_idx = rng.randint(0, n_total - 1)
                if cand_idx not in family_set:
                    if is_valid_addition(cand_idx, family_set):
                        family_set.add(cand_idx)
                        family_size += 1
                        improved = True
                        if family_size > best_size:
                            best_size = family_size
                            best_family = family_set.copy()
                        
            # 2. Try Swap: Remove one, Add one
            if not improved:
                # Pick a random set to remove
                if family_set:
                    remove_idx = rng.choice(list(family_set))
                    # Pick a random set to add
                    cand_idx = rng.randint(0, n_total - 1)
                    if cand_idx not in family_set:
                        # Check validity of new family
                        # Temporarily remove
                        new_family = family_set - {remove_idx}
                        if is_valid_addition(cand_idx, new_family):
                            family_set = new_family
                            family_set.add(cand_idx)
                            # family_size remains same
                            # Try to add more after swap
                            # (Optional: run a quick greedy pass)
                            pass
                            
            # 3. Try 2-Swap: Remove two, Add two
            if not improved and len(family_set) >= 2:
                # Pick two random sets to remove
                list_fam = list(family_set)
                r1, r2 = rng.sample(list_fam, 2)
                # Pick two random sets to add
                c1 = rng.randint(0, n_total - 1)
                c2 = rng.randint(0, n_total - 1)
                if c1 not in family_set and c2 not in family_set and c1 != c2:
                    new_family = family_set - {r1, r2}
                    # Check if c1 can be added
                    if is_valid_addition(c1, new_family):
                        # Check if c2 can be added to (new_family + c1)
                        new_family_with_c1 = new_family | {c1}
                        if is_valid_addition(c2, new_family_with_c1):
                            family_set = new_family_with_c1
                            family_set.add(c2)
                            
    # Convert best_family (set of indices) to list of sorted lists
    result = []
    for idx in best_family:
        m = all_masks[idx]
        # Convert mask to sorted list of elements
        s = []
        for i in range(1, 17):
            if m & (1 << (i - 1)):
                s.append(i)
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
