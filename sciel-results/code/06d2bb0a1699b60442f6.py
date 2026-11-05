"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)

    m = len(all_masks)
    # m = C(16,4) = 1820

    # Precompute pairwise intersections and "conflict sets"
    # conflict_pairs[i][j] is not stored fully, but we compute forbidden sets on the fly.
    # However, we can precompute the intersection bitmask for speed.
    inter = [[0] * m for _ in range(m)]
    for i in range(m):
        mi = all_masks[i]
        for j in range(i + 1, m):
            v = mi & all_masks[j]
            inter[i][j] = v
            inter[j][i] = v

    def get_forbidden_indices(a_idx, b_idx):
        """
        Returns a list of indices c such that a, b, c form a sunflower.
        """
        ma = all_masks[a_idx]
        mb = all_masks[b_idx]
        ab = ma & mb
        # C must contain ab
        # C must be disjoint from (ma & ~mb) and (mb & ~ma)
        # Let diff = (ma & ~mb) | (mb & ~ma)
        # C must be subset of ~diff
        # C must have size 4
        # C must contain ab
        
        diff = (ma & ~mb) | (mb & ~ma)
        allowed_mask = ~diff & ((1 << n) - 1)
        
        # C must be a 4-subset of allowed_mask that contains ab
        # ab is a subset of allowed_mask?
        # ab = ma & mb.
        # ma & ~mb and mb & ~ma are disjoint from ab.
        # So yes, ab is subset of allowed_mask.
        
        forbidden = []
        # Iterate through all masks c
        # Optimization: Only check masks that are subsets of allowed_mask and supersets of ab
        # Since m is small (1820), we can just iterate all and check.
        for c_idx in range(m):
            mc = all_masks[c_idx]
            if c_idx == a_idx or c_idx == b_idx:
                continue
            # Check if ab is subset of mc
            if (ab & ~mc) != 0:
                continue
            # Check if mc is subset of allowed_mask
            # This is equivalent to (mc & diff) == 0
            if (mc & diff) != 0:
                continue
            forbidden.append(c_idx)
        return forbidden

    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result

    best_size = 0
    best_family_idx = []

    def greedy_construct(order):
        family = []
        # conf_count[c] = number of pairs (a, b) in family such that c is forbidden by (a, b)
        conf_count = [0] * m
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if conf_count[idx] == 0:
                family.append(idx)
                # Update conf_count for all pairs involving idx
                for a_idx in family[:-1]:
                    forbids = get_forbidden_indices(a_idx, idx)
                    for c_idx in forbids:
                        conf_count[c_idx] += 1
        
        return family

    # Strategy 1: Greedy in natural order
    family = greedy_construct(list(range(m)))
    if len(family) > best_size:
        best_size = len(family)
        best_family_idx = family[:]

    # Strategy 2: Multiple random shuffles
    random.seed(12345)
    max_attempts = 500 # More attempts due to faster greedy
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break

        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)

        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Strategy 3: Local Search (Swap)
    # We need a way to do local search with conf_count.
    # It's easier to just re-run greedy on a modified order or do explicit swaps.
    # Explicit swaps with conf_count update:
    # 1. Start with best_family_idx
    # 2. Compute conf_count for this family
    # 3. Try to remove one, update counts, find valid add, update counts.
    
    def compute_conf_counts(family):
        conf_count = [0] * m
        for i in range(len(family)):
            for j in range(i+1, len(family)):
                forbids = get_forbidden_indices(family[i], family[j])
                for c_idx in forbids:
                    conf_count[c_idx] += 1
        return conf_count

    def local_search(family):
        if time.time() > deadline - 2:
            return family
        
        current_family = family[:]
        conf_count = compute_conf_counts(current_family)
        in_family = set(current_family)
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            # Try to add a set
            for c_idx in range(m):
                if c_idx not in in_family and conf_count[c_idx] == 0:
                    current_family.append(c_idx)
                    in_family.add(c_idx)
                    # Update counts
                    for a_idx in current_family[:-1]:
                        forbids = get_forbidden_indices(a_idx, c_idx)
                        for f_idx in forbids:
                            conf_count[f_idx] += 1
                    improved = True
                    break
            
            if not improved:
                # Try swaps
                for i in range(len(current_family)):
                    if time.time() > deadline - 1:
                        break
                    remove_idx = current_family[i]
                    # Temporarily remove
                    # We need to decrement counts for pairs involving remove_idx
                    # This is complex to do in-place without a full recompute or careful tracking.
                    # Given time constraints, a simpler approach is to just try adding.
                    # Or, we can do a "kick" heuristic:
                    # Pick a random set to remove, recompute conf_count, try to add.
                    # This is expensive.
                    pass
                    
        return current_family

    # Apply local search to the best family found
    if time.time() < deadline - 5:
        improved_family = local_search(best_family_idx)
        if len(improved_family) > best_size:
            best_size = len(improved_family)
            best_family_idx = improved_family[:]

    # More random restarts with local search
    for seed_val in [7, 42, 99, 2024, 555, 777, 1234, 4321, 8888, 13579, 2468, 1357, 9999, 1111, 2222, 3333, 4444, 5555, 6666, 7777]:
        if time.time() > deadline - 3:
            break
        random.seed(seed_val)
        order = list(range(m))
        random.shuffle(order)
        family = greedy_construct(order)
        
        if time.time() < deadline - 2:
            family = local_search(family)
            
        if len(family) > best_size:
            best_size = len(family)
            best_family_idx = family[:]

    # Cap at 250
    if len(best_family_idx) > 250:
        best_family_idx = best_family_idx[:250]

    # Convert back to sorted lists of 1..16
    result = [mask_to_list(all_masks[idx]) for idx in best_family_idx]

    return result
# EVOLVE-BLOCK-END
