import time
import random
from itertools import combinations

P = {'n': 12, 'm': 40}

def _count_comparable_pairs(sets):
    """Count unordered pairs where one is a strict subset of the other."""
    m = len(sets)
    set_list = [frozenset(s) for s in sets]
    cnt = 0
    for i in range(m):
        for j in range(i + 1, m):
            si = set_list[i]
            sj = set_list[j]
            if len(si) < len(sj):
                if si.issubset(sj):
                    cnt += 1
            elif len(sj) < len(si):
                if sj.issubset(si):
                    cnt += 1
    return cnt

def _initial_solution():
    """Create a reasonable initial solution using a dense core."""
    n = P['n']
    m = P['m']
    
    core_size = 6
    core = list(range(core_size))
    
    all_subsets_core = []
    for mask in range(1 << core_size):
        s = tuple(i for i in range(core_size) if (mask >> i) & 1)
        all_subsets_core.append(s)
    
    all_subsets_core.sort(key=lambda x: (len(x), x))
    
    return [list(s) for s in all_subsets_core[:m]]

def _build_subset_superset_tables(n):
    """Precompute for each mask: list of masks that are strict subsets, and strict supersets."""
    size = 1 << n
    subset_of = [[] for _ in range(size)]  # subset_of[mask] = list of masks m' where m' is strict subset of mask
    superset_of = [[] for _ in range(size)]  # superset_of[mask] = list of masks m' where mask is strict subset of m'
    
    for mask in range(size):
        # Find all strict subsets of mask
        sub = mask
        while sub > 0:
            sub = (sub - 1) & mask
            if sub < mask:
                subset_of[mask].append(sub)
        # Empty set is always a strict subset (if mask != 0)
        if mask != 0:
            subset_of[mask].append(0)
        
        # Find all strict supersets of mask
        # A superset of mask is mask | extra where extra has bits not in mask
        # Equivalently, for each other mask2, check if mask is subset of mask2
        # More efficient: iterate over all masks and check
        # Actually, let's do it differently: for each mask, its supersets are all masks that contain it
        # We can compute this by: for each mask, iterate over all masks2 and check
        # But that's O(4^n). Let's use the property:
        # supersets of mask = {mask2 : mask2 & mask == mask and mask2 != mask}
        # We can enumerate by adding one bit at a time... but that's complex.
        # Alternative: just iterate all masks for each mask. With n=12, size=4096, this is 4096^2 ~ 16M, acceptable.
    
    # More efficient approach for supersets:
    # For each mask, its strict supersets are all mask2 where (mask2 | mask) == mask2 and mask2 != mask
    # i.e., mask is a subset of mask2 and mask2 != mask
    # We can compute: for each mask2, it is a superset of all its subsets.
    # So: for each mask2, for each strict subset s of mask2, add mask2 to superset_of[s]
    
    for mask2 in range(size):
        sub = mask2
        while sub > 0:
            sub = (sub - 1) & mask2
            if sub < mask2:
                superset_of[sub].append(mask2)
        if mask2 != 0:
            superset_of[0].append(mask2)
    
    return subset_of, superset_of

def _local_search_bitmask(sets, time_limit=85.0):
    """Improve solution using bitmask representation with precomputed tables."""
    start = time.time()
    n = P['n']
    m = P['m']
    size = 1 << n
    
    # Convert to bitmask representation
    current_masks = [0] * m
    for i, s in enumerate(sets):
        mask = 0
        for v in s:
            mask |= (1 << v)
        current_masks[i] = mask
    
    # Precompute subset/superset tables
    subset_of, superset_of = _build_subset_superset_tables(n)
    
    # Precompute popcount
    popcount = [0] * size
    for mask in range(1, size):
        popcount[mask] = popcount[mask >> 1] + (mask & 1)
    
    # Current set of masks for O(1) duplicate check
    current_set = set(current_masks)
    
    # Compute current score
    current_score = 0
    for i in range(m):
        for j in range(i + 1, m):
            mi, mj = current_masks[i], current_masks[j]
            if mi & mj == mi and mi != mj:
                current_score += 1
            elif mi & mj == mj and mi != mj:
                current_score += 1
    
    # Helper: compute comparable pairs for a mask against all others
    def _comparable_count(mask, exclude_idx):
        """Count how many other masks (excluding exclude_idx) are comparable with mask."""
        cnt = 0
        # mask is strict subset of other
        for other in superset_of[mask]:
            if other in current_set:
                # Check if this other is at exclude_idx
                # We need to know which index has this mask
                # This is tricky with just a set. Let's use a dict.
                pass
        # This approach is getting complicated. Let me use a different method.
        # For each other index, check if comparable.
        for i in range(m):
            if i == exclude_idx:
                continue
            other = current_masks[i]
            if mask & other == mask and mask != other:
                cnt += 1
            elif mask & other == other and mask != other:
                cnt += 1
        return cnt
    
    # Better approach: maintain a count of how many times each mask appears (should be 0 or 1)
    # and use the precomputed tables for fast counting
    
    # Let's redo: for a given mask, the number of comparable pairs it forms with the current family
    # = (number of masks in family that are strict supersets of mask) + (number of masks in family that are strict subsets of mask)
    # We can compute this by iterating over superset_of[mask] and subset_of[mask] and checking membership in current_set
    
    def _comparable_count_fast(mask, exclude_mask):
        """Count comparable pairs between mask and current family, excluding exclude_mask itself."""
        cnt = 0
        # Strict supersets of mask that are in the family
        for sup in superset_of[mask]:
            if sup in current_set and sup != exclude_mask:
                cnt += 1
        # Strict subsets of mask that are in the family
        for sub in subset_of[mask]:
            if sub in current_set and sub != exclude_mask:
                cnt += 1
        return cnt
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for idx in range(m):
            if (time.time() - start) > time_limit:
                break
            old_mask = current_masks[idx]
            old_comparable = _comparable_count_fast(old_mask, old_mask)
            
            # Try replacing with each other mask
            best_delta = 0
            best_new_mask = None
            for new_mask in range(size):
                if new_mask == old_mask:
                    continue
                if new_mask in current_set:
                    continue
                
                new_comparable = _comparable_count_fast(new_mask, old_mask)
                delta = new_comparable - old_comparable
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_delta > 0 and best_new_mask is not None:
                # Accept the swap
                current_set.remove(old_mask)
                current_masks[idx] = best_new_mask
                current_set.add(best_new_mask)
                current_score += best_delta
                improved = True
                break  # Restart from beginning after improvement
    
    # Convert back to lists
    result = []
    for mask in current_masks:
        s = [i for i in range(n) if (mask >> i) & 1]
        result.append(s)
    
    return result, current_score

def _multi_restart_local_search(sets, time_limit=85.0):
    """Run local search multiple times with different initial solutions."""
    start = time.time()
    best_sol = None
    best_score = -1
    
    # First, try the given initial solution
    sol, score = _local_search_bitmask(sets, time_limit=min(time_limit, 40.0))
    if score > best_score:
        best_score = score
        best_sol = sol
    
    # Try a few more random restarts if time permits
    n = P['n']
    m = P['m']
    
    for restart in range(5):
        if (time.time() - start) > time_limit - 10:
            break
        
        # Generate a random initial solution biased toward smaller sets
        random_sol = []
        used_masks = set()
        for _ in range(m):
            # Bias toward smaller sets
            max_size = random.choice([3, 4, 4, 5, 5, 6])
            for attempt in range(100):
                s = random.sample(range(n), random.randint(0, max_size))
                mask = 0
                for v in s:
                    mask |= (1 << v)
                if mask not in used_masks:
                    used_masks.add(mask)
                    random_sol.append(s)
                    break
        
        if len(random_sol) == m:
            sol, score = _local_search_bitmask(random_sol, time_limit=min(time_limit - (time.time() - start), 15.0))
            if score > best_score:
                best_score = score
                best_sol = sol
    
    if best_sol is None:
        best_sol = sets
        best_score = _count_comparable_pairs(sets)
    
    return best_sol, best_score

def construct():
    start = time.time()
    deadline = 85.0  # Safety margin
    
    # Initial solution
    sol = _initial_solution()
    score = _count_comparable_pairs(sol)
    
    # Multi-restart local search
    sol, score = _multi_restart_local_search(sol, time_limit=deadline - (time.time() - start))
    
    # Ensure types are correct: lists of Python ints
    result = []
    for s in sol:
        result.append([int(x) for x in s])
    
    return result
