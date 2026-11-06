import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin

    n = 16
    k = 4
    full_mask = (1 << n) - 1

    # Generate all 4-element subsets as masks
    all_masks = []
    for c in combinations(range(n), k):
        mask = 0
        for x in c:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_sets = len(all_masks)
    
    # Precompute indices for each mask to speed up combination generation
    # mask_indices[mask] = list of indices
    mask_indices = {}
    for mask in all_masks:
        inds = []
        temp = mask
        idx = 0
        while temp:
            if temp & 1:
                inds.append(idx)
            temp >>= 1
            idx += 1
        mask_indices[mask] = inds

    # Precompute conflict scores (static heuristic)
    # score[x] = sum of number of sets blocked by pair (x, y) for all y != x
    conflict_scores = [0] * num_sets
    
    # To avoid O(N^2 * gen) in precomputation which might be slow, 
    # we can estimate or just use random. 
    # Let's try a lighter precomputation: 
    # Just count how many sets overlap with x? No, that's not the conflict.
    # Let's skip static scores and use multiple random restarts with a twist:
    # Sort by random key, but bias towards sets that are "isolated" in some sense?
    # Actually, let's just do random restarts. The "Least Conflicts" dynamic approach is too slow.
    # But we can do a "Greedy with Backtracking" or just "Greedy with Restart".
    
    # Let's implement the on-the-fly forbidden set update.
    
    def popcount(x):
        return bin(x).count('1')
        
    # Helper to generate submasks of a given mask with a given popcount
    # Since we only need sizes 1,2,3,4 and the base mask is small, 
    # we can use combinations on the indices.
    
    def get_blocked(x, y, selected_set, forbidden_set):
        """
        Returns a list of masks z that are blocked by the pair (x, y).
        z must form a sunflower with x and y.
        """
        k_mask = x & y
        f_mask = (x | y) ^ k_mask
        avail_mask = full_mask ^ f_mask
        need = k - popcount(k_mask)
        
        if need == 0:
            # z must be exactly k_mask
            # But z must be distinct from x and y.
            # If need=0, z = k_mask.
            # If k_mask == x or k_mask == y, it's not a valid distinct set.
            # Also z must be a 4-set. k_mask has popcount k_mask.
            # If need=0, popcount(k_mask)=4. So x and y are subsets of k_mask?
            # If x,y are 4-sets and intersect in 4, x=y. But x!=y.
            # So need=0 is impossible for distinct 4-sets x,y.
            return []
            
        # Get indices of avail_mask
        avail_inds = mask_indices[avail_mask]
        # We need to choose 'need' elements from avail_inds
        # And combine with k_mask
        blocked = []
        for combo in combinations(avail_inds, need):
            z_extra = 0
            for idx in combo:
                z_extra |= (1 << idx)
            z_mask = k_mask | z_extra
            
            # z_mask is a candidate.
            # It must be distinct from x and y.
            if z_mask == x or z_mask == y:
                continue
                
            # If z_mask is not already forbidden and not in selected, it becomes forbidden.
            # We return it so the caller can add it to forbidden.
            blocked.append(z_mask)
        return blocked

    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Optimization: Precompute mask_indices for ALL masks? No, only those in all_masks are needed?
    # Wait, avail_mask can be any mask.
    # So we need mask_indices for any mask that is a subset of full_mask.
    # There are 2^16 = 65536 masks.
    # Precomputing indices for all 65536 masks is fast.
    all_indices = [None] * (1 << n)
    for m in range(1 << n):
        inds = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                inds.append(idx)
            temp >>= 1
            idx += 1
        all_indices[m] = inds

    # Re-define get_blocked to use all_indices
    def get_blocked_fast(x, y):
        k_mask = x & y
        f_mask = (x | y) ^ k_mask
        avail_mask = full_mask ^ f_mask
        need = k - popcount(k_mask)
        
        if need == 0:
            return []
            
        avail_inds = all_indices[avail_mask]
        blocked = []
        for combo in combinations(avail_inds, need):
            z_extra = 0
            for idx in combo:
                z_extra |= (1 << idx)
            z_mask = k_mask | z_extra
            if z_mask != x and z_mask != y:
                blocked.append(z_mask)
        return blocked

    # Run multiple greedy passes
    while time.time() < deadline:
        order = list(all_masks)
        rng.shuffle(order)
        
        selected = []
        selected_set = set() # For O(1) lookup
        forbidden = set()
        
        for x in order:
            if x in forbidden:
                continue
            if x in selected_set:
                continue
                
            # Try to add x
            # Check if adding x creates any conflict with existing selected pairs?
            # The 'forbidden' set already contains all z that conflict with ANY pair in selected.
            # So if x is not in forbidden, it is safe to add.
            # Wait, does 'forbidden' contain x if x conflicts with a pair?
            # Yes, when we added the second element of that pair, we added x to forbidden.
            
            # So we just add x.
            selected.append(x)
            selected_set.add(x)
            
            # Update forbidden
            # For every y in selected (except x), add blocks(x, y) to forbidden
            # We iterate over selected[:-1]
            for y in selected[:-1]:
                blocks = get_blocked_fast(x, y)
                for z in blocks:
                    if z not in selected_set:
                        forbidden.add(z)
                        
        if len(selected) > best_size:
            best_size = len(selected)
            best_family = selected.copy()
            
        if best_size >= 250:
            break
            
    # Convert back to sorted lists
    result = []
    for m in best_family:
        s = []
        temp = m
        idx = 0
        while temp:
            if temp & 1:
                s.append(idx + 1)
            temp >>= 1
            idx += 1
        result.append(s)
        
    return result

# EVOLVE-BLOCK-END
