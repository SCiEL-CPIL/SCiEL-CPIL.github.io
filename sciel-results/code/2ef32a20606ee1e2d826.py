"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s limit

    n = 16
    k = 4

    # Generate all 4-element subsets as bitmasks
    all_masks = []
    mask_to_set = {}
    for c in combinations(range(1, 17), 4):
        mask = 0
        for x in c:
            mask |= (1 << (x - 1))
        all_masks.append(mask)
        mask_to_set[mask] = sorted(c)

    num_sets = len(all_masks)  # 1820

    # Precompute: for each mask, list of submasks of size 0..3
    # Actually, for sunflower check we need intersection comparisons
    # Key optimization: precompute for each pair of masks, their intersection
    # But that's 1820^2 = 3.3M pairs - too many to store all at once
    
    # Better: use the grouping insight from current program
    # When adding set C to family F, C forms sunflower with A,B in F iff:
    # A∩B = A∩C = B∩C
    # Let I = A∩C. Then we need B∩C = I and A∩B = I.
    # So B must be in family, B∩C = I, and A∩B = I.
    # For fixed C and I, the sets A in family with A∩C = I form a group.
    # Within that group, if any two sets A,B have A∩B = I, sunflower formed.

    def can_add_fast(c_mask, family):
        """
        Check if c_mask can be added without forming a sunflower.
        Uses grouping by intersection with c_mask.
        """
        # Group family members by their intersection with c_mask
        # For each group (same intersection I with c_mask), 
        # check if any pair in group has intersection exactly I
        inter_map = {}
        for a in family:
            inter = a & c_mask
            if inter not in inter_map:
                inter_map[inter] = [a]
            else:
                lst = inter_map[inter]
                # Check if 'a' with any existing 'b' in lst has a&b == inter
                for b in lst:
                    if (a & b) == inter:
                        return False
                lst.append(a)
        return True

    # Phase 1: Deterministic greedy to get a strong baseline
    # Use the lexicographic order (same as inspiration 1)
    def greedy_pass(order, family_start=None):
        family = family_start[:] if family_start else []
        for c_mask in order:
            if time.time() > deadline - 2.0:
                break
            if can_add_fast(c_mask, family):
                family.append(c_mask)
        return family

    best_family = []
    best_size = 0

    # Deterministic pass first (matches inspiration 1 baseline)
    det_family = greedy_pass(all_masks)
    best_family = det_family
    best_size = len(det_family)

    if best_size >= 250:
        return [mask_to_set[m] for m in best_family]

    # Phase 2: Local search / hill climbing
    # Try to improve by:
    # 1. Random restarts with different orderings
    # 2. Local search: try to swap out a set for a better one
    # 3. Simulated annealing style: allow temporary worsening

    rng = random.Random(12345)

    def local_search(family, max_iters=500):
        """Try to improve family by local moves."""
        current = family[:]
        current_size = len(current)
        
        # Build a set for fast membership
        current_set = set(current)
        
        # For local search, we'll try:
        # - Remove a random set, then greedily add back the best candidates
        # - Or try to find a set that can replace another
        
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            
            # Try random removal and re-addition
            if len(current) == 0:
                break
            
            # Pick a random set to potentially remove
            for _ in range(min(20, len(current))):
                if time.time() > deadline - 3.0:
                    break
                
                idx = rng.randint(0, len(current) - 1)
                removed = current[idx]
                temp_family = current[:idx] + current[idx+1:]
                temp_set = set(temp_family)
                
                # Try to add back a different set that was previously rejected
                # or try to add the removed set back in a different position
                # Better: try to find ANY set not in temp_family that can be added
                # and results in a larger or equal family
                
                # Try all sets not in temp_family (this is expensive, so sample)
                candidates = [m for m in all_masks if m not in temp_set]
                rng.shuffle(candidates)
                
                best_new = None
                best_new_size = len(temp_family)
                
                # Try a sample of candidates
                for c_mask in candidates[:100]:
                    if time.time() > deadline - 3.0:
                        break
                    if can_add_fast(c_mask, temp_family):
                        # This set can be added
                        # Check if it's "better" than the removed one
                        # For now, just accept if we can maintain size
                        best_new = c_mask
                        break
                
                if best_new is not None and best_new != removed:
                    new_family = temp_family + [best_new]
                    if len(new_family) >= len(current):
                        current = new_family
                        improved = True
                        break
                elif best_new is None:
                    # Try to add back the removed set
                    if can_add_fast(removed, temp_family):
                        current = temp_family + [removed]
                        # No improvement, but no loss
                        break
        
        return current

    # Run multiple randomized greedy passes with local search
    for restart in range(200):
        if time.time() > deadline - 5.0:
            break
        
        order = list(all_masks)
        rng.shuffle(order)
        
        greedy_result = greedy_pass(order)
        
        if len(greedy_result) > best_size:
            best_size = len(greedy_result)
            best_family = greedy_result
        
        # Apply local search to improve
        if len(greedy_result) >= best_size - 5:  # Only if close to best
            improved = local_search(greedy_result, max_iters=100)
            if len(improved) > best_size:
                best_size = len(improved)
                best_family = improved
        
        if best_size >= 250:
            break

    # Phase 3: Final local search on the best family
    if time.time() < deadline - 5.0 and len(best_family) < 250:
        final_improved = local_search(best_family, max_iters=200)
        if len(final_improved) > best_size:
            best_size = len(final_improved)
            best_family = final_improved

    # Convert best family masks back to sorted lists
    result = [mask_to_set[m] for m in best_family]

    return result

# EVOLVE-BLOCK-END
