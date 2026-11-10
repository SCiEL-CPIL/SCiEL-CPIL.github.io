import itertools
import time
import random

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 40 distinct subsets of vertices 0..11 maximizing comparable pairs.
    Strategy:
    1. Start with a 'ball' (down-set) initialization on a 6-element subset to ensure high density.
    2. Extend to 12 elements via local search (hill climbing) with random restarts.
    3. Use efficient delta calculation for swaps.
    """
    n = 12
    m = 40
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Precompute all possible masks for n=12? No, too many.
    # But we only care about the 40 selected sets.
    
    def count_comparable(family):
        """Count number of unordered pairs (A, B) such that A is a strict subset of B."""
        count = 0
        # Convert to set of frozensets for O(1) lookup? 
        # Or just nested loop since m=40 is small.
        # 40*39/2 = 780 checks. Very fast.
        for i in range(m):
            A = family[i]
            for j in range(i + 1, m):
                B = family[j]
                # Check if A is strict subset of B or B is strict subset of A
                if len(A) < len(B) and set(A).issubset(set(B)):
                    count += 1
                elif len(B) < len(A) and set(B).issubset(set(A)):
                    count += 1
        return count

    def get_delta(family, idx_out, idx_in):
        """
        Calculate the change in comparable pairs if we remove family[idx_out] and add idx_in.
        idx_in is a mask (int).
        """
        S_out = family[idx_out]
        S_in = set(idx_in)
        
        # Pairs involving S_out and other sets in family (excluding S_out itself)
        # Pairs involving S_in and other sets in family (excluding S_out, since S_in is new)
        
        delta = 0
        
        for i in range(m):
            if i == idx_out:
                continue
            S_i = family[i]
            
            # Contribution of S_out with S_i
            if len(S_out) < len(S_i) and set(S_out).issubset(set(S_i)):
                delta -= 1
            elif len(S_i) < len(S_out) and set(S_i).issubset(set(S_out)):
                delta -= 1
            
            # Contribution of S_in with S_i
            if len(S_in) < len(S_i) and S_in.issubset(set(S_i)):
                delta += 1
            elif len(S_i) < len(S_in) and set(S_i).issubset(S_in):
                delta += 1
        
        return delta

    # Initialization: Ball on 6 elements
    # We want 40 sets from the boolean lattice of 6 elements (64 sets total).
    # The "ball" of radius 2 has 1 + 6 + 15 = 22 sets.
    # We need 18 more from size 3 (20 available).
    # This forms a valid down-set (if we pick any 18 of the 20 size-3 sets, 
    # wait, a down-set requires that if we pick a set, we pick all its subsets.
    # If we pick a size-3 set, we must pick all its size-2, 1, 0 subsets.
    # Since we already have ALL size 0, 1, 2 sets, we can pick ANY subset of size-3 sets.
    # So a "ball" of size 40 can be formed by taking all sets of size <= 2 and any 18 sets of size 3.
    
    base_elements = list(range(6))
    init_family = []
    
    # Add all sets of size 0, 1, 2 from base_elements
    for r in range(3):
        for combo in itertools.combinations(base_elements, r):
            init_family.append(list(combo))
            
    # We have 1 + 6 + 15 = 22 sets.
    # Need 18 more. Pick 18 random size-3 combinations from base_elements.
    size3_combos = list(itertools.combinations(base_elements, 3))
    random.shuffle(size3_combos)
    for combo in size3_combos[:18]:
        init_family.append(list(combo))
        
    # Ensure length is exactly 40
    if len(init_family) < 40:
        # Fallback: add more size 3 or size 4? 
        # Actually 22 + 18 = 40. So it's exact.
        pass
        
    best_family = [list(s) for s in init_family]
    best_score = count_comparable(best_family)
    
    # Local Search
    # We will try to swap any set in the family with any other subset of {0..11} not in the family.
    # To speed up, we can restrict candidate replacements to sets that share some structure or are close in size.
    # Or just random candidates.
    
    # Pre-generate a pool of candidate masks?
    # There are 4096 subsets.
    # We can iterate through all 4096 masks for each swap attempt?
    # 40 * 4096 = 163,840 delta calculations per full pass.
    # Each delta calculation is O(m) = 40 subset checks.
    # Total ops per pass: ~6.5M.
    # In Python, 6.5M simple ops might take ~0.1-0.5s.
    # So we can do many passes.
    
    # Optimization: 
    # Instead of checking all 4096 masks, we can generate candidates more intelligently.
    # But for now, let's try random restarts with limited full passes.
    
    all_masks = list(range(1 << n))
    
    for restart in range(20):
        if time.time() > deadline:
            break
            
        if restart > 0:
            # Perturb the best family found so far
            current_family = [list(s) for s in best_family]
            # Randomly swap 2-5 elements
            num_swaps = random.randint(2, 5)
            for _ in range(num_swaps):
                idx = random.randint(0, m - 1)
                new_mask = random.choice(all_masks)
                # Ensure distinctness
                if any(set(new_mask) == set(family[i]) for i in range(m)):
                    continue
                # Just replace for perturbation, score will be recalculated
                current_family[idx] = [b for b in range(n) if (new_mask >> b) & 1]
        else:
            current_family = [list(s) for s in best_family]
            
        current_score = count_comparable(current_family)
        
        # Hill Climbing
        no_improve_count = 0
        max_no_improve = 5000
        
        while no_improve_count < max_no_improve and time.time() < deadline:
            improved = False
            # Shuffle indices to avoid bias
            indices = list(range(m))
            random.shuffle(indices)
            
            for idx_out in indices:
                if time.time() > deadline:
                    break
                # Try to find a better replacement
                # Sample a subset of candidates to speed up?
                # Let's try all candidates first. If too slow, we'll see.
                # To make it faster, we can pre-filter candidates by size.
                # Sets with very different sizes are less likely to form comparable pairs.
                
                best_delta = 0
                best_idx_in = -1
                
                # Optimization: Only consider masks that are not already in the family
                current_sets = [set(s) for s in current_family]
                
                # Iterate through candidates
                # To speed up, we can randomize the order of all_masks
                # Or just iterate all.
                for mask in all_masks:
                    # Check if mask is already in family
                    # This is O(m) if done naively. 
                    # Better to maintain a set of frozensets or bitmask representation of current family.
                    # Let's maintain a set of masks for O(1) lookup.
                    pass # Placeholder for optimized logic below
                
                # Let's implement a faster version using a set of masks for the current family
                # We'll rebuild the set of masks at the start of each while loop iteration?
                # Or maintain it.
                
                # Actually, let's just do a simpler loop but with a time check.
                # The inner loop over 4096 masks * 40 sets * 40 delta checks is heavy.
                # Let's reduce candidates.
                # Only consider masks that differ in size by at most 2 from the removed set?
                # Or just random sample of 100 candidates.
                
                # Let's try random sampling of candidates for speed.
                num_candidates = 50
                candidates = random.sample(all_masks, num_candidates)
                
                for mask in candidates:
                    # Check distinctness
                    # We need to ensure the new set is not already in the family
                    # We can check against current_family
                    is_dup = False
                    new_set = set([b for b in range(n) if (mask >> b) & 1])
                    for i in range(m):
                        if set(current_family[i]) == new_set:
                            is_dup = True
                            break
                    if is_dup:
                        continue
                        
                    delta = get_delta(current_family, idx_out, new_set)
                    if delta > best_delta:
                        best_delta = delta
                        best_idx_in = mask
                
                if best_idx_in != -1:
                    # Apply swap
                    current_family[idx_out] = [b for b in range(n) if (best_idx_in >> b) & 1]
                    current_score += best_delta
                    improved = True
                    no_improve_count = 0
                    if current_score > best_score:
                        best_score = current_score
                        best_family = [list(s) for s in current_family]
            
            if not improved:
                no_improve_count += 1

    # Final verification and type conversion
    # Ensure distinctness
    seen = set()
    final_family = []
    for s in best_family:
        fs = frozenset(s)
        if fs not in seen:
            seen.add(fs)
            final_family.append([int(x) for x in s])
            
    # If we lost some due to duplicates (shouldn't happen), pad with random valid sets
    if len(final_family) < m:
        all_masks_set = set()
        for s in final_family:
            mask = 0
            for v in s:
                mask |= (1 << v)
            all_masks_set.add(mask)
        
        for mask in all_masks:
            if mask not in all_masks_set:
                new_set = [b for b in range(n) if (mask >> b) & 1]
                final_family.append(new_set)
                all_masks_set.add(mask)
                if len(final_family) == m:
                    break
                    
    return final_family

# EVOLVE-BLOCK-END
