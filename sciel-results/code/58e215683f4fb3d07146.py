import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    all_masks = list(range(total_masks))
    
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    # Helper: check if a is a subset of b
    def is_subset(a, b):
        return (a & b) == a

    # Helper: check if a and b are comparable
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b

    # Helper: count comparable pairs in a family of masks
    def count_comparable(family):
        count = 0
        L = len(family)
        for i in range(L):
            for j in range(i+1, L):
                if is_comparable(family[i], family[j]):
                    count += 1
        return count

    # 1. Greedy Initialization
    # Start with an empty set, add the most "central" set? 
    # Or just start with a random set and grow.
    # A good start is often a random set, but let's try to start with a set of size n/2.
    # Actually, let's just do a few random restarts of the greedy process.
    
    best_family = []
    best_score = -1
    
    # We will perform a few greedy constructions with different random seeds
    # to find a good starting point for local search.
    
    for restart in range(5):
        random.seed(42 + restart)
        family = []
        used = set()
        
        # First element: pick a random mask
        first = random.choice(all_masks)
        family.append(first)
        used.add(first)
        
        for _ in range(m - 1):
            best_candidate = -1
            best_count = -1
            
            # To speed up, we can sample a subset of candidates if m is large, 
            # but 4096 is small enough to check all.
            for c in all_masks:
                if c in used:
                    continue
                
                # Count how many existing family members are comparable with c
                cnt = 0
                for f in family:
                    if is_comparable(c, f):
                        cnt += 1
                
                if cnt > best_count:
                    best_count = cnt
                    best_candidate = c
            
            if best_candidate != -1:
                family.append(best_candidate)
                used.add(best_candidate)
            else:
                # Should not happen
                break
        
        score = count_comparable(family)
        if score > best_score:
            best_score = score
            best_family = family[:]
            
    # 2. Local Search (Hill Climbing)
    # We will try to swap elements in best_family with elements outside.
    
    current_family = best_family
    current_score = best_score
    used = set(current_family)
    
    # Precompute comparability counts for current family
    # comp_count[i] = number of j != i such that family[i] comparable with family[j]
    
    def get_comparability_counts(fam):
        L = len(fam)
        counts = [0] * L
        for i in range(L):
            for j in range(L):
                if i != j and is_comparable(fam[i], fam[j]):
                    counts[i] += 1
        return counts
    
    counts = get_comparability_counts(current_family)
    
    while time.time() < deadline:
        improved = False
        # Shuffle indices to avoid bias
        indices = list(range(m))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline:
                break
                
            current_mask = current_family[i]
            current_comp = counts[i]
            
            best_candidate = -1
            best_new_comp = current_comp
            
            # Try all candidates outside the family
            for c in all_masks:
                if c in used:
                    continue
                
                # Calculate new comparability count for c
                # It is the number of j != i such that c comparable with current_family[j]
                new_comp = 0
                for j in range(m):
                    if j == i:
                        continue
                    if is_comparable(c, current_family[j]):
                        new_comp += 1
                
                if new_comp > best_new_comp:
                    best_new_comp = new_comp
                    best_candidate = c
            
            if best_candidate != -1 and best_new_comp > current_comp:
                # Perform swap
                used.remove(current_mask)
                used.add(best_candidate)
                
                # Update family
                current_family[i] = best_candidate
                
                # Update counts
                # We need to update counts for all j != i because their comparability 
                # with i changed from current_mask to best_candidate.
                # And counts[i] becomes best_new_comp.
                
                old_mask = current_mask
                new_mask = best_candidate
                
                for j in range(m):
                    if j == i:
                        continue
                    # Check if j was comparable with old_mask
                    was_comp = is_comparable(old_mask, current_family[j])
                    # Check if j is comparable with new_mask
                    is_comp = is_comparable(new_mask, current_family[j])
                    
                    if was_comp and not is_comp:
                        counts[j] -= 1
                    elif not was_comp and is_comp:
                        counts[j] += 1
                
                counts[i] = best_new_comp
                
                # Update total score
                current_score += (best_new_comp - current_comp) // 2 
                # Note: The change in total score is 0.5 * (change in sum of counts).
                # Sum of counts changes by (best_new_comp - current_comp) for index i,
                # and for each j, it changes by +/- 1.
                # Actually, it's easier to just recalculate score or track it carefully.
                # Let's just recalculate score if we want to be safe, but it's O(m^2).
                # m=40, so O(1600). This is cheap.
                # But we already updated counts.
                # The change in total score is exactly (best_new_comp - current_comp) / 2?
                # No.
                # Score = 0.5 * sum(counts).
                # Delta Score = 0.5 * (Delta sum of counts).
                # Delta count[i] = best_new_comp - current_comp.
                # Delta count[j] for each j is +/- 1 or 0.
                # So we should just update current_score by 0.5 * (sum of deltas).
                # Or simpler: just recalculate current_score = count_comparable(current_family)
                # It's only 780 pair checks. Very fast.
                current_score = count_comparable(current_family)
                
                improved = True
        
        if not improved:
            # If no improvement found in a full pass, try a random restart or perturb
            # To avoid getting stuck, we can do a random perturbation
            # Pick a random index and swap with a random outside element
            if time.time() < deadline - 5:
                i = random.randint(0, m-1)
                outside = [c for c in all_masks if c not in used]
                if outside:
                    c = random.choice(outside)
                    old_mask = current_family[i]
                    used.remove(old_mask)
                    used.add(c)
                    current_family[i] = c
                    counts = get_comparability_counts(current_family)
                    current_score = count_comparable(current_family)
    
    # Convert masks back to lists of integers
    result = []
    for mask in current_family:
        s = []
        for b in range(n):
            if (mask >> b) & 1:
                s.append(b)
        result.append(s)
        
    return result
