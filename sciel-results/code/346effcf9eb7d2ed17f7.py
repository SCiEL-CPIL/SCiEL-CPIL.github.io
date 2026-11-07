import time
import random

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    deadline = time.time() + 85  # 5s safety margin

    # Precompute subset/superset relationships for speed
    # For each mask, we can check comparability with (a & ~b) == 0 or (b & ~a) == 0

    def is_comparable(a, b):
        return (a & ~b) == 0 or (b & ~a) == 0

    def calc_score(family):
        """Calculate number of comparable pairs in family (list of masks)."""
        score = 0
        k = len(family)
        for i in range(k):
            for j in range(i + 1, k):
                if is_comparable(family[i], family[j]):
                    score += 1
        return score

    def calc_score_incremental(family_set, add_mask):
        """Calculate score increase when adding add_mask to family_set."""
        inc = 0
        for m2 in family_set:
            if m2 != add_mask and is_comparable(add_mask, m2):
                inc += 1
        return inc

    # Initial family: try a chain-like structure
    # Start with empty set, then build up
    # A good initial structure: all subsets of size 0 to some k, but that might be too many
    # Let's start with a random valid family and improve

    # Better initial: take all singletons plus empty set plus some larger sets
    # Actually, let's use a greedy construction

    def greedy_construct():
        """Greedy: start with empty, add masks that maximize comparable pairs."""
        family = [0]  # start with empty set
        remaining = list(range(1, total_masks))
        random.shuffle(remaining)
        
        # Remove empty set from remaining since we already have it
        remaining = [x for x in remaining if x != 0]
        
        while len(family) < m:
            best_mask = None
            best_gain = -1
            # Sample a subset of remaining to keep it fast
            sample_size = min(200, len(remaining))
            sample = random.sample(remaining, sample_size) if len(remaining) > sample_size else remaining
            for mask in sample:
                gain = calc_score_incremental(set(family), mask)
                if gain > best_gain:
                    best_gain = gain
                    best_mask = mask
            if best_mask is not None:
                family.append(best_mask)
                remaining.remove(best_mask)
            else:
                # Fallback: just add a random mask
                mask = random.choice(remaining)
                family.append(mask)
                remaining.remove(mask)
        return family

    # Hill climbing: try to swap a mask in the family with one outside
    def hill_climb(family, deadline):
        family_set = set(family)
        current_score = calc_score(family)
        
        # Precompute all masks not in family
        outside = [x for x in range(total_masks) if x not in family_set]
        
        while time.time() < deadline:
            improved = False
            # Try random swaps
            for _ in range(500):
                if time.time() >= deadline:
                    break
                i = random.randint(0, len(family) - 1)
                old_mask = family[i]
                # Pick a random outside mask
                if not outside:
                    break
                new_mask = random.choice(outside)
                
                # Calculate score change
                # Remove old_mask: lose comparable pairs with other family members
                old_loss = 0
                for j in range(len(family)):
                    if j != i and is_comparable(old_mask, family[j]):
                        old_loss += 1
                
                new_gain = 0
                for j in range(len(family)):
                    if j != i and is_comparable(new_mask, family[j]):
                        new_gain += 1
                
                delta = new_gain - old_loss
                if delta > 0:
                    # Accept the swap
                    family[i] = new_mask
                    family_set.remove(old_mask)
                    family_set.add(new_mask)
                    outside.remove(old_mask)
                    outside.append(new_mask)
                    current_score += delta
                    improved = True
            
            if not improved:
                # Try some random restarts of local search with different neighborhoods
                # Or just continue with more swaps
                pass
        
        return family, current_score

    # Start with greedy construction
    best_family = greedy_construct()
    best_score = calc_score(best_family)
    
    # Improve with hill climbing
    best_family, best_score = hill_climb(best_family, deadline)
    
    # Try a few more restarts if time permits
    restart_count = 0
    while time.time() < deadline - 5 and restart_count < 10:
        restart_count += 1
        candidate = greedy_construct()
        candidate, score = hill_climb(candidate, deadline - 5)
        if score > best_score:
            best_family = candidate
            best_score = score
    
    # Convert masks to lists of vertices
    result = []
    for mask in best_family:
        lst = [b for b in range(n) if (mask >> b) & 1]
        result.append(lst)
    
    # Verify: exactly 40 distinct subsets
    assert len(result) == 40
    assert len(set(tuple(x) for x in result)) == 40
    
    return result
