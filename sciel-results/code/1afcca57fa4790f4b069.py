import time
import random
import itertools

def construct():
    n = 12
    m = 40
    total_masks = 1 << n
    
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin before 60s limit
    
    # Helper to check if two masks are comparable (one is subset of other)
    def is_comparable(a, b):
        return (a & b) == a or (a & b) == b
    
    # Helper to count comparable pairs in a family of masks
    def count_comparable_pairs(family):
        count = 0
        m_len = len(family)
        for i in range(m_len):
            for j in range(i + 1, m_len):
                if is_comparable(family[i], family[j]):
                    count += 1
        return count
    
    # Helper to count how many sets in family are comparable with a given mask
    def count_comparable_with_mask(mask, family, skip_idx=-1):
        count = 0
        for i, other in enumerate(family):
            if i == skip_idx:
                continue
            if is_comparable(mask, other):
                count += 1
        return count
    
    # Greedy construction: build a chain-like family
    # Strategy: pick masks that maximize comparability with existing family
    # We want many pairs where one is a strict subset of the other
    
    # Key insight: if we build a chain A0 ⊂ A1 ⊂ ... ⊂ A39, 
    # then every pair (Ai, Aj) with i < j is comparable.
    # That gives C(40,2) = 780 comparable pairs.
    # But we need 40 DISTINCT subsets of {0,...,11}.
    # The maximum chain length in a 12-element set is 13 (sizes 0 through 12).
    # So we can't have a full chain of 40.
    
    # Alternative: use multiple chains or a "broad" antichain complement strategy.
    # Actually, let's think differently: maximize comparable pairs.
    # A pair is comparable if one is a subset of the other.
    # Incomparable pairs are those where neither is a subset of the other.
    # Total pairs = C(40,2) = 780.
    # We want to minimize incomparable pairs.
    
    # Strategy: Use local search. Start with a reasonable initial family,
    # then iteratively replace elements to improve the count.
    
    # Initial family: try to create a structure with many comparable pairs.
    # One good approach: take all subsets of a small subset, plus some larger sets.
    # For example, take all subsets of {0,1,2,3} (16 sets), then add sets that contain
    # these as subsets.
    
    # Let's try: all subsets of {0,1,2,3,4} = 32 sets, plus 8 more sets that are supersets.
    # Actually, let's just do local search with a good initialization.
    
    # Better initialization: 
    # Take all 16 subsets of {0,1,2,3}, and 24 random sets that are supersets of some of these.
    # Or even better: start with a random family and let local search do the work.
    
    # Let's try a structured initialization:
    # Family = all subsets of {0,1,2,3} ∪ {sets that contain {0,1,2,3} as subset}
    # Subsets of {0,1,2,3}: 16 sets (masks 0 to 15)
    # Supersets of {0,1,2,3}: masks where bits 0-3 are all 1, bits 4-11 arbitrary: 2^8 = 256 sets
    # Pick 24 of those 256.
    
    base_mask = 0b00001111  # bits 0-3 set
    base_subsets = []
    for mask in range(1 << 4):
        base_subsets.append(mask)
    
    # Pick 24 supersets of base_mask
    supersets = []
    for mask in range(base_mask, total_masks):
        if (mask & base_mask) == base_mask:
            supersets.append(mask)
    
    # Shuffle and pick 24
    random.shuffle(supersets)
    init_family = base_subsets + supersets[:24]
    
    # Ensure all are distinct (they should be by construction)
    init_family = list(set(init_family))
    if len(init_family) < 40:
        # Fill with random masks
        existing = set(init_family)
        while len(init_family) < 40:
            r = random.randint(0, total_masks - 1)
            if r not in existing:
                init_family.append(r)
                existing.add(r)
    
    current_family = init_family[:40]
    current_score = count_comparable_pairs(current_family)
    
    # Local search: try to replace one element with another to improve score
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try each position
        for i in range(m):
            if time.time() > deadline:
                break
            old_mask = current_family[i]
            # Try replacing with a random mask
            # Or try all possible masks? 4096 * 40 = 163840 checks per pass, each O(40)
            # That's ~6.5M operations per pass, should be fast enough.
            best_new_mask = None
            best_delta = 0
            current_with_i_removed_count = 0
            
            # For efficiency, precompute counts for each other element
            # Actually, let's just try a subset of candidate replacements
            # Try all 4096 masks
            for new_mask in range(total_masks):
                if new_mask == old_mask:
                    continue
                if new_mask in current_family:
                    continue
                # Check if new_mask is already in family
                # This is O(40) check, but we can skip if we track a set
                # For now, let's just do the comparability count
                # Count comparable pairs involving new_mask with all except position i
                cnt = 0
                for j in range(m):
                    if j == i:
                        continue
                    if is_comparable(new_mask, current_family[j]):
                        cnt += 1
                # Count comparable pairs involving old_mask with all except position i
                cnt_old = 0
                for j in range(m):
                    if j == i:
                        continue
                    if is_comparable(old_mask, current_family[j]):
                        cnt_old += 1
                delta = cnt - cnt_old
                if delta > best_delta:
                    best_delta = delta
                    best_new_mask = new_mask
            
            if best_new_mask is not None and best_delta > 0:
                current_family[i] = best_new_mask
                current_score += best_delta
                improved = True
    
    # Convert masks to lists of vertices
    result = []
    for mask in current_family:
        subset = [b for b in range(n) if (mask >> b) & 1]
        result.append(subset)
    
    return result
