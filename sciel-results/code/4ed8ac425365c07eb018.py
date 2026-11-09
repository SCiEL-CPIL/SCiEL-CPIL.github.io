import time
import random
import math
import numpy as np

N = 12
M = 40
FULL_MASK = (1 << N) - 1
NUM_MASKS = 1 << N

# Precompute comparability matrix
# comp[i][j] is True if i and j are comparable (one is subset of other)
# Using a list of numpy arrays for fast vector operations
# comp[i] is a boolean array of length NUM_MASKS
_comp_cache = None

def _get_comp_matrix():
    global _comp_cache
    if _comp_cache is not None:
        return _comp_cache
    
    comp = [None] * NUM_MASKS
    for i in range(NUM_MASKS):
        # Create an array of bools for all j
        # j is comparable to i if (i & j) == i or (i & j) == j
        # Vectorized:
        all_masks = np.arange(NUM_MASKS, dtype=np.int64)
        inter = i & all_masks
        is_sub = (inter == i)
        is_super = (inter == all_masks)
        comp[i] = (is_sub | is_super).astype(np.int8)
    _comp_cache = comp
    return _comp_cache

def _count_pairs(masks):
    """Count comparable pairs in a list of masks."""
    count = 0
    m = len(masks)
    for i in range(m):
        a = masks[i]
        for j in range(i + 1, m):
            b = masks[j]
            if (a & b) == a or (a & b) == b:
                count += 1
    return count

def _init_family():
    """Initialize family with a mix of chains and greedy fill."""
    random.seed(42)
    family = []
    used = set()
    
    # Generate some maximal chains
    for _ in range(4):
        chain = [0]
        cur = 0
        while cur != FULL_MASK:
            # Pick a random bit to add
            available_bits = [b for b in range(N) if not (cur >> b) & 1]
            if not available_bits:
                break
            b = random.choice(available_bits)
            cur = cur | (1 << b)
            if cur not in used:
                family.append(cur)
                used.add(cur)
            else:
                # If already used, we still track cur but don't add to family
                pass
        # Ensure 0 and FULL_MASK are in family
        if 0 not in used:
            family.append(0)
            used.add(0)
        if FULL_MASK not in used:
            family.append(FULL_MASK)
            used.add(FULL_MASK)
            
    # Greedy fill to reach M
    all_masks = list(range(NUM_MASKS))
    random.shuffle(all_masks)
    
    # Calculate initial score for greedy
    while len(family) < M:
        best_mask = -1
        best_score = -1
        for mask in all_masks:
            if mask in used:
                continue
            score = 0
            for c in family:
                if (mask & c) == mask or (mask & c) == c:
                    score += 1
            if score > best_score:
                best_score = score
                best_mask = mask
        if best_mask == -1:
            break
        family.append(best_mask)
        used.add(best_mask)
        
    # If we have more than M, trim randomly (or keep best, but random is fine for init)
    if len(family) > M:
        family = family[:M]
        used = set(family)
        
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s safety margin
    
    comp = _get_comp_matrix()
    
    # Initialize
    family = _init_family()
    current_score = _count_pairs(family)
    best_score = current_score
    best_family = list(family)
    
    # Maintain compat_count: compat_count[k] = sum_{j in family} comp[k][j]
    # This is a vector of length NUM_MASKS
    compat_count = np.zeros(NUM_MASKS, dtype=np.int32)
    for mask in family:
        compat_count += comp[mask]
    
    # Note: compat_count[k] includes the comparison with all sets in family.
    # When considering replacing family[i] with k:
    # The new contribution of k is sum_{j != i} comp[k][j]
    # This is compat_count[k] - comp[k][family[i]]
    
    # Local Search Loop
    T = 5.0
    cooling = 0.9999
    
    while time.time() < deadline:
        T *= cooling
        if T < 0.1:
            T = 0.1
            
        # Pick a random index to replace
        i = random.randint(0, M - 1)
        old_mask = family[i]
        
        # Calculate scores for all potential replacements
        # scores[k] = compat_count[k] - comp[k][old_mask]
        # We want to pick the best k not in family.
        
        # Vectorized calculation
        # comp[old_mask] is the row for old_mask.
        # We need scores for all k.
        # scores = compat_count - comp[old_mask]
        
        scores = compat_count - comp[old_mask]
        
        # Mask out sets already in family
        # We need to set scores[family[j]] to -inf (or a very low number)
        # But we also need to exclude old_mask itself? No, we are replacing it, so it's not in the "others".
        # However, the formula `compat_count[k] - comp[k][old_mask]` correctly calculates the sum over j != i.
        # We just need to ensure k is not in family (except old_mask, which is allowed as a "no-op" but we want change).
        
        # Create a mask of invalid candidates
        # invalid = np.zeros(NUM_MASKS, dtype=bool)
        # for m in family: invalid[m] = True
        
        # Instead of updating a mask every time, we can just check the top candidates.
        # Find the index of the max score
        # But we must skip sets already in the family.
        
        # Optimization: 
        # The number of sets in family is small (40).
        # We can find the top 100 candidates from `scores` and check if they are in the family.
        
        # Get top 50 indices
        # np.argpartition is faster than full sort
        top_k_indices = np.argpartition(scores, -50)[-50:]
        
        best_candidate = -1
        best_candidate_score = -1
        
        for k in top_k_indices:
            if k in family:
                continue
            # k is a valid candidate
            if scores[k] > best_candidate_score:
                best_candidate_score = scores[k]
                best_candidate = k
        
        if best_candidate == -1:
            # No valid candidate found in top 50? Unlikely. Fallback to random.
            best_candidate = random.randint(0, FULL_MASK)
            while best_candidate in family:
                best_candidate = random.randint(0, FULL_MASK)
            best_candidate_score = scores[best_candidate]
        
        new_mask = best_candidate
        
        # Calculate delta
        # Old contribution of old_mask: sum_{j != i} comp[old_mask][j]
        # This is compat_count[old_mask] - comp[old_mask][old_mask]
        # comp[old_mask][old_mask] is always 1 (reflexive)
        # So old_contrib = compat_count[old_mask] - 1
        
        old_contrib = compat_count[old_mask] - 1
        new_contrib = best_candidate_score
        
        delta = new_contrib - old_contrib
        
        # Acceptance
        if delta > 0 or random.random() < math.exp(delta / T):
            # Accept move
            family[i] = new_mask
            
            # Update compat_count
            # compat_count += comp[new_mask] - comp[old_mask]
            compat_count += comp[new_mask]
            compat_count -= comp[old_mask]
            
            current_score += delta
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(family)
                
        # Occasionally try a swap to escape local minima
        if random.random() < 0.1:
            j = random.randint(0, M - 1)
            if i != j:
                old_i = family[i]
                old_j = family[j]
                
                # Calculate delta for swap
                # This is more complex. 
                # Let's just do a random replacement instead of swap for simplicity and speed,
                # or a specific "chain repair" move.
                # For now, stick to replacement. Swaps are hard to vectorize efficiently for delta.
                pass

    # Convert best_family to lists of vertices
    result = []
    for mask in best_family:
        result.append([b for b in range(N) if (mask >> b) & 1])
    
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    assert len(res) == 40
    assert all(isinstance(s, list) for s in res)
    assert all(isinstance(x, int) for s in res for x in s)
    print("Score:", _count_pairs([int(''.join(str((1 << b) & mask) for b in range(12)) if (mask >> b) & 1 else '0', 2) for mask in [0]*0] or [0]*40)) # Dummy
    # Correct scoring for test
    masks = [sum(1 << b for b in s) for s in res]
    print("Actual Score:", _count_pairs(masks))
