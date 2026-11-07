"""
Maximize a subset S of {1,...,128} such that there are no three distinct a,b,c in S
with a+b, a+c, b+c all in S.
Uses bitmask representation and efficient bit-parallel validity checking.
"""

import time
import random

N = 128
MAX_VAL = 128

def _build_valid_pairs():
    """Precompute pairs (i, j) with i < j, i+1, j+1 <= 128, and (i+1)+(j+1) <= 128."""
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if (i + 1) + (j + 1) <= MAX_VAL:
                pairs.append((i, j))
    return pairs

_VALID_PAIRS = _build_valid_pairs()
_NUM_PAIRS = len(_VALID_PAIRS)

# For each pair (i,j), store the bit index for (i+1)+(j+1)
_PAIR_SUM_BITS = [(i + j + 1) for (i, j) in _VALID_PAIRS]

def _is_valid_mask(mask):
    """
    Check if mask represents a valid set.
    Bit i represents integer i+1.
    Valid means: no three distinct a,b,c in S such that a+b, a+c, b+c all in S.
    
    For each pair (a,b) in S where a+b in S, check if there exists c != a,b in S
    such that c+a in S and c+b in S.
    """
    for idx in range(_NUM_PAIRS):
        i, j = _VALID_PAIRS[idx]
        if not ((mask >> i) & 1):
            continue
        if not ((mask >> j) & 1):
            continue
        sum_bit = _PAIR_SUM_BITS[idx]
        if not ((mask >> sum_bit) & 1):
            continue
        
        # Need to find c != i, != j such that c, c+i, c+j all in mask
        # c is at bit position k where k != i, k != j
        # (k+1) + (i+1) = k+i+2 <= 128 => k+i <= 126
        # (k+1) + (j+1) = k+j+2 <= 128 => k+j <= 126
        
        # Build candidate mask: bits k where k in mask, and k+i in mask, and k+j in mask
        # mask >> (i+1): bit k is set iff bit k+i+1 is set in mask, i.e., (k+i+1)+(i+1) ... 
        # Wait, let me think more carefully.
        # We need c (bit k), c+a (bit k+i), c+b (bit k+j) all in mask
        # where a = i+1, b = j+1
        # c+a = (k+1) + (i+1) = k+i+2, bit index = k+i+1
        # c+b = (k+1) + (j+1) = k+j+2, bit index = k+j+1
        
        # So we need: bit k in mask, bit k+i+1 in mask, bit k+j+1 in mask
        # candidates = mask & (mask >> (i+1)) & (mask >> (j+1))
        # But this gives bit k set iff k, k+i+1, k+j+1 all in mask
        # Wait: (mask >> (i+1)) has bit k set iff mask has bit k+i+1 set. Yes.
        
        candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
        # Exclude i and j themselves
        candidates &= ~((1 << i) | (1 << j))
        if candidates:
            return False
    return True

def _construct_baseline():
    """
    Construct a strong baseline: {65..128} U {odd numbers 1..31} U {64}
    """
    mask = 0
    for v in range(65, 129):
        mask |= (1 << (v - 1))
    for v in range(1, 32, 2):
        mask |= (1 << (v - 1))
    mask |= (1 << (64 - 1))
    return mask

def _greedy_fill(mask, order=None):
    """Try to add elements from order to mask if valid."""
    if order is None:
        order = list(range(1, N + 1))
    for v in order:
        bit = 1 << (v - 1)
        if mask & bit:
            continue
        trial = mask | bit
        if _is_valid_mask(trial):
            mask = trial
    return mask

def _local_search_2opt(mask, deadline):
    """
    Local search: try adding single elements, then try remove-one-add-many.
    """
    current_mask = mask
    current_size = current_mask.bit_count()
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_mask(trial):
                current_mask = trial
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
            
        # Phase 2: Try remove-one, then greedy add
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        for v in range(1, N + 1):
            if time.time() > deadline:
                break
            bit = 1 << (v - 1)
            if not (current_mask & bit):
                continue
            
            reduced_mask = current_mask & ~bit
            candidate = _greedy_fill(reduced_mask)
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
                
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
        else:
            break
            
    return current_mask

def _branch_and_bound(mask, deadline):
    """
    Branch and bound: try removing pairs of elements and see if we can fill more.
    """
    if time.time() > deadline:
        return mask
    
    current_size = mask.bit_count()
    best_mask = mask
    best_size = current_size
    
    elems = [i + 1 for i in range(N) if (mask >> i) & 1]
    
    # Try all pairs of elements to remove, then greedy fill
    n = len(elems)
    for i in range(n):
        for j in range(i + 1, n):
            if time.time() > deadline:
                return best_mask
            
            a = elems[i]
            b = elems[j]
            reduced = mask & ~(1 << (a - 1)) & ~(1 << (b - 1))
            
            candidate = _greedy_fill(reduced)
            if candidate.bit_count() > best_size:
                best_size = candidate.bit_count()
                best_mask = candidate
    
    return best_mask

def _random_restart_search(best_mask, best_size, deadline):
    """
    Random restarts with perturbation and local search.
    """
    current_best = best_mask
    current_best_size = best_size
    
    elems = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        random.seed(restart * 31 + 7)
        
        # Perturbation: remove some random elements
        remove_count = random.choice([3, 5, 7, 10])
        if len(elems) < remove_count:
            continue
        
        to_remove = random.sample(elems, remove_count)
        perturbed = best_mask
        for r in to_remove:
            perturbed &= ~(1 << (r - 1))
        
        # Greedy fill
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        filled = _greedy_fill(perturbed, perm)
        
        # Local search
        searched = _local_search_2opt(filled, deadline - 0.5)
        
        if searched.bit_count() > current_best_size:
            current_best = searched
            current_best_size = searched.bit_count()
            elems = [i + 1 for i in range(N) if (current_best >> i) & 1]
        
        # If time allows, try branch and bound on the best found
        if time.time() < deadline - 2:
            bb = _branch_and_bound(current_best, deadline - 1)
            if bb.bit_count() > current_best_size:
                current_best = bb
                current_best_size = bb.bit_count()
                elems = [i + 1 for i in range(N) if (current_best >> i) & 1]
    
    return current_best, current_best_size

def construct():
    start_time = time.time()
    # Safety margin: return well before 90s
    deadline = start_time + 80.0
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # Verify baseline is valid
    if not _is_valid_mask(best_mask):
        # Fallback to just 65..128
        best_mask = 0
        for v in range(65, 129):
            best_mask |= (1 << (v - 1))
        best_size = best_mask.bit_count()
    
    # 2. Greedy fill with multiple orderings
    if time.time() < deadline - 20:
        filled = _greedy_fill(best_mask, list(range(1, N + 1)))
        if filled.bit_count() > best_size:
            best_mask = filled
            best_size = best_mask.bit_count()
        
        if time.time() < deadline - 20:
            filled = _greedy_fill(best_mask, list(range(N, 0, -1)))
            if filled.bit_count() > best_size:
                best_mask = filled
                best_size = best_mask.bit_count()
        
        if time.time() < deadline - 20:
            perm = sorted(range(1, N + 1), key=lambda x: x % 13)
            filled = _greedy_fill(best_mask, perm)
            if filled.bit_count() > best_size:
                best_mask = filled
                best_size = best_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 15:
        searched = _local_search_2opt(best_mask, deadline - 15)
        if searched.bit_count() > best_size:
            best_mask = searched
            best_size = best_mask.bit_count()
    
    # 4. Branch and bound
    if time.time() < deadline - 10:
        bb = _branch_and_bound(best_mask, deadline - 10)
        if bb.bit_count() > best_size:
            best_mask = bb
            best_size = best_mask.bit_count()
    
    # 5. Random restarts
    if time.time() < deadline - 5:
        best_mask, best_size = _random_restart_search(best_mask, best_size, deadline - 3)
    
    # Final validity check
    if not _is_valid_mask(best_mask):
        best_mask = 0
        for v in range(65, 129):
            best_mask |= (1 << (v - 1))
    
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
