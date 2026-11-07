"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 82.0

N = 12
M = 40
TOTAL_MASKS = 1 << N
FULL_MASK = TOTAL_MASKS - 1

# Precompute comparability table: comp[a][b] = 1 if a and b are comparable
# Use a flat array of bytes for speed
_comp_table = bytearray(TOTAL_MASKS * TOTAL_MASKS)

def _build_comp_table():
    """Precompute the comparability table for all pairs of masks."""
    for a in range(TOTAL_MASKS):
        for b in range(a + 1, TOTAL_MASKS):
            if (a & b) == a or (a & b) == b:
                _comp_table[a * TOTAL_MASKS + b] = 1
                _comp_table[b * TOTAL_MASKS + a] = 1

_build_comp_table()

def _comparable(a, b):
    """Check if a and b are comparable using precomputed table."""
    return _comp_table[a * TOTAL_MASKS + b] == 1

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            aj = family[j]
            if _comp_table[ai * TOTAL_MASKS + aj]:
                count += 1
    return count

def _build_chain_family():
    """Build a strong initial family based on a chain structure."""
    # Start with a full chain: one set of each size from 0 to 12
    # Use a specific chain: {0}, {0,1}, {0,1,2}, ..., {0,1,...,11}
    chain = []
    for size in range(N + 1):
        mask = (1 << size) - 1  # {0, 1, ..., size-1}
        chain.append(mask)
    
    family = chain[:]
    used = set(chain)
    
    # Now add more sets greedily, preferring those comparable to many existing sets
    # Sort candidate masks by how many chain members they're comparable to
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    while len(family) < M:
        best_mask = -1
        best_gain = -1
        for mask in all_masks:
            if mask in used:
                continue
            gain = 0
            for f in family:
                if _comp_table[mask * TOTAL_MASKS + f]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
                if gain == len(family):
                    break
        if best_mask >= 0:
            family.append(best_mask)
            used.add(best_mask)
        else:
            # Fallback: pick any unused mask
            for mask in all_masks:
                if mask not in used:
                    family.append(mask)
                    used.add(mask)
                    break
            else:
                break
    
    return family

def _local_search(family, max_time=None):
    """Local search with 1-swap moves using precomputed comparability."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    no_improve_count = 0
    max_no_improve = 300
    
    while time.time() < max_time:
        i = random.randrange(m)
        new_mask = random.choice(all_masks)
        if new_mask == current_family[i]:
            continue
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != i and current_family[k] == new_mask:
                is_dup = True
                break
        if is_dup:
            continue
        
        old_mask = current_family[i]
        # Compute delta using precomputed table
        delta = 0
        for j in range(m):
            if j == i:
                continue
            other = current_family[j]
            old_rel = _comp_table[old_mask * TOTAL_MASKS + other]
            new_rel = _comp_table[new_mask * TOTAL_MASKS + other]
            if new_rel and not old_rel:
                delta += 1
            elif old_rel and not new_rel:
                delta -= 1
        
        if delta > 0:
            current_family[i] = new_mask
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_family, best_score

def _two_swap_local_search(family, max_time=None):
    """Local search with 2-swap moves: replace two elements with two new elements."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < max_time:
        i = random.randrange(m)
        j = random.randrange(m)
        if i == j:
            continue
        
        new_mask_i = random.randrange(TOTAL_MASKS)
        new_mask_j = random.randrange(TOTAL_MASKS)
        
        # Check distinctness
        if new_mask_i == new_mask_j:
            continue
        is_dup = False
        for k in range(m):
            if k != i and k != j:
                if current_family[k] == new_mask_i or current_family[k] == new_mask_j:
                    is_dup = True
                    break
        if is_dup:
            continue
        if new_mask_i == current_family[i] and new_mask_j == current_family[j]:
            continue
        
        old_mask_i = current_family[i]
        old_mask_j = current_family[j]
        
        # Compute delta
        delta = 0
        for k in range(m):
            if k == i or k == j:
                continue
            other = current_family[k]
            # Remove old contributions
            old_rel_i = _comp_table[old_mask_i * TOTAL_MASKS + other]
            old_rel_j = _comp_table[old_mask_j * TOTAL_MASKS + other]
            # Add new contributions
            new_rel_i = _comp_table[new_mask_i * TOTAL_MASKS + other]
            new_rel_j = _comp_table[new_mask_j * TOTAL_MASKS + other]
            
            old_count = old_rel_i + old_rel_j
            new_count = new_rel_i + new_rel_j
            delta += new_count - old_count
        
        # Account for the i-j pair
        old_ij = _comp_table[old_mask_i * TOTAL_MASKS + old_mask_j]
        new_ij = _comp_table[new_mask_i * TOTAL_MASKS + new_mask_j]
        delta += new_ij - old_ij
        
        if delta > 0:
            current_family[i] = new_mask_i
            current_family[j] = new_mask_j
            current_score += delta
            no_improve_count = 0
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    return best_family, best_score

def _systematic_hill_climb(family, max_time=None):
    """Systematic hill climbing: try all possible replacements for each position."""
    if max_time is None:
        max_time = _DEADLINE - 3.0
    
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        for i in range(m):
            if time.time() > max_time:
                break
            old_mask = current_family[i]
            for new_mask in all_masks:
                if new_mask == old_mask:
                    continue
                # Check distinctness
                is_dup = False
                for k in range(m):
                    if k != i and current_family[k] == new_mask:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                delta = 0
                for j in range(m):
                    if j == i:
                        continue
                    other = current_family[j]
                    old_rel = _comp_table[old_mask * TOTAL_MASKS + other]
                    new_rel = _comp_table[new_mask * TOTAL_MASKS + other]
                    if new_rel and not old_rel:
                        delta += 1
                    elif old_rel and not new_rel:
                        delta -= 1
                
                if delta > 0:
                    current_family[i] = new_mask
                    current_score += delta
                    improved = True
                    break
    
    return current_family, current_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Phase 1: Build strong initial family
    family = _build_chain_family()
    score = _score_family(family)
    
    best_family = family[:]
    best_score = score
    
    # Phase 2: Local search with 1-swaps
    t_end = min(_DEADLINE - 15.0, time.time() + 15.0)
    f2, s2 = _local_search(best_family, max_time=t_end)
    if s2 > best_score:
        best_score = s2
        best_family = f2[:]
    
    # Phase 3: Two-swap local search
    if time.time() < _DEADLINE - 15.0:
        t_end = min(_DEADLINE - 15.0, time.time() + 15.0)
        f3, s3 = _two_swap_local_search(best_family, max_time=t_end)
        if s3 > best_score:
            best_score = s3
            best_family = f3[:]
    
    # Phase 4: More 1-swap local search
    if time.time() < _DEADLINE - 10.0:
        t_end = min(_DEADLINE - 10.0, time.time() + 10.0)
        f4, s4 = _local_search(best_family, max_time=t_end)
        if s4 > best_score:
            best_score = s4
            best_family = f4[:]
    
    # Phase 5: Final systematic hill climb if time permits
    if time.time() < _DEADLINE - 5.0:
        f5, s5 = _systematic_hill_climb(best_family, max_time=_DEADLINE - 2.0)
        if s5 > best_score:
            best_score = s5
            best_family = f5
    
    # Convert to list of lists of ints
    result = []
    for mask in best_family:
        subset = []
        for b in range(N):
            if (mask >> b) & 1:
                subset.append(b)
        result.append(subset)
    
    # Ensure types are correct
    for i in range(len(result)):
        result[i] = [int(x) for x in result[i]]
    
    return result

# EVOLVE-BLOCK-END
