"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 85.0
N = 12
M = 40
TOTAL = 1 << N

def _comparable(a, b):
    """Check if a and b are comparable (one is strict subset of other)."""
    if a == b:
        return False
    return (a & b) == a or (a & b) == b

def _score_family(family):
    """Count unordered pairs where one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        for j in range(i + 1, m):
            if _comparable(family[i], family[j]):
                count += 1
    return count

def _delta_comparable(mask, family, exclude_idx):
    """Count how many members of family (excluding index exclude_idx) are comparable to mask."""
    cnt = 0
    for k, m in enumerate(family):
        if k != exclude_idx and _comparable(mask, m):
            cnt += 1
    return cnt

def _build_chain_family():
    """Build a family using multiple chains to maximize comparability."""
    # In B_12, longest chain has 13 elements (sizes 0..12)
    # Use 4 chains of 10 elements each, or mix
    # Strategy: pick sets that are nested as much as possible
    family = []
    used = set()
    
    # Build several chains
    # Chain 1: sizes 0,1,2,...,9 (10 elements)
    # We need actual masks. Let's build chains by adding one bit at a time.
    
    # Better: use a greedy approach to build dense comparable sets
    # Start with empty set, then add singletons, then pairs, etc.
    
    # Let's build 4 chains of 10 each
    # Chain i uses bits in a specific pattern
    
    # Actually, let's just do a greedy construction:
    # Start with a few base sets, then add sets that are comparable to many existing ones
    
    # Initialize with a chain of 12 sets (sizes 0-11)
    base_chain = [0]
    current = 0
    for bit in range(N):
        current |= (1 << bit)
        base_chain.append(current)
    # base_chain now has 13 sets: 0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095
    
    family = base_chain[:]
    used = set(family)
    
    # Now add more sets that are comparable to many in the family
    # Try all masks, pick the one with max comparability to current family
    for _ in range(M - len(family)):
        best_mask = None
        best_score = -1
        for mask in range(TOTAL):
            if mask in used:
                continue
            s = _delta_comparable(mask, family, -1)
            if s > best_score:
                best_score = s
                best_mask = mask
        if best_mask is None:
            break
        family.append(best_mask)
        used.add(best_mask)
    
    return family

def _hill_climb(family):
    """Hill climbing with delta evaluation."""
    current = family[:]
    current_score = _score_family(current)
    m = len(current)
    improved = True
    while improved and time.time() < _DEADLINE - 10.0:
        improved = False
        for i in range(m):
            if time.time() > _DEADLINE - 10.0:
                break
            # Try replacing current[i] with each other mask
            old_mask = current[i]
            # Compute how many others are comparable to old_mask
            old_comp = _delta_comparable(old_mask, current, i)
            
            best_new = None
            best_delta = 0
            for new_mask in range(TOTAL):
                if new_mask == old_mask:
                    continue
                # Check if new_mask is already in family at another position
                is_dup = False
                for k in range(m):
                    if k != i and current[k] == new_mask:
                        is_dup = True
                        break
                if is_dup:
                    continue
                new_comp = _delta_comparable(new_mask, current, i)
                delta = new_comp - old_comp
                if delta > best_delta:
                    best_delta = delta
                    best_new = new_mask
            
            if best_new is not None and best_delta > 0:
                current[i] = best_new
                current_score += best_delta
                improved = True
    
    return current, current_score

def _local_search(family, max_iters=20000):
    """Random local search."""
    current = family[:]
    current_score = _score_family(current)
    best = current[:]
    best_score = current_score
    m = len(current)
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
        i = random.randrange(m)
        new_mask = random.randrange(TOTAL)
        if new_mask == current[i]:
            continue
        # Check distinctness
        is_dup = False
        for k in range(m):
            if k != i and current[k] == new_mask:
                is_dup = True
                break
        if is_dup:
            continue
        
        old_mask = current[i]
        old_comp = _delta_comparable(old_mask, current, i)
        new_comp = _delta_comparable(new_mask, current, i)
        delta = new_comp - old_comp
        
        if delta > 0 or (delta == 0 and random.random() < 0.1):
            current[i] = new_mask
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best = current[:]
    
    return best, best_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Phase 1: Build initial family using chain + greedy
    family = _build_chain_family()
    
    # Phase 2: Hill climbing
    family, score = _hill_climb(family)
    
    # Phase 3: Local search
    if time.time() < _DEADLINE - 15.0:
        family2, score2 = _local_search(family, max_iters=10000)
        if score2 > score:
            family = family2
            score = score2
    
    # Convert to list of lists of ints
    result = []
    for mask in family:
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
