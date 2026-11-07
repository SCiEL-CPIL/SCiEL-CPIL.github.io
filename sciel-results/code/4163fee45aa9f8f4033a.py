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
    """Build a family using maximal chain packing."""
    family = []
    used = set()
    remaining = set(range(TOTAL))
    
    while len(family) < M and remaining:
        # Pick a candidate center to start a chain
        # Bias towards middle sizes to allow growth in both directions
        candidates = [m for m in remaining if 4 <= bin(m).count('1') <= 8]
        if not candidates:
            candidates = list(remaining)
            
        center = random.choice(candidates)
        
        # Extend upwards
        up_chain = [center]
        current = center
        while True:
            # Find a superset of current in remaining that adds exactly 1 bit?
            # Actually, just find ANY superset in remaining.
            # To keep it a chain, we need strict inclusion.
            # Greedy: find superset with min size > current size?
            # Or just find one.
            best_sup = None
            min_size = N + 1
            for m in remaining:
                if m == current: continue
                if (m & current) == current: # current is subset of m
                    sz = bin(m).count('1')
                    if sz < min_size:
                        min_size = sz
                        best_sup = m
            if best_sup is None:
                break
            up_chain.append(best_sup)
            current = best_sup
            
        # Extend downwards
        down_chain = [center]
        current = center
        while True:
            best_sub = None
            max_size = -1
            for m in remaining:
                if m == current: continue
                if (m & current) == m: # m is subset of current
                    sz = bin(m).count('1')
                    if sz > max_size:
                        max_size = sz
                        best_sub = m
            if best_sub is None:
                break
            down_chain.append(best_sub)
            current = best_sub
            
        # Combine: down_chain (reversed) + up_chain[1:]
        # down_chain is [center, sub1, sub2...]
        # up_chain is [center, sup1, sup2...]
        chain = down_chain[::-1] + up_chain[1:]
        
        # Add to family
        for m in chain:
            if len(family) < M:
                family.append(m)
                used.add(m)
                remaining.remove(m)
            else:
                break
                
        if not remaining:
            break
            
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
