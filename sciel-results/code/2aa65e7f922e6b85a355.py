"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time

P = {'n': 12, 'm': 40}

def _comparability_count(family_masks):
    """Count unordered pairs where one is a strict subset of the other."""
    n = len(family_masks)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = family_masks[i], family_masks[j]
            if (a & b) == a:  # a is subset of b
                count += 1
            elif (a & b) == b:  # b is subset of a
                count += 1
    return count

def _greedy_construct(deadline):
    """Greedy construction: start with empty set, add sets that maximize comparability."""
    # Start with the empty set (it's a subset of everything)
    chosen = [0]  # mask 0 is empty set
    remaining = set(range(1, 64))  # All non-empty subsets of {0..5}
    
    while len(chosen) < P['m'] and time.time() < deadline:
        best_gain = -1
        best_mask = None
        for mask in remaining:
            # Count how many sets in chosen are comparable with mask
            gain = 0
            for c in chosen:
                if (mask & c) == mask:  # mask is subset of c
                    gain += 1
                elif (mask & c) == c:  # c is subset of mask
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_mask = mask
        
        if best_mask is None:
            break
        chosen.append(best_mask)
        remaining.remove(best_mask)
    
    return chosen

def _local_search(family, deadline):
    """Local search: try swapping sets to improve comparability."""
    current = list(family)
    current_count = _comparability_count(current)
    improved = True
    all_masks = set(range(64))
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            for new_mask in range(64):
                if new_mask in current:
                    continue
                # Try swapping current[i] with new_mask
                old_mask = current[i]
                if old_mask == new_mask:
                    continue
                
                # Calculate gain: remove old, add new
                gain = 0
                for j in range(len(current)):
                    if j == i:
                        continue
                    other = current[j]
                    # Check if old was comparable with other
                    old_comp = 1 if ((old_mask & other) == old_mask or (old_mask & other) == other) else 0
                    # Check if new is comparable with other
                    new_comp = 1 if ((new_mask & other) == new_mask or (new_mask & other) == other) else 0
                    gain += new_comp - old_comp
                
                if gain > 0:
                    current[i] = new_mask
                    current_count += gain
                    improved = True
                    if time.time() > deadline:
                        return current
                    break
    
    return current

def construct():
    """Construct a family of 40 distinct subsets of vertices 0..11 that maximizes comparability."""
    start_time = time.time()
    deadline = start_time + 80  # Leave 10s safety margin
    
    # All subsets of {0..5} as bitmasks
    # Mask bit b corresponds to vertex b
    
    # Strategy: Use greedy construction with local search refinement
    # Start with a good initial family
    
    # Initial family: empty set + all singletons + all pairs + some triples
    # This gives us 1 + 6 + 15 + 18 = 40 sets (we need to pick 18 of the 20 triples)
    
    initial = [0]  # empty set
    for i in range(6):
        initial.append(1 << i)  # singletons
    for i in range(6):
        for j in range(i + 1, 6):
            initial.append((1 << i) | (1 << j))  # all 15 pairs
    # Now we have 1 + 6 + 15 = 22 sets, need 18 more
    # Add 18 of the 20 triples
    triples = []
    for i in range(6):
        for j in range(i + 1, 6):
            for k in range(j + 1, 6):
                triples.append((1 << i) | (1 << j) | (1 << k))
    initial.extend(triples[:18])
    
    # Local search to improve
    result = _local_search(initial, deadline)
    
    # Convert masks to lists of vertices
    family = []
    for mask in result:
        s = [b for b in range(12) if (mask >> b) & 1]
        family.append(s)
    
    # Ensure exactly 40 distinct sets
    seen = set()
    final = []
    for s in family:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            final.append(list(s))
            if len(final) == P['m']:
                break
    
    # If we don't have enough, fill with more sets
    if len(final) < P['m']:
        all_masks = set(range(64))
        for mask in range(64):
            s = [b for b in range(12) if (mask >> b) & 1]
            key = tuple(sorted(s))
            if key not in seen:
                seen.add(key)
                final.append(list(s))
                if len(final) == P['m']:
                    break
    
    return final[:P['m']]
# EVOLVE-BLOCK-END
