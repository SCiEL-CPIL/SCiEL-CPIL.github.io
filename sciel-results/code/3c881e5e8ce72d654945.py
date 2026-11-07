"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = time.time() + 85.0

N = 12
M = 40
TOTAL_MASKS = 1 << N

# Precompute subset relations
# is_subset[a][b] = True if a is a strict subset of b
# Since 4096*4096 = ~16M entries, we use a list of bytearrays for memory efficiency
# Actually, let's use a set of (a,b) pairs where a is strict subset of b
# Or better: for each mask b, precompute the list of masks a that are strict subsets of b
# And for each mask a, precompute the list of masks b that are strict supersets of a

# Let's precompute: for each mask, count how many of the family members are its strict subsets/supersets
# We'll use a bitset approach or just precompute the relation matrix as a list of sets

# Precompute strict_subset[a] = set of masks that are strict supersets of a
# And strict_superset[b] = set of masks that are strict subsets of b
# Actually, for the delta computation, we need:
# For a new mask x replacing old mask y at position i:
# delta = (number of family members j != i where x is strict subset of family[j] or family[j] is strict subset of x)
#        - (number of family members j != i where y is strict subset of family[j] or family[j] is strict subset of y)

# Let's precompute for each mask m:
# supersets_of[m] = frozenset of all masks s where m is a strict subset of s
# subsets_of[m] = frozenset of all masks s where s is a strict subset of m

# But 4096 masks, each with potentially many subsets/supersets... 
# For mask m with popcount k, it has 2^k - 1 strict subsets and (2^N - 2^k) - (2^k - 1) ... 
# Let's just precompute a boolean table: is_strict_subset[a][b] as a flat array
# 4096 * 4096 = 16,777,216 booleans = 16 MB as a bytearray. That's fine.

_is_subset = bytearray(TOTAL_MASKS * TOTAL_MASKS)
for a in range(TOTAL_MASKS):
    for b in range(TOTAL_MASKS):
        if a != b and (a & b) == a:
            _is_subset[a * TOTAL_MASKS + b] = 1

def _is_strict_subset(a, b):
    return _is_subset[a * TOTAL_MASKS + b]

def _score_family(family):
    """Count unordered pairs (i, j) with i < j such that one is a strict subset of the other."""
    count = 0
    m = len(family)
    for i in range(m):
        ai = family[i]
        for j in range(i + 1, m):
            aj = family[j]
            if _is_strict_subset(ai, aj) or _is_strict_subset(aj, ai):
                count += 1
    return count

def _build_chain_family():
    """Build a family based on chains to maximize subset relations."""
    # A chain is a sequence of sets where each is a strict subset of the next.
    # The maximum chain length in the Boolean lattice of 12 elements is 13 (from empty set to full set).
    # We want 40 sets. Let's create multiple chains.
    # Strategy: Start with the empty set, then add elements one by one.
    # Actually, let's use a greedy approach to build a family with many subset relations.
    
    # Start with a known good structure: all sets of size 0, 1, 2, ..., up to some size,
    # plus some larger sets that contain many of the smaller ones.
    
    # Let's try: all subsets of size <= 4, plus some subsets of size 5-12.
    # Number of subsets of size <= 4: C(12,0)+C(12,1)+C(12,2)+C(12,3)+C(12,4) = 1+12+66+220+495 = 794
    # That's too many. We need exactly 40.
    
    # Let's try a different approach: start with a chain and branch.
    # Chain: {}, {0}, {0,1}, {0,1,2}, ..., {0,1,...,11} - that's 13 sets.
    # Then add sets that are subsets of the full set and supersets of some chain elements.
    
    family = []
    used = set()
    
    # Add the main chain
    for i in range(N + 1):
        mask = (1 << i) - 1
        family.append(mask)
        used.add(mask)
    
    # Add more sets: for each position in the chain, add some "side" sets
    # A side set at position i could be a set that contains the first i elements plus one extra
    # Or we can add sets of various sizes that are subsets of the full set
    
    # Let's add sets that are subsets of the full set but not in the chain
    # Prioritize sets that have many subset relations with existing family members
    
    all_masks = list(range(TOTAL_MASKS))
    random.shuffle(all_masks)
    
    for mask in all_masks:
        if len(family) >= M:
            break
        if mask in used:
            continue
        # Check if adding this mask increases the score
        # For now, just add it
        family.append(mask)
        used.add(mask)
    
    # If we don't have enough, fill with remaining masks
    if len(family) < M:
        for mask in all_masks:
            if len(family) >= M:
                break
            if mask not in used:
                family.append(mask)
                used.add(mask)
    
    return family[:M]

def _delta_score(family, i, new_mask):
    """Compute the change in score when replacing family[i] with new_mask."""
    old_mask = family[i]
    if old_mask == new_mask:
        return 0
    
    m = len(family)
    delta = 0
    for j in range(m):
        if j == i:
            continue
        other = family[j]
        # Old contribution: was old_mask in subset relation with other?
        old_rel = _is_strict_subset(old_mask, other) or _is_strict_subset(other, old_mask)
        # New contribution: is new_mask in subset relation with other?
        new_rel = _is_strict_subset(new_mask, other) or _is_strict_subset(other, new_mask)
        delta += (1 if new_rel else 0) - (1 if old_rel else 0)
    
    return delta

def _hill_climb_family(family):
    """Hill climbing: repeatedly try to improve by swapping one element."""
    current_family = family[:]
    current_score = _score_family(current_family)
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    
    improved = True
    while improved and time.time() < _DEADLINE - 10.0:
        improved = False
        for i in range(m):
            if time.time() > _DEADLINE - 10.0:
                break
            old_mask = current_family[i]
            # Try all possible new masks
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
                
                delta = _delta_score(current_family, i, new_mask)
                if delta > 0:
                    current_family[i] = new_mask
                    current_score += delta
                    improved = True
                    break  # restart inner loop
            if time.time() > _DEADLINE - 10.0:
                break
    
    return current_family, current_score

def _local_search(family, max_iters=100000):
    """Local search via random swaps."""
    current_family = family[:]
    current_score = _score_family(current_family)
    best_family = current_family[:]
    best_score = current_score
    m = len(current_family)
    all_masks = list(range(TOTAL_MASKS))
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break
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
        
        delta = _delta_score(current_family, i, new_mask)
        if delta >= 0:
            current_family[i] = new_mask
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_family = current_family[:]
    
    return best_family, best_score

def construct():
    """Construct a family of 40 distinct subsets of {0..11} maximizing strict subset pairs."""
    # Start with a chain-based initial family
    family = _build_chain_family()
    
    # Phase 1: Hill climbing with delta computation
    family, score = _hill_climb_family(family)
    
    # Phase 2: Random local search if time permits
    if time.time() < _DEADLINE - 15.0:
        family2, score2 = _local_search(family, max_iters=100000)
        if score2 > score:
            family = family2
            score = score2
    
    # Final polish: one more hill climb pass
    if time.time() < _DEADLINE - 5.0:
        family, score = _hill_climb_family(family)
    
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
