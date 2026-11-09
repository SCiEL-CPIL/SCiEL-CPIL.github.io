"""Size-efficient sorting network on 17 wires via bitmask local search."""
import time

N = 17
SEED_COUNT = 136
DEADLINE = 82.0  # seconds; evaluator kills at 90s, keep safety margin
MAX_COMPARATORS = 4096

# EVOLVE-BLOCK-START

# ---- Precomputed constants ----
MASK = (1 << N) - 1
# For each wire position p, the expected sorted-output bitmask has 1s in the
# first p wires (positions 0..p-1) when the input has exactly p ones.
# We verify: for every subset S of wires (represented as bitmask), after
# applying the network, the output bitmask must equal the "canonical" form:
# the popcount(S) lowest wires set.
# canonical(popcount) = (1 << popcount) - 1

def _canonical_mask(x):
    """Return mask with popcount(x) lowest bits set."""
    c = x.bit_count()
    return (1 << c) - 1

def _apply_comparator(mask, i, j):
    """Apply compare-exchange (i,j): min to wire i, max to wire j.
    mask is a bitmask where bit k = 1 means wire k has value 1.
    After compare-exchange: wire i gets min, wire j gets max.
    So if wire i has 1 and wire j has 0 -> swap: i becomes 0, j becomes 1.
    """
    bit_i = (mask >> i) & 1
    bit_j = (mask >> j) & 1
    if bit_i == 1 and bit_j == 0:
        # Need to swap: clear bit i, set bit j
        mask ^= (1 << i) | (1 << j)
    return mask

def _apply_network(mask, comps):
    """Apply full network to a single bitmask input."""
    for i, j in comps:
        bit_i = (mask >> i) & 1
        bit_j = (mask >> j) & 1
        if bit_i == 1 and bit_j == 0:
            mask ^= (1 << i) | (1 << j)
    return mask

def _verify_network(comps):
    """Verify network sorts all 2^17 inputs using zero-one principle.
    Returns True if valid."""
    for x in range(1 << N):
        out = _apply_network(x, comps)
        if out != _canonical_mask(x):
            return False
    return True

def _seed_network():
    """Odd-even transposition network for N wires: N*(N-1)/2 comparators.
    For N=17: 136 comparators."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove_comparator(comps, idx):
    """Try removing comparator at index idx; return True if still valid."""
    new_comps = comps[:idx] + comps[idx+1:]
    return _verify_network(new_comps)

def _try_replace_comparator(comps, idx, new_i, new_j):
    """Try replacing comparator at idx with (new_i, new_j); return True if valid."""
    if new_i >= new_j or new_i < 0 or new_j >= N:
        return False
    new_comps = comps[:idx] + [(new_i, new_j)] + comps[idx+1:]
    return _verify_network(new_comps)

def _local_search_remove(comps, deadline):
    """Iteratively try to remove redundant comparators."""
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            if _try_remove_comparator(comps, idx):
                comps = comps[:idx] + comps[idx+1:]
                changed = True
                break  # restart scan from beginning
    return comps

def _local_search_replace(comps, deadline):
    """Try to replace comparators with better (shorter distance) ones."""
    # Priority: try removing first, then replacing with adjacent
    for idx in range(len(comps)):
        if time.time() > deadline:
            break
        i, j = comps[idx]
        # Try replacing with a closer comparator
        for new_i in range(max(0, i - 1), min(N - 1, i + 1)):
            for new_j in range(max(new_i + 1, j - 1), min(N, j + 1)):
                if (new_i, new_j) == (i, j):
                    continue
                if _try_replace_comparator(comps, idx, new_i, new_j):
                    comps[idx] = (new_i, new_j)
                    break
            if time.time() > deadline:
                break
    return comps

def _try_insert_comparator(comps, i, j, deadline):
    """Try inserting a comparator at various positions."""
    # Try inserting after each existing comparator
    for pos in range(len(comps) + 1):
        if time.time() > deadline:
            return None
        new_comps = comps[:pos] + [(i, j)] + comps[pos:]
        if _verify_network(new_comps):
            return new_comps
    return None

def _construct_search(deadline):
    """Main search: start from seed, remove redundancies, try improvements."""
    comps = _seed_network()
    
    # Phase 1: Remove redundant comparators (greedy)
    comps = _local_search_remove(comps, deadline)
    
    if time.time() >= deadline:
        return comps
    
    # Phase 2: Try to reduce depth by reordering
    # Phase 3: Try more aggressive removals with different orderings
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(comps)):
            if time.time() > deadline:
                break
            if _try_remove_comparator(comps, idx):
                comps = comps[:idx] + comps[idx+1:]
                improved = True
    
    return comps

# Cache for known good networks (hardcoded 71-comparator if found)
_KNOWN_71 = None

def _try_known_71():
    """Attempt to verify a known 71-comparator network if available."""
    # The known best 17-wire network has 71 comparators, depth 12.
    # We'll rely on search to find it, but structure the search to be effective.
    return None

def construct():
    """Construct a sorting network for 17 wires.
    Returns ordered list of (i, j) pairs."""
    start = time.time()
    deadline = start + DEADLINE
    
    # Start with seed
    comps = _seed_network()
    
    # Verify seed is correct (should be)
    if not _verify_network(comps):
        raise ValueError("Seed network is invalid")
    
    best = comps
    best_len = len(best)
    
    # Iterative local search with multiple passes
    for iteration in range(50):
        if time.time() > deadline - 2.0:
            break
        
        # Try removal
        new_comps = _local_search_remove(list(best), deadline - 1.0)
        if len(new_comps) < best_len:
            best = new_comps
            best_len = len(best)
        
        # Try a few random perturbations + repair
        if time.time() > deadline - 2.0:
            break
        
        # Try swapping adjacent comparators
        for idx in range(len(best) - 1):
            if time.time() > deadline - 1.0:
                break
            swapped = list(best)
            swapped[idx], swapped[idx+1] = swapped[idx+1], swapped[idx]
            if _verify_network(swapped):
                # Check if swap helps (it doesn't reduce count, but may enable removals)
                removed = _local_search_remove(swapped, deadline - 1.0)
                if len(removed) < best_len:
                    best = removed
                    best_len = len(best)
    
    # Final verification
    if not _verify_network(best):
        # Fallback to seed
        best = _seed_network()
    
    return best

# EVOLVE-BLOCK-END
