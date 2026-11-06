"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
from itertools import combinations

# Precompute all 4-subsets of [1,16] as frozensets and bitmasks
_ALL_SUBSETS = []
_ALL_SUBSETS_BITMASK = []
_BM_TO_IDX = {}
for _row in combinations(range(1, 17), 4):
    _s = frozenset(_row)
    _bm = 0
    for _x in _row:
        _bm |= (1 << _x)
    _ALL_SUBSETS.append(_s)
    _ALL_SUBSETS_BITMASK.append(_bm)
    _BM_TO_IDX[_bm] = len(_ALL_SUBSETS) - 1

_NUM_SUBSETS = len(_ALL_SUBSETS)

def _is_sunflower(a_bm, b_bm, c_bm):
    """Check if three 4-element sets (as bitmasks) form a sunflower.
    Sunflower condition: all three pairwise intersections are equal."""
    ab = a_bm & b_bm
    ac = a_bm & c_bm
    bc = b_bm & c_bm
    return (ab == ac) and (ac == bc)

def _can_add(candidate_bm, chosen_bitmasks):
    """Check if candidate_bm can be added to the family without forming a sunflower."""
    n = len(chosen_bitmasks)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_sunflower(chosen_bitmasks[i], chosen_bitmasks[j], candidate_bm):
                return False
    return True

def _greedy_build(order, deadline):
    """Build a family using greedy addition in the given order."""
    chosen_bitmasks = []
    for idx in order:
        if time.time() > deadline:
            break
        bm = _ALL_SUBSETS_BITMASK[idx]
        if _can_add(bm, chosen_bitmasks):
            chosen_bitmasks.append(bm)
    return chosen_bitmasks

def _try_add_any(family_bitmasks, deadline):
    """Try to add any remaining subset to the family. Returns (new_family, added_idx or None)."""
    family_set = set(family_bitmasks)
    for idx in range(_NUM_SUBSETS):
        if time.time() > deadline:
            break
        bm = _ALL_SUBSETS_BITMASK[idx]
        if bm in family_set:
            continue
        if _can_add(bm, family_bitmasks):
            return family_bitmasks + [bm], idx
    return family_bitmasks, None

def _local_improve(family_bitmasks, deadline):
    """Try to improve the family by swapping out sets to allow more additions."""
    family_set = set(family_bitmasks)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(family_bitmasks)):
            if time.time() > deadline:
                break
            removed = family_bitmasks[i]
            remaining = family_bitmasks[:i] + family_bitmasks[i+1:]
            # Find a candidate not in family that can be added to remaining
            best_candidate = None
            for idx in range(_NUM_SUBSETS):
                bm = _ALL_SUBSETS_BITMASK[idx]
                if bm in family_set:
                    continue
                if _can_add(bm, remaining):
                    best_candidate = idx
                    break
            if best_candidate is not None:
                new_bm = _ALL_SUBSETS_BITMASK[best_candidate]
                family_bitmasks[i] = new_bm
                family_set.remove(removed)
                family_set.add(new_bm)
                improved = True
    return family_bitmasks

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    best_family = []
    
    # Strategy 1: Lexicographic order (reproduces current behavior)
    lex_order = list(range(_NUM_SUBSETS))
    family1 = _greedy_build(lex_order, deadline)
    if len(family1) > len(best_family):
        best_family = family1
    
    # Strategy 2: Multiple random restarts with different seeds
    rng = random.Random(42)
    num_restarts = 100
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        order = list(range(_NUM_SUBSETS))
        rng.shuffle(order)
        family = _greedy_build(order, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Strategy 3: Try to add more to the best family
    if time.time() < deadline:
        new_family, _ = _try_add_any(list(best_family), deadline)
        if len(new_family) > len(best_family):
            best_family = new_family
    
    # Strategy 4: Local improvement on best family  
    if time.time() < deadline:
        improved = _local_improve(list(best_family), deadline)
        if len(improved) > len(best_family):
            best_family = improved
        # Try to add more after improvement
        if time.time() < deadline:
            new_family, _ = _try_add_any(list(best_family), deadline)
            if len(new_family) > len(best_family):
                best_family = new_family
    
    # Convert back to sorted lists
    result = [sorted(_ALL_SUBSETS[_BM_TO_IDX[bm]]) for bm in best_family]
    return result

# EVOLVE-BLOCK-END
