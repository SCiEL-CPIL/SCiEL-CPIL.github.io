"""Improved solver: maximise a 4-uniform family on [16] with no 3-set sunflower."""
import time
import random
from itertools import combinations

# Pre-compute all 4-subsets as 16-bit integers
_ALL_SETS = []
_SET_MASKS = {}
for _c in combinations(range(16), 4):
    _m = 0
    for _x in _c:
        _m |= (1 << _x)
    _ALL_SETS.append(_m)
    _SET_MASKS[_m] = _c  # bitmask -> tuple of elements

# Pre-compute popcount for 0..(1<<16) is too big; use int.bit_count() (Py3.8+)
# Actually, int.bit_count() is available in Python 3.8+? Let's use bin(x).count('1') fallback.
def _popcount(x):
    return x.bit_count() if hasattr(x, 'bit_count') else bin(x).count('1')

# For a set A (4-element), the "signature" of intersections with another 4-set B:
# |A ∩ B| can be 0,1,2,3,4. Since A≠B, it's 0..3.
# A triple {A,B,C} is a sunflower iff all three pairwise intersections are equal
# and equal to the common core. Equivalently, |A∩B| = |A∩C| = |B∩C| = k for some k,
# AND A∩B∩C has size k (i.e., the pairwise intersections are all the same set).
# Actually, the condition is: A∩B = A∩C = B∩C. This means all three pairwise
# intersections are the same subset. Let's call it the core. The core can have size 0,1,2,3.

def _is_sunflower_triple(a, b, c):
    """Check if three 4-element sets (as bitmasks) form a sunflower.
    A sunflower: A∩B = A∩C = B∩C (all equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_construct(order, deadline):
    """Greedy: iterate through sets in given order, add if no sunflower triple formed."""
    chosen = []
    chosen_masks = []
    for m in order:
        if time.time() > deadline:
            break
        # Check against all pairs in chosen
        ok = True
        for i in range(len(chosen_masks)):
            mi = chosen_masks[i]
            # Quick filter: if |m ∩ mi| > 3, skip (impossible for distinct 4-sets)
            # Actually |m ∩ mi| <= 3 for distinct sets.
            for j in range(i+1, len(chosen_masks)):
                mj = chosen_masks[j]
                if _is_sunflower_triple(m, mi, mj):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen_masks.append(m)
    return chosen_masks

def _local_search(chosen_masks, deadline, max_iters=500):
    """Try to add sets by swapping: remove one set, try to add others."""
    # Represent as set for O(1) lookup
    mask_set = set(chosen_masks)
    best_size = len(chosen_masks)
    
    # Candidate pool: all sets not in chosen
    all_set_list = list(_ALL_SETS)
    not_in = [m for m in all_set_list if m not in mask_set]
    random.shuffle(not_in)
    
    def _can_add(mask, current_set):
        """Check if mask can be added to current_set without forming a sunflower."""
        for i, mi in enumerate(current_set):
            for mj in current_set[i+1:]:
                if _is_sunflower_triple(mask, mi, mj):
                    return False
        return True
    
    improved = True
    iters = 0
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        # Try to add any set from not_in
        for m in not_in[:200]:  # limit candidates per pass
            if time.time() > deadline:
                break
            if m in mask_set:
                continue
            if _can_add(m, list(mask_set)):
                mask_set.add(m)
                not_in.remove(m)
                improved = True
                best_size = len(mask_set)
                break
        if not improved:
            # Try swap: remove one, see if we can add two
            for m_remove in list(mask_set):
                if time.time() > deadline:
                    break
                mask_set.remove(m_remove)
                not_in.append(m_remove)
                added = False
                for m_add in not_in[:100]:
                    if m_add in mask_set:
                        continue
                    if _can_add(m_add, list(mask_set)):
                        mask_set.add(m_add)
                        if m_add in not_in:
                            not_in.remove(m_add)
                        added = True
                        break
                if not added:
                    # Revert
                    mask_set.add(m_remove)
                    not_in.remove(m_remove)
                else:
                    improved = True
                    best_size = len(mask_set)
                    break
    return list(mask_set)

def construct():
    start = time.time()
    deadline = start + 85  # 85s of 90s budget, 5s safety
    
    # Start with greedy in natural order
    random.seed(42)
    order = list(_ALL_SETS)
    random.shuffle(order)
    
    chosen = _greedy_construct(order, deadline)
    best_size = len(chosen)
    best_chosen = chosen[:]
    
    # Try multiple random orders
    for trial in range(50):
        if time.time() > deadline:
            break
        order = list(_ALL_SETS)
        random.shuffle(order)
        chosen = _greedy_construct(order, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_chosen = chosen[:]
    
    # Local search on the best
    if time.time() < deadline:
        improved = _local_search(best_chosen, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_chosen = improved[:]
    
    # Convert back to sorted lists
    result = []
    for m in best_chosen:
        result.append(list(_SET_MASKS[m]))
    result.sort()
    return result
