"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks
_ALL_SUBSETS_BITMASK = []
_BM_TO_IDX = {}
_IDX_TO_SET = []
for _row in combinations(range(1, 17), 4):
    _bm = 0
    for _x in _row:
        _bm |= (1 << _x)
    _ALL_SUBSETS_BITMASK.append(_bm)
    _BM_TO_IDX[_bm] = len(_ALL_SUBSETS_BITMASK)
    _IDX_TO_SET.append(_row)

_NUM_SUBSETS = len(_ALL_SUBSETS_BITMASK)

def _is_sunflower(a_bm, b_bm, c_bm):
    """Check if three 4-element sets (as bitmasks) form a sunflower.
    Sunflower condition: all three pairwise intersections are equal."""
    ab = a_bm & b_bm
    ac = a_bm & c_bm
    if ab != ac:
        return False
    bc = b_bm & c_bm
    return (ac == bc)

def _can_add(candidate_bm, chosen_bitmasks):
    """Check if candidate_bm can be added to the family without forming a sunflower."""
    n = len(chosen_bitmasks)
    for i in range(n):
        a_bm = chosen_bitmasks[i]
        for j in range(i + 1, n):
            b_bm = chosen_bitmasks[j]
            if _is_sunflower(a_bm, b_bm, candidate_bm):
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

def _local_search(family_bitmasks, deadline, rng):
    """
    Perform local search to improve the family size.
    Strategies:
    1. Try to add any remaining set.
    2. If stuck, try swaps (remove one, add one) to break local optima.
    """
    family_set = set(family_bitmasks)
    current_size = len(family_bitmasks)
    best_family = list(family_bitmasks)
    best_size = current_size
    
    # Precompute list of all bitmasks for quick access
    all_bms = _ALL_SUBSETS_BITMASK
    
    while time.time() < deadline:
        improved = False
        
        # Step 1: Try to add any set
        # To make this faster, we can iterate through unselected sets in a random or fixed order
        # But checking all 1820 sets is O(1820 * M^2). 
        # If M is large, this is slow.
        # Optimization: Only check a subset of candidates?
        # For now, iterate all, but break early if time is tight.
        
        added = False
        # Shuffle indices to avoid bias? Or just iterate?
        # Iterating all is deterministic. Let's try to find *any* valid addition.
        for idx in range(_NUM_SUBSETS):
            if time.time() > deadline:
                break
            bm = all_bms[idx]
            if bm in family_set:
                continue
            if _can_add(bm, family_bitmasks):
                family_bitmasks.append(bm)
                family_set.add(bm)
                current_size += 1
                added = True
                if current_size > best_size:
                    best_size = current_size
                    best_family = list(family_bitmasks)
                break # Found one, restart loop to try more adds
        
        if added:
            continue
            
        # Step 2: If no addition possible, try swaps
        # Pick a random element to remove and a random element to add
        # We need to ensure the new family is valid.
        
        # To avoid getting stuck in a loop of useless swaps, we limit swap attempts per "stuck" phase
        swaps_tried = 0
        max_swaps = 50
        
        while swaps_tried < max_swaps and time.time() < deadline:
            swaps_tried += 1
            
            # Pick random index to remove
            if not family_bitmasks:
                break
            i = rng.randint(0, len(family_bitmasks) - 1)
            remove_bm = family_bitmasks[i]
            
            # Pick random index to add (not in family)
            # We can pick a random global index and check if it's in family
            # Or maintain a list of unselected? Maintaining is expensive to update.
            # Just pick random global idx.
            while True:
                j = rng.randint(0, _NUM_SUBSETS - 1)
                add_bm = all_bms[j]
                if add_bm not in family_set:
                    break
            
            # Perform swap: remove i, add j
            # Check validity of (family - {remove_bm}) + {add_bm}
            # This is O(M^2)
            
            # Optimization: 
            # The new family is valid iff:
            # 1. The old family without remove_bm is valid (it is, since old family was valid)
            # 2. add_bm does not form a sunflower with any pair in (family - {remove_bm})
            
            # So we just need to check _can_add(add_bm, family_bitmasks without remove_bm)
            
            # To do this efficiently, we create a view or list without the element
            # Since M is small enough, list comprehension is fine
            
            temp_family = [x for k, x in enumerate(family_bitmasks) if k != i]
            
            if _can_add(add_bm, temp_family):
                # Valid swap
                family_bitmasks[i] = add_bm
                family_set.remove(remove_bm)
                family_set.add(add_bm)
                # Size remains same, so best_size doesn't change
                # But we might have opened up possibilities for future adds
                # We break out of swap loop to try adds again
                break

    return best_family

def construct():
    start_time = time.time()
    # Safety margin: 85s out of 90s
    deadline = start_time + 85
    
    best_family = []
    best_size = 0
    
    rng = random.Random(42)
    
    # Multiple restarts
    num_restarts = 20
    restart_deadline = deadline - 10 # Leave 10s for final polish
    
    for r in range(num_restarts):
        if time.time() > restart_deadline:
            break
            
        # Generate a random order
        order = list(range(_NUM_SUBSETS))
        rng.shuffle(order)
        
        # Greedy build
        family = _greedy_build(order, restart_deadline)
        
        # Local search
        family = _local_search(family, restart_deadline, rng)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
    
    # Final polish on the best family
    if time.time() < deadline:
        best_family = _local_search(best_family, deadline, rng)
    
    # Convert back to sorted lists
    result = [sorted(_IDX_TO_SET[_BM_TO_IDX[bm]]) for bm in best_family]
    return result

# EVOLVE-BLOCK-END
