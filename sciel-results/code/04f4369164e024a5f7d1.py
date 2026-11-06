import random
import time
import sys

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _can_add(sums, inv):
    """Check if adding inv to the set of sums causes a collision."""
    # For small sets, generator is faster due to early exit on collision
    # For large sets, set intersection might be faster if no collision, 
    # but we are usually near the limit where collisions are frequent.
    # Let's use the set intersection method as it's C-optimized for the disjoint check.
    # However, building the shifted set is the cost.
    # Alternative: 
    # for s in sums:
    #     if (s + inv) in sums:
    #         return False
    # return True
    
    # Benchmarking suggests for N>1000, set comp is better if no collision.
    # But we stop on collision.
    # Let's stick to the efficient set method used in the seed, 
    # but optimize the loop structure in the main search.
    shifted = {s + inv for s in sums}
    return sums.isdisjoint(shifted)

def _add_to_set(chosen, sums, x):
    """Add element x to chosen and update sums. Assumes x is valid."""
    inv = _INV[x]
    shifted = {s + inv for s in sums}
    new_sums = sums | shifted
    chosen.add(x)
    return chosen, new_sums

def _remove_from_set(chosen, sums, x):
    """Remove element x from chosen and rebuild sums. 
    Note: Rebuilding is O(2^(k-1)) * k. 
    For k=16, 2^15 * 15 ~ 500k ops. Acceptable for local search steps."""
    chosen.discard(x)
    new_sums = {0}
    for y in chosen:
        inv = _INV[y]
        shifted = {s + inv for s in new_sums}
        new_sums = new_sums | shifted
    return chosen, new_sums

def _greedy_build(order, deadline):
    """Greedily build a set based on the given order."""
    chosen = set()
    sums = {0}
    for x in order:
        if x in chosen:
            continue
        inv = _INV[x]
        if _can_add(sums, inv):
            chosen, sums = _add_to_set(chosen, sums, x)
        if time.time() > deadline:
            break
    return chosen, sums

def _local_search_with_swaps(chosen, sums, deadline, allow_neutral=True):
    """
    Local search that tries to increase size.
    If allow_neutral is True, it also tries 1-for-1 swaps that maintain size
    to escape local optima, with a limit on neutral moves to prevent cycling.
    """
    current_size = len(chosen)
    best_size = current_size
    best_chosen = set(chosen)
    best_sums = set(sums)
    
    neutral_moves_limit = 50
    neutral_count = 0
    no_improve_count = 0
    
    remaining = [x for x in _ALL if x not in chosen]
    
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        if remaining:
            random.shuffle(remaining)
            for x in remaining:
                if x in chosen:
                    continue
                inv = _INV[x]
                if _can_add(sums, inv):
                    chosen, sums = _add_to_set(chosen, sums, x)
                    remaining.remove(x)
                    improved = True
                    current_size = len(chosen)
                    if current_size > best_size:
                        best_size = current_size
                        best_chosen = set(chosen)
                        best_sums = set(sums)
                        neutral_count = 0
                        no_improve_count = 0
                    break
            if improved:
                continue
        
        # 2. If no addition possible, try swaps
        # Try 1-for-1 swaps
        candidates_in = list(chosen)
        candidates_out = remaining
        if not candidates_out:
            break
        
        random.shuffle(candidates_in)
        random.shuffle(candidates_out)
        
        swap_found = False
        for rem in candidates_in:
            if time.time() > deadline:
                break
            # Try removing rem
            temp_chosen, temp_sums = _remove_from_set(set(chosen), set(sums), rem)
            
            # Try adding any from candidates_out
            for add in candidates_out:
                if add in temp_chosen:
                    continue
                inv = _INV[add]
                if _can_add(temp_sums, inv):
                    # Valid swap
                    temp_chosen, temp_sums = _add_to_set(temp_chosen, temp_sums, add)
                    
                    new_size = len(temp_chosen)
                    if new_size > current_size:
                        # Improvement
                        chosen = temp_chosen
                        sums = temp_sums
                        current_size = new_size
                        remaining = [x for x in _ALL if x not in chosen]
                        improved = True
                        neutral_count = 0
                        no_improve_count = 0
                        if current_size > best_size:
                            best_size = current_size
                            best_chosen = set(chosen)
                            best_sums = set(sums)
                        swap_found = True
                        break
                    elif new_size == current_size and allow_neutral:
                        # Neutral move
                        if neutral_count < neutral_moves_limit:
                            chosen = temp_chosen
                            sums = temp_sums
                            remaining = [x for x in _ALL if x not in chosen]
                            neutral_count += 1
                            no_improve_count += 1
                            swap_found = True
                            break
                        else:
                            # Limit reached, try another rem
                            continue
            if swap_found:
                break
        
        if not improved and not swap_found:
            no_improve_count += 1
            # If we've done many neutral moves without improvement, 
            # we might be stuck. Break or randomize harder?
            if no_improve_count > 20:
                break
        
        if not swap_found:
             # If we didn't even find a valid swap structure, we are stuck
             # unless we try 2-for-2, which is expensive.
             # For now, break to try a new random start.
             break

    return best_chosen, best_sums

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin
    
    best_size = 0
    best_set = set()
    
    # Warm starts with different orderings
    orders = [
        _ALL[:],
        sorted(_ALL, reverse=True),
        sorted(_ALL, key=lambda x: _INV[x]),
        sorted(_ALL, key=lambda x: _INV[x], reverse=True),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        chosen, sums = _greedy_build(order, deadline)
        # Local search on this
        chosen, sums = _local_search_with_swaps(chosen, sums, min(deadline, time.time() + 5))
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
            
    # Randomized restarts
    while time.time() < deadline - 5:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_build(order, deadline - 5)
        
        # Aggressive local search
        # Give it a specific time slice
        slice_deadline = min(deadline - 2, time.time() + 3)
        chosen, sums = _local_search_with_swaps(chosen, sums, slice_deadline)
        
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)
            
        # If we hit 17 or higher, we might want to spend more time polishing
        if best_size >= 17:
             # Try to polish the best set found so far
             if time.time() < deadline - 1:
                 # Rebuild sums for best_set
                 b_sums = {0}
                 for x in best_set:
                     inv = _INV[x]
                     shifted = {s + inv for s in b_sums}
                     b_sums = b_sums | shifted
                 
                 # Try to add more to the best set
                 # Use a tighter deadline for this polish
                 polish_deadline = min(deadline, time.time() + 2)
                 # Note: _local_search_with_swaps modifies the passed sets
                 # We pass copies
                 b_chosen, b_sums = _local_search_with_swaps(set(best_set), set(b_sums), polish_deadline)
                 if len(b_chosen) > best_size:
                     best_size = len(b_chosen)
                     best_set = set(b_chosen)

    if not best_set:
        best_set = {1}
        
    return sorted(best_set)

# EVOLVE-BLOCK-END
