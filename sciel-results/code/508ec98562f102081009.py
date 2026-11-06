"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

TARGET = 200
N = 100

def _bitset_update(reachable_bits, x):
    """Update reachable sums (as a bitset) after adding x. Returns new bitset."""
    # Shift left by x, mask to keep only bits up to TARGET
    new_bits = (reachable_bits << x) & ((1 << (TARGET + 1)) - 1)
    return reachable_bits | new_bits

def _is_feasible(reachable_bits, x):
    """Check if adding x to the set keeps no subset summing to TARGET."""
    # If bit (TARGET - x) is set in reachable_bits, then adding x would create sum TARGET
    if x > TARGET:
        return True
    target_bit = 1 << (TARGET - x)
    return (reachable_bits & target_bit) == 0

def _greedy_desc():
    """Greedy: add elements from largest to smallest if feasible using bitset."""
    selected = []
    reachable_bits = 1  # bit 0 is set (sum 0 is reachable)
    for x in range(N, 0, -1):
        if _is_feasible(reachable_bits, x):
            selected.append(x)
            reachable_bits = _bitset_update(reachable_bits, x)
    return selected

def _greedy_asc():
    """Greedy: add elements from smallest to largest if feasible using bitset."""
    selected = []
    reachable_bits = 1  # bit 0 is set (sum 0 is reachable)
    for x in range(1, N + 1):
        if _is_feasible(reachable_bits, x):
            selected.append(x)
            reachable_bits = _bitset_update(reachable_bits, x)
    return selected

def _greedy_random(seed=0):
    """Greedy with random ordering using bitset."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    reachable_bits = 1
    for x in order:
        if _is_feasible(reachable_bits, x):
            selected.append(x)
            reachable_bits = _bitset_update(reachable_bits, x)
    return selected

def _local_search_improve(initial):
    """Try to improve by adding elements or swapping using bitset."""
    selected = list(initial)
    reachable_bits = 1
    for x in selected:
        reachable_bits = _bitset_update(reachable_bits, x)
    
    selected_set = set(selected)
    improved = True
    while improved:
        improved = False
        # Try adding any element not in selected
        for x in range(1, N + 1):
            if x not in selected_set:
                if _is_feasible(reachable_bits, x):
                    selected.append(x)
                    selected_set.add(x)
                    reachable_bits = _bitset_update(reachable_bits, x)
                    improved = True
                    break
        # If no additions, try swapping: remove one, add another
        if not improved:
            for i, x in enumerate(selected):
                # Remove x
                new_selected = selected[:i] + selected[i+1:]
                new_selected_set = set(new_selected)
                new_reachable_bits = 1
                for y in new_selected:
                    new_reachable_bits = _bitset_update(new_reachable_bits, y)
                # Try to add a different element
                for z in range(1, N + 1):
                    if z not in new_selected_set:
                        if _is_feasible(new_reachable_bits, z):
                            # Only accept if z < x (smaller elements allow more room)
                            if z < x:
                                selected = new_selected + [z]
                                selected_set = new_selected_set | {z}
                                reachable_bits = _bitset_update(new_reachable_bits, z)
                                improved = True
                                break
                if improved:
                    break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Try multiple strategies
    best = []
    
    # Strategy 1: Greedy largest first (desc)
    sol1 = _greedy_desc()
    if len(sol1) > len(best):
        best = sol1
    
    # Strategy 2: Greedy smallest first (asc)
    if time.time() < deadline:
        sol2 = _greedy_asc()
        if len(sol2) > len(best):
            best = sol2
    
    # Strategy 3: Random orders
    if time.time() < deadline:
        for seed in range(100):
            if time.time() >= deadline:
                break
            sol = _greedy_random(seed)
            if len(sol) > len(best):
                best = sol
    
    # Local search improvement on best found
    if time.time() < deadline:
        improved = _local_search_improve(best)
        if len(improved) > len(best):
            best = improved
    
    # Verify and return
    if not best:
        return [1]
    
    # Final verification using bitset
    reachable_bits = 1
    for x in best:
        reachable_bits = _bitset_update(reachable_bits, x)
    if (reachable_bits >> TARGET) & 1:
        # Shouldn't happen, but fallback
        return _greedy_desc()
    
    return best
# EVOLVE-BLOCK-END
