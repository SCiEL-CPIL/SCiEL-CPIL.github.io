"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX = 48
_START_TIME = time.time()
_DEADLINE = _START_TIME + 85.0  # Safety margin before 90s kill

def _time_left():
    return _DEADLINE - time.time()

def _valid_set(S):
    """Check if set S is valid: no element divides sum of any 3 distinct others."""
    S_list = list(S)
    n = len(S_list)
    for i, a in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _is_valid_addition(S, x):
    """Check if adding x to valid set S keeps it valid."""
    S_list = list(S)
    n = len(S_list)
    
    # Check 1: x does not divide sum of any 3 in S
    for t in combinations(S_list, 3):
        if sum(t) % x == 0:
            return False
    
    # Check 2: for each a in S, a does not divide sum of x + 2 others from S
    for i, a in enumerate(S_list):
        others = [S_list[j] for j in range(n) if j != i]
        for t in combinations(others, 2):
            if (x + t[0] + t[1]) % a == 0:
                return False
            
    return True

def _greedy_construct(order):
    """Greedy construction based on given order."""
    selected = []
    for x in order:
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def _local_search(S, time_budget):
    """
    Local search to maximize size of S.
    Uses random insertions and valid swaps.
    """
    S = list(S)
    start = time.time()
    
    while time.time() - start < time_budget and _time_left() > 1.0:
        improved = False
        
        # Phase 1: Try to insert any valid element
        # Try candidates in random order, stop at first valid
        candidates = [x for x in range(1, _N + 1) if x not in S]
        if candidates:
            random.shuffle(candidates)
            for x in candidates:
                if _is_valid_addition(S, x):
                    S.append(x)
                    improved = True
                    break
        
        if improved:
            continue
            
        # Phase 2: If no insertion, try a valid swap to change structure
        # Pick random element to remove and random candidate to add
        if S:
            a_remove = random.choice(S)
            candidates = [x for x in range(1, _N + 1) if x not in S]
            if candidates:
                x_add = random.choice(candidates)
                # Check if swap is valid
                S_temp = [x for x in S if x != a_remove]
                if _is_valid_addition(S_temp, x_add):
                    S = S_temp
                    S.append(x_add)
                    # Do not mark as improved (size didn't change)
                    # But we changed the set, so we loop back to try insertion again
                    continue
                    # If we don't continue, we might do multiple swaps in a row.
                    # It's better to try an insertion after every swap.
        
        # If we reached here, no insertion was possible and no valid swap was found (or S was empty)
        # To avoid infinite loop of failed swaps, we break
        # However, the swap check above only breaks if valid.
        # If no valid swap found, we should break out of the while loop to save time
        # Or try a few more swaps?
        # Let's just break if we couldn't insert and didn't perform a swap in this iteration.
        # The logic above: if valid swap, we 'continue'. 
        # If no valid swap, we fall through to the end.
        break

    return S

def construct():
    """
    Main entry point.
    Uses Iterated Local Search (ILS).
    """
    best_set = []
    
    # Initial Greedy
    S1 = _greedy_construct(range(_N, 0, -1))
    S2 = _greedy_construct(range(1, _N + 1))
    best_set = max([S1, S2], key=len)
    
    # ILS
    current_set = list(best_set)
    no_improve_count = 0
    max_no_improve = 50
    
    while _time_left() > 2.0:
        # Perturb current_set
        # Remove 2-4 random elements
        k = random.randint(2, 4)
        if len(current_set) > k:
            to_remove = random.sample(current_set, k)
            perturbed = [x for x in current_set if x not in to_remove]
        else:
            perturbed = list(current_set)
            
        # Local Search on perturbed
        # Give it a small time budget
        time_budget = min(1.0, _time_left() - 1.0)
        if time_budget < 0.1:
            break
            
        new_set = _local_search(perturbed, time_budget)
        
        if len(new_set) > len(best_set):
            best_set = new_set
            current_set = new_set
            no_improve_count = 0
        else:
            # If new_set is better than current_set (in size), update current_set
            # This helps explore the neighborhood of the new local optimum
            if len(new_set) > len(current_set):
                current_set = new_set
            no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                # Restart from best_set
                current_set = list(best_set)
                no_improve_count = 0
                
    return best_set
# EVOLVE-BLOCK-END
