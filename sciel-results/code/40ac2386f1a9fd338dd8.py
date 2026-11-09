import time
import random
import itertools

_LCM = 232792560

def _get_sums(subset):
    """Calculate all subset sums of the scaled reciprocals."""
    scaled = [_LCM // x for x in subset]
    sums = {0}
    for v in scaled:
        sums |= {s + v for s in sums}
    return sums

def _is_dissociated(subset):
    """Check if the set is dissociated."""
    sums = _get_sums(subset)
    return len(sums) == (1 << len(subset))

def _can_add(subset, x):
    """Check if adding x to subset keeps it dissociated."""
    v = _LCM // x
    # Get current sums
    scaled = [_LCM // y for y in subset]
    sums = {0}
    for val in scaled:
        sums |= {s + val for s in sums}
    
    # Check if any new sum collides with an existing one
    # New sums are {s + v for s in sums}
    # Collision if (s + v) is in sums for any s in sums
    for s in sums:
        if (s + v) in sums:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    universe = list(range(1, 23))
    best_S = []
    best_size = 0
    
    # We will run multiple random restarts
    # Each restart:
    # 1. Randomly shuffle universe
    # 2. Greedy add
    # 3. Local search with swaps
    
    for restart in range(100): # Limit restarts, but check time
        if time.time() > deadline - 5:
            break
            
        random.shuffle(universe)
        S = []
        # Greedy phase
        for x in universe:
            if time.time() > deadline - 5:
                break
            if _can_add(S, x):
                S.append(x)
        
        if len(S) > best_size:
            best_S = list(S)
            best_size = len(S)
            
        # Local Search Phase
        # We try to increase size by adding elements.
        # If we can't, we try to swap elements to change the structure.
        current_S = list(S)
        current_size = len(current_S)
        in_set = set(current_S)
        
        stuck_count = 0
        max_stuck = 500 # If we are stuck for 500 swaps, give up on this restart
        
        while time.time() < deadline - 2:
            # Try to add any element
            improved = False
            not_in_set = [x for x in universe if x not in in_set]
            
            # Shuffle not_in_set to randomize order
            random.shuffle(not_in_set)
            
            for x in not_in_set:
                if time.time() > deadline - 2:
                    break
                if _can_add(current_S, x):
                    current_S.append(x)
                    in_set.add(x)
                    current_size += 1
                    improved = True
                    stuck_count = 0
                    if current_size > best_size:
                        best_S = list(current_S)
                        best_size = current_size
                    break # Restart the loop to try adding more
            
            if improved:
                continue
            
            # If we couldn't add, try a swap
            # Pick a random element to remove and a random element to add
            if not current_S:
                break
                
            a = random.choice(current_S)
            # Try to find a b not in S that can replace a
            # We want S' = S - {a} + {b} to be dissociated
            
            # Optimization: Pre-calculate sums of S - {a}?
            # That's expensive. Just check full dissociation of candidate.
            
            # To make it faster, we can just pick a random b and check.
            # If valid, perform swap.
            # If not, try another b.
            
            # Limit attempts per swap step to avoid wasting time
            found_swap = False
            attempts = 0
            max_attempts = 20
            
            while attempts < max_attempts:
                if time.time() > deadline - 2:
                    break
                b = random.choice(not_in_set)
                if b == a:
                    continue
                candidate = [z for z in current_S if z != a] + [b]
                if _is_dissociated(candidate):
                    # Perform swap
                    current_S = candidate
                    in_set.remove(a)
                    in_set.add(b)
                    found_swap = True
                    stuck_count = 0
                    break
                attempts += 1
                
            if not found_swap:
                stuck_count += 1
                if stuck_count > max_stuck:
                    break
            
    return best_S
