import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    time_limit = 55.0
    
    def is_valid_state(state):
        """
        Check if the entire state is valid.
        state: list of ints
        """
        n = len(state)
        for i in range(n):
            a = state[i]
            # Check if any 3 distinct others sum to multiple of a
            # Others are state without a
            # We can just check all 3-combinations of the other n-1 elements
            others = [state[j] for j in range(n) if j != i]
            for u, v, w in combinations(others, 3):
                if (u + v + w) % a == 0:
                    return False
        return True

    def try_add(state, x):
        """
        Try to add x to state. Returns new state if valid, else None.
        Optimized check:
        1. Check if x divides any 3-sum of state.
        2. Check if any a in state divides (x + u + v) for u,v in state, u!=v, u!=a, v!=a.
        """
        # Check 1: x is divisor
        # Sums of 3 distinct from state
        # If len(state) < 3, no 3-sums exist, so check passes.
        if len(state) >= 3:
            for u, v, w in combinations(state, 3):
                if (u + v + w) % x == 0:
                    return None
        
        # Check 2: a in state is divisor, sum includes x
        # Sum = x + u + v
        # u, v distinct, from state, u!=v.
        # Also u!=a, v!=a is required? 
        # "3 DISTINCT OTHER selected integers".
        # If a is the divisor, the summands must be distinct from a.
        # So u != a and v != a.
        
        # We iterate a in state.
        for a in state:
            # We need to check if there exist u, v in state such that:
            # 1. u != v
            # 2. u != a
            # 3. v != a
            # 4. (x + u + v) % a == 0
            
            # Optimization:
            # For a given a, we can check all pairs (u,v) in state.
            # This is O(n^2) per a. Total O(n^3).
            
            # To make it faster, we can precompute a set of state.
            # But we need distinct u,v.
            
            # Let's just iterate. n is small.
            for i in range(len(state)):
                u = state[i]
                if u == a: continue
                for j in range(i+1, len(state)):
                    v = state[j]
                    if v == a: continue
                    if (x + u + v) % a == 0:
                        return None
        return state + [x]

    def try_swap(state, remove_idx, add_x):
        """
        Remove state[remove_idx] and add add_x.
        Returns new state if valid, else None.
        """
        new_state = [state[i] for i in range(len(state)) if i != remove_idx]
        return try_add(new_state, add_x)

    # 1. Greedy Initialization
    current = []
    for x in range(192, 0, -1):
        if try_add(current, x):
            current = try_add(current, x) # Re-evaluate to get new list? 
            # try_add returns new list or None. 
            # If it returns list, we update current.
            # But try_add checks validity against OLD current.
            # So:
            #   new = try_add(current, x)
            #   if new: current = new
            
    # Correct Greedy
    current = []
    for x in range(192, 0, -1):
        new_state = try_add(current, x)
        if new_state:
            current = new_state

    best = list(current)
    best_size = len(current)
    
    # 2. Local Search
    # We want to increase size.
    # Strategy:
    # - If size < 48, try to add a random element.
    # - If no add possible, try a swap.
    # - If size == 48, we are done (max possible).
    
    random.seed(42)
    
    while len(current) < 48:
        if time.time() - start_time > time_limit:
            break
            
        # Try to add
        added = False
        candidates = [x for x in range(1, 193) if x not in current]
        random.shuffle(candidates)
        
        for x in candidates:
            new_state = try_add(current, x)
            if new_state:
                current = new_state
                added = True
                if len(current) > best_size:
                    best = list(current)
                    best_size = len(current)
                break # Restart add search with new state
        
        if not added:
            # Try swap
            # Remove a random element, try to add a random candidate
            if not current:
                break
            removed_idx = random.randint(0, len(current) - 1)
            removed_val = current[removed_idx]
            
            # Try to add any candidate that works with the reduced set
            base = [current[i] for i in range(len(current)) if i != removed_idx]
            
            found_swap = False
            for x in candidates: # Reuse shuffled candidates or new?
                if x == removed_val:
                    continue
                new_state = try_add(base, x)
                if new_state:
                    current = new_state
                    found_swap = True
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
                    break
            
            if not found_swap:
                # Stuck? Try random removals to escape?
                # Or just break if stuck for a while?
                # For now, just break to avoid infinite loop if no moves.
                # But we might be stuck in a local optimum.
                # Let's try a few random swaps even if they don't increase size?
                # No, we only want to increase size or maintain.
                # If we can't add or swap to same size, we are stuck.
                # Since we want MAX size, and we are < 48, we should try harder.
                # Let's try all swaps?
                # That might be slow.
                # For now, break.
                break

    return best
