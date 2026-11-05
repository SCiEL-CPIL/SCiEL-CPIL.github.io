import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin for safety
    
    best_set = set()
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            if is_valid(S):
                best_set = set(S)
                best_size = len(S)
                
    def get_bitset(S):
        reachable = 1
        for x in S:
            reachable |= (reachable << x)
        return reachable

    def is_valid(S):
        reachable = get_bitset(S)
        return ((reachable >> 200) & 1) == 0

    def improve_additions(S):
        # Greedily add elements if possible
        S_curr = set(S)
        base_reach = get_bitset(S_curr)
        
        changed = True
        while changed:
            changed = False
            missing = [x for x in range(1, 101) if x not in S_curr]
            random.shuffle(missing)
            
            for x in missing:
                new_reach = base_reach | (base_reach << x)
                if ((new_reach >> 200) & 1) == 0:
                    S_curr.add(x)
                    base_reach = new_reach
                    changed = True
        return S_curr

    def local_search_with_swaps(S):
        S_curr = set(S)
        # First, try to add
        S_curr = improve_additions(S_curr)
        
        # Now try swaps to potentially unlock additions
        # We limit the number of swap attempts to avoid getting stuck in infinite loops of non-improving swaps
        max_swap_attempts = 5000
        attempts = 0
        
        while attempts < max_swap_attempts and time.time() < deadline:
            attempts += 1
            # Try to add again (maybe a swap opened a door)
            S_curr = improve_additions(S_curr)
            
            # Check if we can add anything
            base_reach = get_bitset(S_curr)
            can_add = False
            for x in range(1, 101):
                if x not in S_curr:
                    new_reach = base_reach | (base_reach << x)
                    if ((new_reach >> 200) & 1) == 0:
                        can_add = True
                        break
            
            if can_add:
                # improve_additions already handled this in the loop above?
                # Wait, improve_additions runs to completion. So if we are here, no single addition works.
                # So we must swap.
                pass
            
            # Pick a random element to remove and a random element to add
            if not S_curr:
                break
                
            x_to_remove = random.choice(list(S_curr))
            candidates_to_add = [x for x in range(1, 101) if x not in S_curr]
            if not candidates_to_add:
                break
                
            y_to_add = random.choice(candidates_to_add)
            
            # Try the swap
            new_S = S_curr - {x_to_remove} | {y_to_add}
            
            if is_valid(new_S):
                # Accept the swap. It doesn't increase size, but might change reachability.
                S_curr = new_S
                
        # Final polish: try to add again
        S_curr = improve_additions(S_curr)
        return S_curr

    # 1. Deterministic Greedies
    # Large to Small
    s = []
    reachable = 1
    for x in range(100, 0, -1):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # Small to Large
    s = []
    reachable = 1
    for x in range(1, 101):
        new_reach = reachable | (reachable << x)
        if not ((new_reach >> 200) & 1):
            s.append(x)
            reachable = new_reach
    update_best(set(s))
    
    # 2. Randomized Local Search
    # Run multiple trials with different random seeds/permutations
    while time.time() < deadline:
        # Generate a random greedy seed
        elements = list(range(1, 101))
        random.shuffle(elements)
        
        S = []
        reachable = 1
        for x in elements:
            new_reach = reachable | (reachable << x)
            if not ((new_reach >> 200) & 1):
                S.append(x)
                reachable = new_reach
        
        S_set = set(S)
        
        # Run local search with swaps
        S_improved = local_search_with_swaps(S_set)
        update_best(S_improved)
        
    return list(best_set)
