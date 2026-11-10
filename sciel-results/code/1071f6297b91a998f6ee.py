"""Combined program for the no-divide-sum-of-three-others task."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    n = 192
    all_elements = list(range(1, n + 1))
    
    def is_valid_set(S):
        """
        Checks if the set S is valid.
        Condition: No element x in S divides the sum of any 3 distinct other elements in S.
        """
        S_set = set(S)
        for x in S:
            # Check if x divides sum of any 3 distinct others
            # Iterate pairs (a, b) from S \ {x}
            others = [y for y in S if y != x]
            if len(others) < 3:
                continue
                
            # We need to check if there exist distinct a, b, c in 'others' such that (a+b+c) % x == 0
            # Optimization: For each pair (a, b), check if the required remainder exists in others \ {a, b}
            # rem = (-a - b) % x
            # We need c in others such that c % x == rem and c != a and c != b
            
            # To speed this up, we can use a set of remainders or just check membership.
            # Since len(others) <= 47, a double loop is ~1100 iterations.
            # Inside, we check if any c works.
            # We can precompute a list of (value, remainder) for others?
            
            # Let's just do the double loop and check membership in a set of available values?
            # No, we need c such that c % x == rem.
            # We can build a dictionary: remainder -> list of values in others.
            rem_to_vals = {}
            for v in others:
                r = v % x
                if r not in rem_to_vals:
                    rem_to_vals[r] = []
                rem_to_vals[r].append(v)
            
            for i in range(len(others)):
                a = others[i]
                for j in range(i + 1, len(others)):
                    b = others[j]
                    target_rem = (-a - b) % x
                    if target_rem in rem_to_vals:
                        candidates = rem_to_vals[target_rem]
                        # Check if any candidate c is distinct from a and b
                        for c in candidates:
                            if c != a and c != b:
                                return False
        return True

    def greedy_construct(order):
        S = []
        for x in order:
            S_new = S + [x]
            if is_valid_set(S_new):
                S = S_new
        return S

    best_set = []
    best_size = 0
    
    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]

    # 1. Greedy strategies
    # Descending
    S1 = greedy_construct(sorted(all_elements, reverse=True))
    update_best(S1)
    
    # Ascending
    S2 = greedy_construct(sorted(all_elements))
    update_best(S2)
    
    # Random Greedies
    for _ in range(100):
        if time.time() > deadline:
            break
        order = all_elements[:]
        random.shuffle(order)
        S3 = greedy_construct(order)
        update_best(S3)

    # 2. Local Search (Swap-based)
    if best_set:
        current_set = best_set[:]
        current_set_set = set(current_set)
        unselected = set(all_elements) - current_set_set
        
        # To make local search efficient, we don't re-check the whole set every time.
        # But for N=48, is_valid_set is fast enough (~1000-5000 ops).
        # Let's do a simple hill climbing.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try to add any unselected element
            for cand in list(unselected):
                if time.time() > deadline:
                    break
                new_set = current_set + [cand]
                if is_valid_set(new_set):
                    current_set = new_set
                    current_set_set.add(cand)
                    unselected.remove(cand)
                    update_best(current_set)
                    improved = True
                    break # Restart loop to check for more additions
            
            if not improved and unselected and current_set:
                # Try swaps
                # Pick a random element to remove and a random element to add
                # To be more systematic, we can iterate, but random is faster to escape local minima
                for _ in range(1000): # Limit iterations per phase
                    if time.time() > deadline:
                        break
                    
                    if not current_set or not unselected:
                        break
                    
                    idx = random.randint(0, len(current_set) - 1)
                    remove_elem = current_set[idx]
                    add_elem = random.choice(list(unselected))
                    
                    # Create new set
                    new_set = [x for i, x in enumerate(current_set) if i != idx] + [add_elem]
                    
                    if is_valid_set(new_set):
                        # Accept swap
                        current_set = new_set
                        current_set_set.remove(remove_elem)
                        current_set_set.add(add_elem)
                        unselected.remove(add_elem)
                        unselected.add(remove_elem)
                        update_best(current_set)
                        improved = True
                        break # Restart add/swap loop

    return best_set
# EVOLVE-BLOCK-END
