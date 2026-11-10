import time
import random

def construct():
    """
    Strategy: Greedy Construction + Local Search with Heuristic Swaps.
    1. Start with a greedy descending set.
    2. Attempt to add any missing element.
    3. If stuck, remove an element (heuristic: random or smallest) and try to add others.
    4. Use multiple restarts to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 256
    best_set = set()
    best_size = 0
    
    # Helper to check if a specific element can be added to a set
    def can_add(x, S):
        # S is a set. We convert to sorted list for efficiency in loops
        # But to avoid repeated conversion, we can pass a sorted list if we maintain one.
        # For now, keep it simple, but optimize the inner loops.
        
        S_list = sorted(S)
        
        # Case 1: x is 'a' (smallest)
        # Need b, c in S such that x < b <= c and x | (b+c)
        # b > x, c >= b
        for b in S_list:
            if b <= x:
                continue
            target_rem = (-b) % x
            # Find c in S such that c >= b and c % x == target_rem
            for c in S_list:
                if c < b:
                    continue
                if (c % x) == target_rem:
                    return False
        
        # Case 2: x is 'b' (middle)
        # a < x <= c. a | (x+c)
        # a in S, a < x. c in S, c >= x.
        for a in S_list:
            if a >= x:
                break # S_list is sorted
            target_rem = (-x) % a
            for c in S_list:
                if c < x:
                    continue
                if (c % a) == target_rem:
                    return False
        
        # Case 3: x is 'c' (largest)
        # a < b <= x. a | (b+x)
        # a, b in S. a < b < x.
        for a in S_list:
            if a >= x:
                break
            target_rem = (-x) % a
            for b in S_list:
                if b <= a:
                    continue
                if b >= x:
                    break # b < x required
                if (b % a) == target_rem:
                    return False
        
        return True

    # Strategy 1: Greedy Descending
    S_greedy = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S_greedy):
            S_greedy.add(x)
    
    if len(S_greedy) > best_size:
        best_size = len(S_greedy)
        best_set = S_greedy.copy()

    # Strategy 2: Local Search with Restarts
    def local_search(S):
        S = S.copy()
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            
            # 1. Try to add any element
            candidates = [x for x in range(1, n+1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    return S
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    break # Restart loop to try more additions
            
            # 2. If no additions, try a swap/remove heuristic
            if not improved:
                # Heuristic: Remove an element to make space
                # Prioritize removing smaller elements (they constrain more sums)
                # or random elements
                S_list = list(S)
                # Sort by value to try removing small ones first? 
                # Or random? Let's try a mix.
                random.shuffle(S_list)
                
                for e in S_list:
                    if time.time() > deadline:
                        return S
                    S_temp = S.copy()
                    S_temp.remove(e)
                    
                    # Try to add elements to S_temp
                    candidates = [x for x in range(1, n+1) if x not in S_temp]
                    random.shuffle(candidates)
                    added_any = False
                    for x in candidates:
                        if time.time() > deadline:
                            break
                        if can_add(x, S_temp):
                            S_temp.add(x)
                            added_any = True
                    
                    if len(S_temp) > len(S):
                        S = S_temp
                        improved = True
                        break # Restart loop with new S
        
        return S

    # Run multiple iterations
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        # Start from current best
        current = local_search(best_set)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
        
        # Occasionally restart from scratch or random seed
        if iterations % 5 == 0:
            # Generate a random valid seed
            S_rand = set()
            for _ in range(50):
                if time.time() > deadline - 2:
                    break
                x = random.randint(1, n)
                if x not in S_rand and can_add(x, S_rand):
                    S_rand.add(x)
            
            current = local_search(S_rand)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current

    return sorted(best_set)
