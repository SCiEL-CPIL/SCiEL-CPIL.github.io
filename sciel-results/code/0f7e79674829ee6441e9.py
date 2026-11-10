import time
import random

def construct():
    """
    Strategy: Local Search with Randomized Restarts.
    1. Start with a valid base set (e.g., upper tail or greedy result).
    2. Iteratively try to add any element from [1, 256] not in the set.
    3. If no additions possible, try swapping an element in the set with one outside
       to see if it allows more additions later (simple hill climbing).
    4. Use multiple random restarts with different initial random subsets to explore
       different local optima within the time budget.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = 256
    best_set = set()
    best_size = 0
    
    # Helper to check if a specific element can be added to a set
    def can_add(x, S):
        # For every a < b <= c in S U {x}, a must NOT divide b+c.
        # We only need to check constraints involving x.
        # Cases:
        # 1. x is 'a': x < b <= c, x divides b+c. 
        #    Here b,c are from S, and b > x, c >= b.
        # 2. x is 'b': a < x <= c, a divides x+c.
        #    Here a is from S, a < x. c is from S, c >= x.
        # 3. x is 'c': a < b <= x, a divides b+x.
        #    Here a,b are from S, b <= x.
        
        S_list = list(S)
        
        # Case 1: x is the smallest (a)
        # Need b, c in S such that x < b <= c and x | (b+c)
        # Since x is added, we check pairs in S.
        # Optimization: b+c must be a multiple of x.
        # b > x, c >= b.
        # Max b+c is 256+256=512. Min b+c is (x+1) + (x+1) = 2x+2.
        # Multiples of x in range: x*k.
        # x*k >= 2x+2 => k >= 2 + 2/x. So k starts at 3 if x>2, 2 if x<=2?
        # Actually, b > x, so b >= x+1. c >= b.
        # Smallest sum is (x+1) + (x+1) = 2x+2.
        # Largest sum is 256 + 256 = 512.
        # So we check multiples of x in [2x+2, 512].
        # For each multiple m = k*x, check if there exist b,c in S 
        # such that b+c=m, x < b <= c.
        
        # To speed up, we can precompute sums? No, S changes.
        # Let's just iterate.
        # Since n=256 is small, O(|S|^2) per check is fine if |S| is ~100-150.
        # 150^2 = 22500 ops. 256 candidates * 22500 = ~5.7M ops per pass.
        # With time limit, we can do many passes.
        
        # More efficient check for Case 1:
        # For each b in S where b > x:
        #   We need c in S such that c >= b and (b+c) % x == 0.
        #   This implies c % x == (-b) % x.
        #   So for a fixed b, we look for c in S with c >= b and c % x == (-b) % x.
        
        for b in S_list:
            if b <= x:
                continue
            target_rem = (-b) % x
            # Find c in S such that c >= b and c % x == target_rem
            # We can just iterate S again.
            for c in S_list:
                if c >= b and (c % x) == target_rem:
                    return False
        
        # Case 2: x is 'b' (middle)
        # a < x <= c. a divides x+c.
        # a in S, a < x.
        # c in S, c >= x.
        for a in S_list:
            if a >= x:
                continue
            # We need c in S, c >= x, such that (x+c) % a == 0.
            # c % a == (-x) % a.
            target_rem = (-x) % a
            for c in S_list:
                if c >= x and (c % a) == target_rem:
                    return False
        
        # Case 3: x is 'c' (largest)
        # a < b <= x. a divides b+x.
        # a, b in S. b <= x.
        # Note: b can be equal to x? No, x is new, so b < x strictly?
        # Condition: a < b <= c. If c=x, then b <= x.
        # Since x is not in S yet, b must be in S and b <= x.
        # If b == x, it's not in S, so b < x.
        # So a, b in S, a < b < x.
        # a divides b+x.
        # (b+x) % a == 0 => b % a == (-x) % a.
        for a in S_list:
            if a >= x: # a < b < x, so a < x
                continue
            target_rem = (-x) % a
            for b in S_list:
                if b <= x and a < b:
                    if (b % a) == target_rem:
                        return False
        
        return True

    # Helper to check if a set is valid (for initial seeds or verification)
    def is_valid(S):
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j, b in enumerate(S_list):
                if b <= a:
                    continue
                for k, c in enumerate(S_list):
                    if c < b:
                        continue
                    if a % (b+c) == 0 or (b+c) % a == 0: # Wait, condition is a divides b+c
                        if (b+c) % a == 0:
                            return False
        return True

    # Strategy 1: Greedy Descending (fast baseline)
    S_greedy = set()
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if can_add(x, S_greedy):
            S_greedy.add(x)
    
    if len(S_greedy) > best_size:
        best_size = len(S_greedy)
        best_set = S_greedy.copy()

    # Strategy 2: Local Search / Hill Climbing with Restarts
    # We will try to improve the greedy set or start from random valid sets.
    
    def local_search(S):
        S = S.copy()
        improved = True
        while improved:
            if time.time() > deadline:
                return S
            improved = False
            # Try adding any element not in S
            candidates = [x for x in range(1, n+1) if x not in S]
            random.shuffle(candidates)
            for x in candidates:
                if time.time() > deadline:
                    return S
                if can_add(x, S):
                    S.add(x)
                    improved = True
                    # Restart loop to try more additions
                    break
        
        # If stuck, try swapping?
        # Simple swap: remove one element, try to add two others?
        # Or just remove one element that is "blocking" many others.
        # Let's try a simple swap heuristic:
        # For each element e in S, try removing it and then greedily adding back
        # from the remaining candidates to see if we get a larger set.
        # This is expensive, so we limit it.
        
        if time.time() < deadline - 5:
            S_list = list(S)
            random.shuffle(S_list)
            for e in S_list:
                if time.time() > deadline:
                    return S
                S_temp = S.copy()
                S_temp.remove(e)
                # Try to add elements
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
                    # Continue local search on this new set? 
                    # To save time, just return this improved set.
                    return S
        return S

    # Run multiple iterations of local search starting from greedy and random seeds
    iterations = 0
    while time.time() < deadline - 2:
        iterations += 1
        # Start from current best
        current = local_search(best_set)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current
        
        # Occasionally start from a random valid seed
        if iterations % 10 == 0:
            # Generate a random small valid set
            S_rand = set()
            # Pick random elements, ensuring validity
            for _ in range(20):
                x = random.randint(1, n)
                if x not in S_rand and can_add(x, S_rand):
                    S_rand.add(x)
            current = local_search(S_rand)
            if len(current) > best_size:
                best_size = len(current)
                best_set = current

    return sorted(best_set)

# Ensure the function is named construct and returns a list
# The code above defines construct() and returns sorted(best_set) which is a list.
