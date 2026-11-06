# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # Boolean array for O(1) membership
    in_set = [False] * (n + 1)
    S = set()
    
    # Helper to check if x can be added to current set (represented by in_set)
    # without forming a 3-AP.
    # Assumes x is NOT in set.
    def can_add(x):
        # Check all a in [1, n]
        # If a is in set, check if it forms AP with x and some b in set
        for a in range(1, n + 1):
            if not in_set[a]:
                continue
            
            # Case 1: a, b, x (x is largest) -> 2b = a + x -> b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if b > a and b < x and in_set[b]:
                    return False
            
            # Case 2: x, a, b (x is smallest) -> 2a = x + b -> b = 2a - x
            b = 2 * a - x
            if b > a and b <= n and in_set[b]:
                return False
            
            # Case 3: a, x, b (x is middle) -> 2x = a + b -> b = 2x - a
            b = 2 * x - a
            if b > x and b <= n and in_set[b]:
                return False
        
        return True

    # Greedy Phase: Add smallest numbers first to maximize 1/x
    for x in range(1, n + 1):
        if can_add(x):
            in_set[x] = True
            S.add(x)
    
    # Local Search Phase
    # We try to swap a large y in S with a small x not in S
    # to increase the sum of reciprocals.
    
    # Get lists for iteration
    # Candidates: numbers not in S, sorted ascending (smallest first, highest 1/x)
    # Elements: numbers in S, sorted descending (largest first, lowest 1/x)
    
    improved = True
    while improved:
        improved = False
        candidates = sorted([x for x in range(1, n + 1) if not in_set[x]])
        elements = sorted(list(S), reverse=True)
        
        for x in candidates:
            # We want to find a y in S such that y > x (so 1/x > 1/y)
            # and S \ {y} U {x} is valid.
            for y in elements:
                if y <= x:
                    break # Further y are smaller, so 1/y >= 1/x, no gain
                
                # Check if swapping y for x is valid
                # Temporarily remove y, add x
                in_set[y] = False
                S.remove(y)
                
                if can_add(x):
                    # Valid swap
                    in_set[x] = True
                    S.add(x)
                    improved = True
                    break # Restart local search
                else:
                    # Revert
                    in_set[y] = True
                    S.add(y)
            
            if improved:
                break
    
    return list(S)
# EVOLVE-BLOCK-END
