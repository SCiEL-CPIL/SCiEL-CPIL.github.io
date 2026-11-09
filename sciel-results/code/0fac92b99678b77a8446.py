import random

def construct():
    n = 512
    numbers = list(range(1, n + 1))
    
    def can_add(S, x):
        # Check if adding x to S creates a 4-AP
        # 4-AP: a, a+d, a+2d, a+3d
        # x is one of the terms.
        
        # Case 1: x = a
        # Need x+d, x+2d, x+3d in S
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if (x + d in S) and (x + 2*d in S) and (x + 3*d in S):
                return False
        
        # Case 2: x = a+d
        # a = x-d, a+2d = x+d, a+3d = x+2d
        # Need x-d, x+d, x+2d in S
        # Constraints: x-d >= 1 => d <= x-1
        #             x+2d <= n => d <= (n-x)//2
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            if (x - d in S) and (x + d in S) and (x + 2*d in S):
                return False
        
        # Case 3: x = a+2d
        # a = x-2d, a+d = x-d, a+3d = x+d
        # Need x-2d, x-d, x+d in S
        # Constraints: x-2d >= 1 => d <= (x-1)//2
        #             x+d <= n => d <= n-x
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            if (x - 2*d in S) and (x - d in S) and (x + d in S):
                return False
        
        # Case 4: x = a+3d
        # a = x-3d, a+d = x-2d, a+2d = x-d
        # Need x-3d, x-2d, x-d in S
        # Constraints: x-3d >= 1 => d <= (x-1)//3
        max_d = (x - 1) // 3
        for d in range(1, max_d + 1):
            if (x - 3*d in S) and (x - 2*d in S) and (x - d in S):
                return False
        
        return True

    best_set = set()
    best_size = 0
    
    # Run multiple randomized greedy attempts
    # 512 elements, can_add is ~O(500) ops. 512*500 = 250k ops per run.
    # 1000 runs = 250M ops. Might be tight in Python in 60s.
    # Let's optimize can_add or reduce runs.
    # Actually, the loops in can_add are small on average.
    # Let's try 500 runs first.
    
    for _ in range(500):
        S = set()
        # Random permutation
        perm = numbers.copy()
        random.shuffle(perm)
        
        for x in perm:
            if can_add(S, x):
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return sorted(best_set)
