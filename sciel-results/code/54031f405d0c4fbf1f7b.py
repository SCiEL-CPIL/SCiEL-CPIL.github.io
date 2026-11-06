import time
import random

def construct():
    N = 512
    deadline = time.time() + 80  # safety margin
    
    # Helper: check if adding x to set S creates a 4-AP
    def creates_4ap(x, S):
        # x as 1st term: need x+d, x+2d, x+3d in S
        for d in range(1, (N - x) // 3 + 1):
            if (x + d) in S and (x + 2*d) in S and (x + 3*d) in S:
                return True
        # x as 2nd term: need x-d, x+d, x+2d in S
        for d in range(1, min(x, (N - x) // 2) + 1):
            if (x - d) in S and (x + d) in S and (x + 2*d) in S:
                return True
        # x as 3rd term: need x-2d, x-d, x+d in S
        for d in range(1, (x - 1) // 2 + 1):
            if (x - 2*d) in S and (x - d) in S and (x + d) in S:
                return True
        # x as 4th term: need x-3d, x-2d, x-d in S
        for d in range(1, (x - 1) // 3 + 1):
            if (x - 3*d) in S and (x - 2*d) in S and (x - d) in S:
                return True
        return False
    
    # Base-3 no-zero-digit construction
    def base3_no_zero(x):
        if x == 0:
            return True
        while x > 0:
            if x % 3 == 0:
                return False
            x //= 3
        return True
    
    # Initial set: base-3 no-zero digits
    S = set()
    for x in range(1, N + 1):
        if base3_no_zero(x):
            S.add(x)
    
    best_size = len(S)
    best_set = set(S)
    
    # Greedy addition: try to add elements
    def greedy_add(S):
        added = True
        while added and time.time() < deadline - 5:
            added = False
            for x in range(1, N + 1):
                if x not in S and not creates_4ap(x, S):
                    S.add(x)
                    added = True
        return S
    
    S = greedy_add(S)
    best_size = len(S)
    best_set = set(S)
    
    # Local search with random kicks
    random.seed(42)
    for iteration in range(200):
        if time.time() > deadline - 10:
            break
        
        # Random kick: remove k random elements
        k = random.randint(1, 5)
        S = set(best_set)
        if len(S) > k:
            to_remove = random.sample(list(S), k)
            for x in to_remove:
                S.discard(x)
        
        # Greedy fill
        S = greedy_add(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
    
    # Final greedy pass on best_set
    S = set(best_set)
    S = greedy_add(S)
    if len(S) > best_size:
        best_size = len(S)
        best_set = set(S)
    
    return sorted(best_set)

construct()
