"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _precompute_forbidden(N=180):
    """Precompute all forbidden pairs for each element.
    
    For 1/a = 1/b + 1/c with distinct a, b, c:
    We can assume a < b and a < c (since 1/a > 1/b and 1/a > 1/c).
    Without loss of generality, assume b <= c.
    Then c = a*b / (b - a), which must be a positive integer in [1, N] distinct from a and b.
    
    We store for each element x, the set of elements y such that x and y appear together
    in some forbidden triple. Actually, it's better to store for each pair (x, y) whether
    they form a forbidden triple with some z.
    
    Let's think differently: for each a, and each b > a, if c = a*b/(b-a) is integer and in [1,N]
    and c != a and c != b, then (a, b, c) is a forbidden triple.
    In this triple, a is the smallest. So the constraint is: if a is in the set, then we cannot
    have both b and c in the set.
    
    So for each a, we have a list of pairs (b, c) such that having both b and c with a is forbidden.
    """
    # forbidden[a] = list of (b, c) pairs where b < c, both > a, and 1/a = 1/b + 1/c
    forbidden = [[] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            d = b - a
            if (a * b) % d == 0:
                c = (a * b) // d
                if 1 <= c <= N and c != a and c != b:
                    # Ensure b <= c for canonical form
                    if b > c:
                        b, c = c, b
                    forbidden[a].append((b, c))
    return forbidden

def _is_valid(S, forbidden, N=180):
    """Check if set S has no forbidden triple."""
    S_set = set(S)
    for a in S:
        for (b, c) in forbidden[a]:
            if b in S_set and c in S_set:
                return False
    return True

def _greedy_construct(forbidden, N=180):
    """Greedy construction: add elements in some order, skip if it creates a forbidden triple."""
    S = set()
    # Try different orderings
    best_S = set()
    
    # Strategy 1: Add from large to small
    for a in range(N, 0, -1):
        valid = True
        for (b, c) in forbidden[a]:
            if b in S and c in S:
                valid = False
                break
        if valid:
            S.add(a)
    
    if len(S) > len(best_S):
        best_S = set(S)
    
    # Strategy 2: Add from small to large
    S = set()
    for a in range(1, N + 1):
        valid = True
        for (b, c) in forbidden[a]:
            if b in S and c in S:
                valid = False
                break
        if valid:
            S.add(a)
    
    if len(S) > len(best_S):
        best_S = set(S)
    
    # Strategy 3: Random orderings
    for _ in range(100):
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for a in order:
            valid = True
            for (b, c) in forbidden[a]:
                if b in S and c in S:
                    valid = False
                    break
            if valid:
                S.add(a)
        if len(S) > len(best_S):
            best_S = set(S)
    
    return list(best_S)

def _local_search(S, forbidden, N=180, max_iters=5000):
    """Local search: try to add elements or swap elements to improve the set."""
    S_set = set(S)
    best_size = len(S_set)
    
    for _ in range(max_iters):
        improved = False
        
        # Try adding each element not in S
        for a in range(1, N + 1):
            if a in S_set:
                continue
            valid = True
            for (b, c) in forbidden[a]:
                if b in S_set and c in S_set:
                    valid = False
                    break
            if valid:
                S_set.add(a)
                best_size = len(S_set)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one element, add another
        for remove in list(S_set):
            S_set.remove(remove)
            for add in range(1, N + 1):
                if add in S_set:
                    continue
                valid = True
                for (b, c) in forbidden[add]:
                    if b in S_set and c in S_set:
                        valid = False
                        break
                if valid:
                    S_set.add(add)
                    if len(S_set) > best_size:
                        best_size = len(S_set)
                    improved = True
                    break
            if improved:
                break
            else:
                S_set.add(remove)
        
        if not improved:
            break
    
    return list(S_set)

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline with 10s safety margin
    
    N = 180
    forbidden = _precompute_forbidden(N)
    
    # Start with greedy
    best_S = _greedy_construct(forbidden, N)
    best_size = len(best_S)
    
    # Local search from the greedy result
    if time.time() < deadline:
        improved_S = _local_search(best_S, forbidden, N, max_iters=10000)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(improved_S)
    
    # Try more random restarts with local search
    while time.time() < deadline - 10:
        # Random initial set
        S = set()
        order = list(range(1, N + 1))
        random.shuffle(order)
        for a in order:
            valid = True
            for (b, c) in forbidden[a]:
                if b in S and c in S:
                    valid = False
                    break
            if valid:
                S.add(a)
        
        # Local search
        S_list = list(S)
        improved_S = _local_search(S_list, forbidden, N, max_iters=5000)
        if len(improved_S) > best_size:
            best_S = improved_S
            best_size = len(improved_S)
    
    return best_S

# EVOLVE-BLOCK-END
