import random

def construct():
    n = 512
    
    # Precompute conflict triples for each x
    # conflicts[x] is a list of frozensets of 3 integers
    # If any frozenset is a subset of the current selected set, x cannot be added.
    conflicts = [[] for _ in range(n + 1)]
    
    for x in range(1, n + 1):
        # x is the 4th term: a, a+d, a+2d, x => a=x-3d
        # d can go up to (x-1)//3
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            conflicts[x].append(frozenset((a, a + d, a + 2 * d)))
            
        # x is the 2nd term: a, x, x+d, x+2d => a=x-d
        # d can go up to min(x-1, (n-x)//2)
        max_d = min(x - 1, (n - x) // 2)
        for d in range(1, max_d + 1):
            a = x - d
            conflicts[x].append(frozenset((a, x + d, x + 2 * d)))
            
        # x is the 3rd term: a, a+d, x, x+d => a=x-2d
        # d can go up to min((x-1)//2, n-x)
        max_d = min((x - 1) // 2, n - x)
        for d in range(1, max_d + 1):
            a = x - 2 * d
            conflicts[x].append(frozenset((a, a + d, x + d)))
            
        # x is the 1st term: x, x+d, x+2d, x+3d
        # d can go up to (n-x)//3
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            conflicts[x].append(frozenset((x + d, x + 2 * d, x + 3 * d)))
            
    # Greedy function with a given order
    def greedy(order):
        selected = set()
        for x in order:
            can_add = True
            for tri in conflicts[x]:
                if tri.issubset(selected):
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected

    # Multi-start random greedy
    best_set = set()
    best_size = 0
    
    # Run multiple times
    # 1000 iterations should be fast enough.
    # n=512. Precomputation is ~512 * 170 * 4 ~ 350k ops. Fast.
    # Each greedy pass is 512 * 170 * subset_check.
    # subset_check on a set of size < 512 is fast.
    
    # Let's try 2000 iterations to be safe within 60s.
    # If it's too slow, reduce.
    
    for _ in range(2000):
        order = list(range(1, n + 1))
        random.shuffle(order)
        s = greedy(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # Also try deterministic orders like 1..n, n..1, etc.
    # 1..n
    s = greedy(list(range(1, n + 1)))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    # n..1
    s = greedy(list(range(n, 0, -1)))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        
    return sorted(best_set)
