def construct():
    from itertools import combinations
    from math import gcd
    import random
    
    n = 128
    nums = list(range(1, n + 1))
    
    # Precompute gcd table
    gcd_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd(i, j)
    
    # Find all forbidden triples
    forbidden = set()
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                if gcd_table[a][b] == gcd_table[a][c] == gcd_table[b][c]:
                    forbidden.add((a, b, c))
    
    # Greedy with backtracking to find large independent set
    def is_valid(selected, x):
        """Check if adding x to selected keeps it independent."""
        for a, b in combinations(selected, 2):
            if (min(a,b,x), min(max(a,b,x), max(a,b,x)), max(a,b,x)) in forbidden:
                # Need to check if {a,b,x} is a forbidden triple
                triple = tuple(sorted([a, b, x]))
                if triple in forbidden:
                    return False
        return True
    
    # Better: precompute for each pair (a,b), the set of c that form forbidden triple
    conflict = {}
    for (a, b, c) in forbidden:
        conflict.setdefault((a, b), set()).add(c)
        conflict.setdefault((a, c), set()).add(b)
        conflict.setdefault((b, c), set()).add(a)
    
    def can_add(selected, x):
        for a, b in combinations(selected, 2):
            key = (min(a, b), max(a, b))
            if x in conflict.get(key, set()):
                return False
        return True
    
    # Greedy: try all orderings, pick best
    best = set()
    
    # Try greedy with different orderings
    for trial in range(50):
        order = nums[:]
        if trial % 2 == 0:
            random.shuffle(order)
        else:
            order.sort(key=lambda x: -len(conflict.get((x, x+1), set())))  # heuristic
        
        selected = []
        for x in order:
            if can_add(selected, x):
                selected.append(x)
        
        if len(selected) > len(best):
            best = set(selected)
    
    # Local search: try to swap elements to improve
    improved = True
    while improved:
        improved = False
        best_set = set(best)
        for x in best_set:
            # Try removing x and adding some other y
            best_set.remove(x)
            for y in nums:
                if y not in best_set:
                    if can_add(list(best_set), y):
                        if len(best_set) + 1 > len(best):
                            best = best_set | {y}
                            improved = True
                            break
            best_set.add(x)
            if improved:
                break
    
    return list(best)
