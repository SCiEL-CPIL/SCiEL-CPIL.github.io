"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    import random
    
    # Precompute perfect squares up to 2*256
    squares = set()
    s = 2
    while s * s <= 2 * n:
        squares.add(s * s)
        s += 1
    
    # For each v, find all u < v such that u+v is a perfect square
    conflict_smaller = [set() for _ in range(n + 1)]
    self_conflict = [False] * (n + 1)
    
    for v in range(1, n + 1):
        if 2 * v in squares:
            self_conflict[v] = True
        for u in range(1, v):
            if (u + v) in squares:
                conflict_smaller[v].add(u)
    
    best_set = set()
    
    # Strategy 1: highest degree first
    order = sorted(
        range(1, n + 1),
        key=lambda x: len(conflict_smaller[x]) + (1 if self_conflict[x] else 0),
        reverse=True
    )
    chosen = set()
    for v in order:
        if self_conflict[v]:
            continue
        if not (chosen & conflict_smaller[v]):
            chosen.add(v)
    if len(chosen) > len(best_set):
        best_set = chosen.copy()
    
    # Strategy 2: lowest degree first
    order2 = sorted(
        range(1, n + 1),
        key=lambda x: len(conflict_smaller[x]) + (1 if self_conflict[x] else 0)
    )
    chosen2 = set()
    for v in order2:
        if self_conflict[v]:
            continue
        if not (chosen2 & conflict_smaller[v]):
            chosen2.add(v)
    if len(chosen2) > len(best_set):
        best_set = chosen2.copy()
    
    # Strategy 3: many random orderings
    random.seed(42)
    for trial in range(1000):
        order3 = list(range(1, n + 1))
        random.shuffle(order3)
        chosen3 = set()
        for v in order3:
            if self_conflict[v]:
                continue
            if not (chosen3 & conflict_smaller[v]):
                chosen3.add(v)
        if len(chosen3) > len(best_set):
            best_set = chosen3.copy()
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
