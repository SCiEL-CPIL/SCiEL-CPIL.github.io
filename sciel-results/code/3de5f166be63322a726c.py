"""Rewritten program using branch-and-bound for maximum weight independent set."""

def construct():
    from math import gcd
    import time
    
    N = 512
    n = 511
    start_time = time.time()
    time_limit = 115
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n+1)]
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            if gcd(a, b) > 1:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    # Order by weight descending
    order = list(range(511, 0, -1))
    weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
    
    # Greedy initial solution
    greedy_selected = set()
    greedy_available = set(range(1, 512))
    for v in order:
        if v in greedy_available:
            greedy_selected.add(v)
            greedy_available -= conflicts[v]
    
    best_weight = sum(weights[v] for v in greedy_selected)
    best_set = set(greedy_selected)
    
    # Check if available set is an independent set (no conflicts among them)
    def is_independent(vertices):
        vs = list(vertices)
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                if vs[j] in conflicts[vs[i]]:
                    return False
        return True
    
    import sys
    sys.setrecursionlimit(10000)
    
    def search(idx, selected, available, current_weight):
        nonlocal best_weight, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        if not available:
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected)
            return
        
        # If available is independent, we can take all
        if is_independent(available):
            total = current_weight + sum(weights[v] for v in available)
            if total > best_weight:
                best_weight = total
                best_set = set(selected | available)
            return
        
        # Upper bound
        ub = current_weight + sum(weights[v] for v in available)
        if ub <= best_weight:
            return
        
        if idx >= len(order):
            if current_weight > best_weight:
                best_weight = current_weight
                best_set = set(selected)
            return
        
        v = order[idx]
        
        if v not in available:
            search(idx + 1, selected, available, current_weight)
            return
        
        # Try include
        new_selected = selected | {v}
        new_available = available - {v} - conflicts[v]
        search(idx + 1, new_selected, new_available, current_weight + weights[v])
        
        if time.time() - start_time > time_limit:
            return
        
        # Try exclude
        new_available2 = available - {v}
        search(idx + 1, selected, new_available2, current_weight)
    
    initial_available = frozenset(range(1, 512))
    search(0, frozenset(), initial_available, 0.0)
    
    return list(best_set)
