"""Backtracking with branch-and-bound for maximum independent set in hypergraph."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin before 60s limit
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            lcm_table[i][j] = i * j // math.gcd(i, j)
    
    # Precompute conflict triples: for each pair (a,b), list of c that form conflict
    pair_conflicts = {}
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            pair_conflicts[(a, b)] = []
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    pair_conflicts[(a, b)].append(c)
                    pair_conflicts[(a, c)].append(b)
                    pair_conflicts[(b, c)].append(a)
    
    # For each element x, precompute which other elements conflict with x
    # conflict_with[x] = set of y such that there exists z where {x,y,z} is a conflict triple
    # Actually, we need: when building a set, if we add x, we need to check if it creates
    # a conflict with any pair already in the set.
    
    # Better approach: for each x, and each pair (a,b) that doesn't include x,
    # check if x is in pair_conflicts[(a,b)]
    
    # Precompute: for each x, list of (a,b) pairs where x would create a conflict
    # i.e., x is in pair_conflicts[(a,b)]
    creates_conflict = [[] for _ in range(n + 1)]
    for (a, b), c_list in pair_conflicts.items():
        for c in c_list:
            creates_conflict[c].append((a, b))
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a conflict triple."""
        if x in selected_set:
            return True
        # Check if x creates a conflict with any pair in selected_set
        selected_list = list(selected_set)
        for (a, b) in creates_conflict[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def greedy_solve(order):
        """Greedy: add elements in given order if valid."""
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected
    
    def local_search(selected, max_iters=10000):
        """Fast local search with 1-swap moves."""
        selected = set(selected)
        current_size = len(selected)
        no_improve = 0
        iteration = 0
        
        while iteration < max_iters and time.time() < deadline - 5:
            iteration += 1
            improved = False
            
            # Try 1-swap: remove one, add one
            for x in list(selected):
                for y in range(1, n + 1):
                    if y in selected:
                        continue
                    new_selected = set(selected)
                    new_selected.discard(x)
                    if can_add(new_selected, y):
                        new_selected.add(y)
                        if len(new_selected) > current_size:
                            selected = new_selected
                            current_size = len(selected)
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 50:
                    break
            else:
                no_improve = 0
        
        return selected
    
    # Multiple greedy initializations with different orderings
    best = set()
    
    # Ordering 1: natural order
    best = greedy_solve(elements)
    best = local_search(best)
    
    # Ordering 2: reverse
    best2 = greedy_solve(elements[::-1])
    best2 = local_search(best2)
    if len(best2) > len(best):
        best = best2
    
    # Ordering 3: large numbers first [65,128] then rest
    order3 = list(range(65, 129)) + list(range(1, 65))
    best3 = greedy_solve(order3)
    best3 = local_search(best3)
    if len(best3) > len(best):
        best = best3
    
    # Ordering 4: powers of 2 first, then others
    powers_of_2 = [2**k for k in range(7) if 2**k <= 128]
    others4 = [x for x in elements if x not in powers_of_2]
    order4 = powers_of_2 + others4
    best4 = greedy_solve(order4)
    best4 = local_search(best4)
    if len(best4) > len(best):
        best = best4
    
    # Ordering 5: random
    import random
    for _ in range(3):
        if time.time() > deadline - 10:
            break
        random.shuffle(elements)
        best5 = greedy_solve(elements)
        best5 = local_search(best5)
        if len(best5) > len(best):
            best = best5
    
    # Final local search on best
    if time.time() < deadline - 5:
        best = local_search(best, max_iters=20000)
    
    return list(best)
# EVOLVE-BLOCK-END
