import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 90s limit, 5s safety margin
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: for each element x, store set of (a,b) pairs
    # such that {x,a,b} is a bad triple (all three pairwise LCMs equal)
    conflict = [set() for _ in range(n + 1)]
    
    # Enumerate all bad triples efficiently
    # For each LCM value L, find all pairs (a,b) with lcm(a,b)=L
    # Then for each such pair, any c with lcm(a,c)=L and lcm(b,c)=L forms a bad triple
    lcm_to_pairs = defaultdict(list)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm_table[a][b]
            lcm_to_pairs[L].append((a, b))
    
    # For each LCM value, find all elements that form bad triples
    for L, pairs in lcm_to_pairs.items():
        if len(pairs) < 2:
            continue
        
        # For each element c, check if it forms a bad triple with any two elements from pairs
        # c forms a bad triple with (a,b) if lcm(a,c)=L and lcm(b,c)=L
        # Build a map: for each element e, how many pairs (a,b) in this L group have lcm(e,a)=L and lcm(e,b)=L?
        # Actually simpler: for each c, find all a such that lcm(a,c)=L, then check if any pair (a,b) exists with lcm(b,c)=L
        
        # More efficient: for each c in [1,n], find all a in [1,n] with lcm(a,c)=L
        # Then for each such a, find all b with lcm(b,c)=L and lcm(a,b)=L
        # This is still expensive. Let's just do the direct triple enumeration but optimized.
        
        # Alternative: for each pair (a,b) in pairs, find all c such that lcm(a,c)=L and lcm(b,c)=L
        # Precompute for each L and each element c, the set of elements a with lcm(a,c)=L
        # This would be O(n^2) memory per L, which is too much.
        
        # Let's just do direct enumeration with early termination
        pass
    
    # Direct triple enumeration - this is O(n^3) = 2M operations, should be fast enough
    for a in range(1, n + 1):
        if time.time() > deadline - 30:
            break
        for b in range(a + 1, n + 1):
            lab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    conflict[a].add(frozenset((b, c)))
                    conflict[b].add(frozenset((a, c)))
                    conflict[c].add(frozenset((a, b)))
    
    # Now solve the maximum independent set problem on the 3-uniform hypergraph
    # Use a backtracking algorithm with branch-and-bound
    
    # Convert conflict to a more efficient representation
    # For each element x, store the set of pairs (a,b) as a set of integers (a*n+b for a<b)
    conflict_pairs = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for pair in conflict[x]:
            a, b = tuple(pair)
            if a > b:
                a, b = b, a
            conflict_pairs[x].add(a * (n + 1) + b)
    
    def count_conflicts(x, selected_set):
        """Check if adding x to selected_set creates a bad triple."""
        for pair_code in conflict_pairs[x]:
            a = pair_code // (n + 1)
            b = pair_code % (n + 1)
            if a in selected_set and b in selected_set:
                return True
        return False
    
    # Use a branch-and-bound backtracking algorithm
    # State: (current_set, remaining_candidates, upper_bound)
    # Upper bound: |current_set| + |remaining_candidates|
    
    best_result = set()
    best_size = 0
    
    # Initial greedy to get a good starting solution
    def greedy_solve(order):
        selected = set()
        for x in order:
            if not count_conflicts(x, selected):
                selected.add(x)
        return selected
    
    # Try several deterministic orderings
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: len(conflict_pairs[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict_pairs[x])),
        sorted(range(1, n + 1), key=lambda x: (len(conflict_pairs[x]), x)),
        # Powers of 2 first
        [2**k for k in range(0, 8) if 2**k <= n] + [x for x in range(1, n+1) if x not in {2**k for k in range(0, 8) if 2**k <= n}],
        # Odd numbers first
        list(range(1, n + 1, 2)) + list(range(2, n + 1, 2)),
        # Even numbers first
        list(range(2, n + 1, 2)) + list(range(1, n + 1, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline - 40:
            break
        selected = greedy_solve(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected.copy()
    
    # Randomized greedy restarts
    import random
    rng = random.Random(42)
    for restart in range(100):
        if time.time() > deadline - 40:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = greedy_solve(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected.copy()
    
    # Local search: try to add elements that fit
    current = best_result.copy()
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if not count_conflicts(x, current):
                current.add(x)
                improved = True
                break
    
    # 1-for-1 swaps
    swap_found = True
    while swap_found and time.time() < deadline - 20:
        swap_found = False
        for y in sorted(list(current), key=lambda y: -len(conflict_pairs[y])):
            if time.time() > deadline - 20:
                break
            for x in range(1, n + 1):
                if x in current:
                    continue
                # Check if x can replace y
                temp = current - {y}
                if not count_conflicts(x, temp):
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found:
                break
    
    # Try to add more after swaps
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if not count_conflicts(x, current):
                current.add(x)
                improved = True
                break
    
    # 1-for-2 swaps (remove 1, add 2)
    if time.time() < deadline - 15:
        for x1 in range(1, n + 1):
            if x1 in current or time.time() > deadline - 15:
                continue
            for x2 in range(x1 + 1, n + 1):
                if x2 in current or time.time() > deadline - 15:
                    continue
                for y in list(current):
                    temp = current - {y}
                    if not count_conflicts(x1, temp) and not count_conflicts(x2, temp | {x1}):
                        current.remove(y)
                        current.add(x1)
                        current.add(x2)
                        break
    
    # Final add pass
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if not count_conflicts(x, current):
                current.add(x)
                improved = True
                break
    
    # More aggressive local search with randomized restarts of local search
    if time.time() < deadline - 10:
        for restart in range(20):
            if time.time() > deadline - 10:
                break
            # Perturb current solution
            perturbed = current.copy()
            # Remove a random element
            if perturbed:
                to_remove = list(perturbed)
                rng.shuffle(to_remove)
                perturbed.remove(to_remove[0])
            
            # Try to add back
            improved = True
            while improved and time.time() < deadline - 10:
                improved = False
                for x in range(1, n + 1):
                    if x in perturbed:
                        continue
                    if not count_conflicts(x, perturbed):
                        perturbed.add(x)
                        improved = True
                        break
            
            # 1-for-1 swaps
            swap_found = True
            while swap_found and time.time() < deadline - 10:
                swap_found = False
                for y in list(perturbed):
                    if time.time() > deadline - 10:
                        break
                    for x in range(1, n + 1):
                        if x in perturbed:
                            continue
                        temp = perturbed - {y}
                        if not count_conflicts(x, temp):
                            perturbed.remove(y)
                            perturbed.add(x)
                            swap_found = True
                            break
                    if swap_found:
                        break
            
            if len(perturbed) > len(current):
                current = perturbed
    
    # Final verification and return
    # Verify that current is a valid solution
    def is_valid(solution):
        sol_set = set(solution)
        for a in sol_set:
            for b in sol_set:
                if a >= b:
                    continue
                for c in sol_set:
                    if c <= b:
                        continue
                    if lcm_table[a][b] == lcm_table[a][c] == lcm_table[b][c]:
                        return False
        return True
    
    if not is_valid(current):
        # Fallback to best_result if current is invalid
        if is_valid(best_result):
            current = best_result
        else:
            # Last resort: return a small valid set
            current = {1, 2, 4, 8, 16, 32, 64}
    
    return list(current)
