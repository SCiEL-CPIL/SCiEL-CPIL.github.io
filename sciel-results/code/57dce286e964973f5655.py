import random
import time
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe within 60s
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        lcm_table[i][i] = i
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Build conflict structure: for each element x, store list of pairs (a,b)
    # such that {x,a,b} is a bad triple.
    conflict = [[] for _ in range(n + 1)]
    
    # Precompute divisors for all numbers up to max_lcm_val
    max_lcm_val = 0
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if lcm_table[a][b] > max_lcm_val:
                max_lcm_val = lcm_table[a][b]
                
    divisors = [[] for _ in range(max_lcm_val + 1)]
    for d in range(1, n + 1):
        for multiple in range(d, max_lcm_val + 1, d):
            divisors[multiple].append(d)
            
    # Enumerate bad triples using divisors
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            L = lcm_table[a][b]
            for c in divisors[L]:
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == L and lcm_table[b][c] == L:
                    conflict[a].append((b, c))
                    conflict[b].append((a, c))
                    conflict[c].append((a, b))
                    
    def can_add(x, current_set):
        for u, v in conflict[x]:
            if u in current_set and v in current_set:
                return False
        return True
        
    def greedy_solve(order):
        selected = set()
        for x in order:
            if can_add(x, selected):
                selected.add(x)
        return selected
        
    best_result = set()
    best_size = 0
    
    orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
        sorted(range(1, n + 1), key=lambda x: len(conflict[x])),
        sorted(range(1, n + 1), key=lambda x: -len(conflict[x])),
    ]
    
    for order in orders:
        if time.time() > deadline - 30:
            break
        selected = greedy_solve(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected.copy()
            
    rng = random.Random(42)
    for i in range(50):
        if time.time() > deadline - 20:
            break
        order = list(range(1, n + 1))
        rng.shuffle(order)
        selected = greedy_solve(order)
        if len(selected) > best_size:
            best_size = len(selected)
            best_result = selected.copy()
            
    # Branch and bound for maximum independent set in the "conflict graph"
    # We model this as: we want the largest subset S such that no bad triple exists.
    # This is a 3-uniform hypergraph independent set problem.
    # We'll use a recursive branch-and-bound with a greedy upper bound.
    
    def get_upper_bound(candidates, current_size):
        """Greedy upper bound: greedily add elements to see how many more we could add."""
        remaining = set(candidates)
        current_set = set()
        count = 0
        # Sort by conflict count for better bound
        sorted_candidates = sorted(remaining, key=lambda x: len(conflict[x]))
        for x in sorted_candidates:
            if can_add(x, current_set | best_result):
                current_set.add(x)
                count += 1
        return current_size + count
    
    def branch_and_bound(candidates, current_set, depth=0):
        """Branch and bound to find maximum independent set."""
        nonlocal best_result, best_size
        
        if time.time() > deadline - 5:
            return
        
        if not candidates:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_result = current_set.copy()
            return
        
        # Upper bound
        ub = get_upper_bound(candidates, len(current_set))
        if ub <= best_size:
            return
        
        # Choose a vertex to branch on
        # Pick vertex with fewest conflicts in candidates
        v = min(candidates, key=lambda x: sum(1 for u, w in conflict[x] if u in candidates and w in candidates))
        
        # Branch 1: include v
        if can_add(v, current_set):
            new_current = current_set | {v}
            # Remove v and any candidates that would form a bad triple with v
            new_candidates = set()
            for c in candidates:
                if c == v:
                    continue
                if can_add(c, new_current):
                    new_candidates.add(c)
            branch_and_bound(new_candidates, new_current, depth + 1)
        
        # Branch 2: exclude v
        new_candidates = candidates - {v}
        branch_and_bound(new_candidates, current_set, depth + 1)
    
    # Initialize with best greedy result
    current = best_result.copy()
    candidates = set(range(1, n + 1)) - current
    
    # Run branch and bound
    branch_and_bound(candidates, current)
    
    # Local search improvement
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if can_add(x, current):
                current.add(x)
                improved = True
                break
                
    swap_found = True
    while swap_found and time.time() < deadline - 5:
        swap_found = False
        for y in list(current):
            if time.time() > deadline - 5:
                break
            temp_current = current - {y}
            for x in range(1, n + 1):
                if x in current:
                    continue
                if can_add(x, temp_current):
                    current.remove(y)
                    current.add(x)
                    swap_found = True
                    break
            if swap_found:
                break
                
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for x in range(1, n + 1):
            if x in current:
                continue
            if can_add(x, current):
                current.add(x)
                improved = True
                break
                
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
        if is_valid(best_result):
            current = best_result
        else:
            current = {2**k for k in range(0, 8) if 2**k <= n}
            
    return list(current)
