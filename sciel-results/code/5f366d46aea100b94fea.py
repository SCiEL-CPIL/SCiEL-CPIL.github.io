"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations
from functools import lru_cache

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    n = 128
    elements = list(range(1, n + 1))
    
    # Precompute LCM matrix
    lcm_mat = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_mat[i][j] = _lcm(i, j)
            lcm_mat[j][i] = lcm_mat[i][j]
    
    # Build conflict graph: for each pair (a,b), find all c such that
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # This means: a, b, c are all divisors of L = lcm(a,b), and
    # lcm(a,c) = L and lcm(b,c) = L
    # So c must satisfy: lcm(a,c) = L and lcm(b,c) = L
    
    # For a given pair (a,b) with L = lcm(a,b), c is "conflicting" with both
    # if lcm(a,c) == L and lcm(b,c) == L
    # This means c must be a divisor of L such that lcm(a,c)=L and lcm(b,c)=L
    
    # Precompute for each pair (a,b) the set of c that form a bad triple
    conflict_with = [set() for _ in range(n + 1)]
    # For each element x, conflicts[x] = set of pairs (a,b) where x completes a bad triple with a,b
    # Actually, let's think differently:
    # A triple (a,b,c) is bad if lcm(a,b) == lcm(a,c) == lcm(b,c)
    # For a fixed x, it conflicts with pair (a,b) if lcm(a,b)==lcm(a,x)==lcm(b,x)
    
    # Let's precompute: for each x, which pairs (a,b) would form a bad triple with x
    # bad_pairs_with[x] = set of frozenset({a,b}) such that (a,b,x) is a bad triple
    
    # More efficient: for each pair (a,b), compute L = lcm(a,b), then find all c 
    # where lcm(a,c)==L and lcm(b,c)==L
    # Group by L to speed up
    
    # Alternative approach: for each x, determine which other elements y have the property
    # that there exists z such that lcm(x,y)==lcm(x,z)==lcm(y,z)
    
    # Let me think about the structure more carefully.
    # lcm(a,b) == lcm(a,c) == lcm(b,c) = L
    # This means a|L, b|L, c|L, and the LCM of any two is L.
    # So {a,b,c} is a subset of divisors of L where no single element's prime factors
    # cover all of L's prime factors (otherwise its LCM with another would be less than L... wait no)
    # Actually lcm(a,b)=L means that for each prime power p^k || L, at least one of a,b has p^k.
    # Similarly for other pairs.
    # So for each prime power p^k || L, at least two of {a,b,c} must have p^k.
    
    # This is a constraint satisfaction problem. Let me use a different approach.
    
    # For each x, precompute: conflict_set[x] = set of y such that there exists z 
    # forming a bad triple. But this is transitive and complex.
    
    # Better: precompute all bad triples and use them in the algorithm.
    # But there could be many. Let's just check on the fly with the lcm matrix.
    
    def is_bad_triple(a, b, c):
        return lcm_mat[a][b] == lcm_mat[a][c] == lcm_mat[b][c]
    
    # Greedy remove then add back strategy:
    # Phase 1: Start with all elements, remove elements that are in bad triples
    # Phase 2: Try to add back removed elements if they don't create new bad triples
    
    # First, let's try a simple greedy add to get a baseline
    def greedy_add(order):
        selected = set()
        for x in order:
            # Check if adding x creates any bad triple
            can_add = True
            for a, b in combinations(sorted(selected), 2):
                if is_bad_triple(a, b, x):
                    can_add = False
                    break
            if can_add:
                selected.add(x)
        return selected
    
    # Try multiple orders
    import random
    random.seed(42)
    
    best_set = set()
    
    # Order 1: ascending
    s1 = greedy_add(elements)
    if len(s1) > len(best_set):
        best_set = s1
    
    # Order 2: descending
    s2 = greedy_add(list(reversed(elements)))
    if len(s2) > len(best_set):
        best_set = s2
    
    # Order 3: random
    for _ in range(10):
        if time.time() > deadline:
            break
        order = elements[:]
        random.shuffle(order)
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
    
    # Local search: try to improve by swapping or adding/removing
    def local_search(current_set, max_iters=5000):
        current = set(current_set)
        current_list = sorted(current)
        improved = True
        iterations = 0
        while improved and iterations < max_iters and time.time() < deadline:
            improved = False
            iterations += 1
            
            # Try adding each unselected element
            for x in range(1, n + 1):
                if x in current:
                    continue
                can_add = True
                for a, b in combinations(current_list, 2):
                    if is_bad_triple(a, b, x):
                        can_add = False
                        break
                if can_add:
                    current.add(x)
                    current_list = sorted(current)
                    improved = True
                    break  # restart after adding
            
            # Try removing one element and adding two
            if not improved and iterations < max_iters:
                # Try 1-for-2 swaps
                for x in list(current):
                    temp_current = current - {x}
                    temp_list = sorted(temp_current)
                    # Try to add two elements that aren't in current
                    candidates = [y for y in range(1, n + 1) if y not in current]
                    found_pair = False
                    for i, y1 in enumerate(candidates):
                        if time.time() > deadline:
                            return current
                        # Check if y1 can be added to temp_current
                        can_add_y1 = True
                        for a, b in combinations(temp_list, 2):
                            if is_bad_triple(a, b, y1):
                                can_add_y1 = False
                                break
                        if not can_add_y1:
                            continue
                        # Now try to add y2 after y1
                        temp_with_y1 = temp_current | {y1}
                        temp_list_with_y1 = sorted(temp_with_y1)
                        for y2 in candidates[i+1:]:
                            can_add_y2 = True
                            for a, b in combinations(temp_list_with_y1, 2):
                                if is_bad_triple(a, b, y2):
                                    can_add_y2 = False
                                    break
                            if can_add_y2:
                                current = temp_with_y1 | {y2}
                                current_list = sorted(current)
                                improved = True
                                found_pair = True
                                break
                        if found_pair:
                            break
                    if found_pair:
                        break
        
        return current
    
    # Apply local search
    if time.time() < deadline:
        best_set = local_search(best_set, max_iters=10000)
    
    # More aggressive local search with different initializations
    for _ in range(5):
        if time.time() > deadline:
            break
        order = elements[:]
        random.shuffle(order)
        s = greedy_add(order)
        if len(s) > len(best_set):
            best_set = s
        if time.time() < deadline:
            s = local_search(s, max_iters=5000)
            if len(s) > len(best_set):
                best_set = s
    
    # Final polish: try all single removals and re-additions
    if time.time() < deadline:
        current = set(best_set)
        current_list = sorted(current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for x in list(current):
                temp = current - {x}
                temp_list = sorted(temp)
                # Try to add back x or other elements
                for y in range(1, n + 1):
                    if y in temp:
                        continue
                    can_add = True
                    for a, b in combinations(temp_list, 2):
                        if is_bad_triple(a, b, y):
                            can_add = False
                            break
                    if can_add:
                        new_set = temp | {y}
                        if len(new_set) > len(current):
                            current = new_set
                            current_list = sorted(current)
                            improved = True
                            break
                if improved:
                    break
    
    return sorted(current)
# EVOLVE-BLOCK-END
