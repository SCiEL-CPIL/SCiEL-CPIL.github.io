"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict, Counter

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    
    # Precompute lcm for all pairs
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples: (a,b,c) with a<b<c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    bad_triples = []
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, n+1))
    
    # Build mapping: element -> set of bad triples it's in
    elem_triples = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples[x].add(t)
    
    # Precompute: for each element x, the set of pairs (y,z) such that (x,y,z) is a bad triple
    # This is used for fast can_add checks
    # conflict_pairs[x] = list of (y, z) where {x,y,z} is a bad triple
    conflict_pairs = {x: [] for x in range(1, n+1)}
    for (a, b, c) in bad_triples:
        conflict_pairs[a].append((b, c))
        conflict_pairs[b].append((a, c))
        conflict_pairs[c].append((a, b))
    
    def can_add(sol_set, x):
        """Check if adding x to sol_set creates any bad triple."""
        for (y, z) in conflict_pairs[x]:
            if y in sol_set and z in sol_set:
                return False
        return True
    
    def is_valid(sol_set):
        """Check if sol_set is a valid solution (no bad triples)."""
        for x in sol_set:
            for (y, z) in conflict_pairs[x]:
                if y in sol_set and z in sol_set:
                    return False
        return True
    
    # Strategy: Multiple approaches and combine results
    
    best_solution = set()
    best_size = 0
    
    # Approach 1: Greedy hitting set (remove elements to hit all bad triples)
    removed_greedy = set()
    uncovered = set(bad_triples)
    elem_triples_map = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples_map[x].add(t)
    
    while uncovered:
        if time.time() > deadline - 15:
            break
        best_x = None
        best_count = 0
        for x in range(1, n+1):
            if x in removed_greedy:
                continue
            count = len(elem_triples_map[x] & uncovered)
            if count > best_count:
                best_count = count
                best_x = x
        if best_x is None:
            break
        removed_greedy.add(best_x)
        uncovered -= elem_triples_map[best_x]
    
    sol1 = set(range(1, n+1)) - removed_greedy
    if len(sol1) > best_size:
        best_size = len(sol1)
        best_solution = set(sol1)
    
    # Approach 2: Greedy inclusion (add elements in order of least constraint)
    included = set()
    for x in sorted(range(1, n+1), key=lambda x: len(conflict_pairs[x])):
        if can_add(included, x):
            included.add(x)
    if len(included) > best_size:
        best_size = len(included)
        best_solution = set(included)
    
    # Approach 3: Seeded with large range [65,128], then greedily add
    included3 = set(range(65, 129))
    for x in sorted(range(1, 65), key=lambda x: len(conflict_pairs[x])):
        if can_add(included3, x):
            included3.add(x)
    if len(included3) > best_size:
        best_size = len(included3)
        best_solution = set(included3)
    
    # Approach 4: Randomized greedy with multiple restarts
    random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        order = list(range(1, n+1))
        random.shuffle(order)
        # Bias towards larger numbers
        order.sort(key=lambda x: (random.random() > 0.5, -x))
        included4 = set()
        for x in order:
            if can_add(included4, x):
                included4.add(x)
        if len(included4) > best_size:
            best_size = len(included4)
            best_solution = set(included4)
    
    # Local search phase: 1-swap and 2-swap improvements
    current = set(best_solution)
    current_size = best_size
    
    def try_improve_1swap(sol, size):
        """Try 1-swap: remove one element, add another."""
        improved = False
        for x in list(sol):
            if time.time() > deadline - 3:
                break
            temp = sol - {x}
            for y in range(1, n+1):
                if time.time() > deadline - 3:
                    break
                if y not in temp and y != x:
                    if can_add(temp, y):
                        new_size = size  # same size, but different composition
                        # Only accept if it leads to future improvement potential
                        # For now, skip same-size moves in 1-swap
                        pass
        return improved
    
    def try_improve_add(sol):
        """Try to add any element that can be added."""
        improved = False
        for x in range(1, n+1):
            if time.time() > deadline - 2:
                break
            if x not in sol:
                if can_add(sol, x):
                    sol.add(x)
                    improved = True
        return improved
    
    def try_improve_2swap(sol, size):
        """Try 2-swap: remove two elements, add up to three others."""
        improved = False
        sol_list = list(sol)
        for i in range(len(sol_list)):
            if time.time() > deadline - 2:
                break
            for j in range(i+1, len(sol_list)):
                if time.time() > deadline - 2:
                    break
                x, y = sol_list[i], sol_list[j]
                temp = sol - {x, y}
                added = 0
                for z in range(1, n+1):
                    if time.time() > deadline - 2:
                        break
                    if z not in temp and z != x and z != y:
                        if can_add(temp, z):
                            temp.add(z)
                            added += 1
                if added > 2:
                    sol.clear()
                    sol.update(temp)
                    improved = True
                    return improved
        return improved
    
    # Iterated local search
    for iteration in range(100):
        if time.time() > deadline - 3:
            break
        
        # Try to add elements
        if try_improve_add(current):
            current_size = len(current)
            if current_size > best_size:
                best_size = current_size
                best_solution = set(current)
        
        # Try 2-swap
        if try_improve_2swap(current, current_size):
            current_size = len(current)
            if current_size > best_size:
                best_size = current_size
                best_solution = set(current)
        
        # If stuck, do a random kick
        if time.time() > deadline - 5:
            break
        # Random perturbation: remove 1-3 random elements and try to re-add
        kick_size = random.randint(1, 3)
        kick_elements = random.sample(list(current), min(kick_size, len(current)))
        for x in kick_elements:
            current.discard(x)
        # Try to add back and new elements
        try_improve_add(current)
        current_size = len(current)
        if current_size > best_size:
            best_size = current_size
            best_solution = set(current)
    
    # Final cleanup: try to add any remaining elements
    for x in range(1, n+1):
        if time.time() > deadline - 1:
            break
        if x not in best_solution:
            if can_add(best_solution, x):
                best_solution.add(x)
    
    result = sorted(best_solution)
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
