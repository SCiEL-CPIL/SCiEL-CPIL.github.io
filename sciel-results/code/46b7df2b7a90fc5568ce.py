"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples grouped by LCM value for efficiency
    lcm_to_pairs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = lcm_table[i][j]
            lcm_to_pairs[l].append((i, j))
    
    bad_triples = set()
    for l, pairs in lcm_to_pairs.items():
        if len(pairs) < 3:
            continue
        elem_neighbors = defaultdict(list)
        for (i, j) in pairs:
            elem_neighbors[i].append(j)
            elem_neighbors[j].append(i)
        for x in elem_neighbors:
            nbrs = elem_neighbors[x]
            for ii in range(len(nbrs)):
                for jj in range(ii + 1, len(nbrs)):
                    a, b = nbrs[ii], nbrs[jj]
                    bad_triples.add(tuple(sorted((x, a, b))))
    
    bad_triples = list(bad_triples)
    
    # Precompute conflicting pairs for each element
    bad_pairs_for = defaultdict(set)
    for (a, b, c) in bad_triples:
        bad_pairs_for[a].add((b, c))
        bad_pairs_for[b].add((a, c))
        bad_pairs_for[c].add((a, b))
    bad_pairs_for_list = {i: sorted(bad_pairs_for.get(i, [])) for i in range(1, n + 1)}
    
    def can_add(x, current_set):
        for (a, b) in bad_pairs_for_list[x]:
            if a in current_set and b in current_set:
                return False
        return True
    
    best_solution = set()
    best_size = 0
    
    # Multiple randomized greedy construction attempts
    for trial in range(100):
        if time.time() > deadline - 5:
            break
        
        elems = list(range(1, n + 1))
        random.shuffle(elems)
        
        sol = set()
        for x in elems:
            if can_add(x, sol):
                sol.add(x)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = set(sol)
        
        # Add back remaining elements
        remaining = [i for i in range(1, n + 1) if i not in sol]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if can_add(x, sol):
                sol.add(x)
        
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = set(sol)
    
    # Local search with swaps
    for _round in range(10):
        if time.time() > deadline - 3:
            break
        improved = False
        not_in = [i for i in range(1, n + 1) if i not in best_solution]
        random.shuffle(not_in)
        for x in not_in:
            if time.time() > deadline - 3:
                break
            conflicts = set()
            for (a, b) in bad_pairs_for_list[x]:
                if a in best_solution and b in best_solution:
                    conflicts.add(a)
                    conflicts.add(b)
            if not conflicts:
                best_solution.add(x)
                improved = True
                continue
            for y in list(conflicts):
                test_sol = best_solution - {y}
                if can_add(x, test_sol):
                    valid = True
                    for t in bad_triples:
                        if all(e in test_sol | {x} for e in t):
                            valid = False
                            break
                    if valid:
                        best_solution = test_sol | {x}
                        improved = True
                        break
        if not improved:
            break
    
    result = list(best_solution)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
