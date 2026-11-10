"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations
from collections import Counter, defaultdict

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
    bad_triples = set()
    
    for a in range(1, n+1):
        if time.time() > deadline - 10:
            break
        for b in range(a+1, n+1):
            Lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if time.time() > deadline - 10:
                    break
                if lcm_table[a][c] == Lab and lcm_table[b][c] == Lab:
                    bad_triples.add((a,b,c))
    
    if time.time() > deadline - 5:
        return [1, 2, 4, 8, 16, 32, 64, 128]
    
    if not bad_triples:
        return list(range(1, n+1))
    
    # Build adjacency: for each element, which other elements form a bad triple with it
    # For branch and bound, we need: for each element x, the set of elements that cannot be 
    # in the solution together with x (i.e., there exists a bad triple containing x and that element)
    
    # Actually for maximum independent set in 3-uniform hypergraph, we need:
    # A set S is valid if no hyperedge (bad triple) is fully contained in S.
    # 
    # Branch and bound approach:
    # - We decide for each element whether to include or exclude it
    # - Bound: current_size + remaining_elements <= best_size -> prune
    # - Forward checking: if including x would create a bad triple with two already-included elements, skip
    
    # Precompute: for each pair (a,b), the set of c such that (a,b,c) is a bad triple
    pair_to_third = defaultdict(set)
    for (a,b,c) in bad_triples:
        pair_to_third[(a,b)].add(c)
        pair_to_third[(b,a)].add(c)
        pair_to_third[(a,c)].add(b)
        pair_to_third[(c,a)].add(b)
        pair_to_third[(b,c)].add(a)
        pair_to_third[(c,b)].add(a)
    
    # For each element x, precompute which pairs (y,z) with y,z > x or in some order
    # such that (x,y,z) is a bad triple. This helps with forward checking.
    
    # Element to bad triples it participates in
    elem_in_triples = defaultdict(set)
    for (a,b,c) in bad_triples:
        elem_in_triples[a].add((a,b,c))
        elem_in_triples[b].add((a,b,c))
        elem_in_triples[c].add((a,b,c))
    
    # Count degree of each element
    elem_degree = Counter()
    for x in range(1, n+1):
        elem_degree[x] = len(elem_in_triples[x])
    
    # Sort elements by degree (descending) for better branching
    elements_by_degree = sorted(range(1, n+1), key=lambda x: -elem_degree[x])
    
    # Initial greedy solution for lower bound
    # Greedy: include elements in order of lowest degree first (least restrictive)
    included = set()
    for x in sorted(range(1, n+1), key=lambda x: elem_degree[x]):
        # Check if adding x creates a bad triple
        can_add = True
        for t in elem_in_triples[x]:
            others = [y for y in t if y != x]
            if len(others) == 2 and others[0] in included and others[1] in included:
                can_add = False
                break
        if can_add:
            included.add(x)
    
    best_size = len(included)
    best_solution = set(included)
    
    # Branch and bound
    # State: index in elements_by_degree, current included set, current size
    # We'll use a recursive approach with iterative deepening or just straightforward B&B
    
    # For efficiency, we'll use a different approach:
    # Process elements in order of degree (most constrained first)
    # At each step, decide include or exclude
    
    # Use a stack-based DFS to avoid recursion depth issues
    # State: (pos, included_set, size)
    # But included_set as a set is expensive to copy. Use bitmasks or frozensets.
    
    # Since n=128, we can use Python sets. Let's try with a time limit.
    
    # Alternative: use the complement. Find minimum hitting set (minimum set of elements to remove to hit all bad triples)
    # Then answer = 128 - min_hitting_set_size
    
    # Greedy hitting set for initial upper bound on removals
    removed_greedy = set()
    uncovered = set(bad_triples)
    elem_triples_map = defaultdict(set)
    for t in bad_triples:
        for x in t:
            elem_triples_map[x].add(t)
    
    while uncovered:
        if time.time() > deadline - 3:
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
    
    greedy_solution_size = n - len(removed_greedy)
    if greedy_solution_size > best_size:
        best_size = greedy_solution_size
        best_solution = set(range(1, n+1)) - removed_greedy
    
    # Now try to improve with local search / simulated annealing
    # Start from best_solution and try swaps
    
    current_solution = set(best_solution)
    current_size = best_size
    
    # Precompute for fast checking: for each element not in solution, 
    # which elements in solution would conflict
    # For an element x not in current_solution, it can be added if no bad triple 
    # containing x has both other elements in current_solution
    
    def can_add(x, sol):
        """Check if x can be added to sol without creating a bad triple"""
        for t in elem_in_triples[x]:
            others = [y for y in t if y != x]
            if len(others) == 2 and others[0] in sol and others[1] in sol:
                return False
        return True
    
    def try_add(x, sol):
        """Try to add x to sol, return new sol or None if not possible"""
        if x in sol:
            return sol
        if can_add(x, sol):
            new_sol = sol | {x}
            return new_sol
        return None
    
    # Local search: repeatedly try to add elements, or swap (remove one, add two)
    improved = True
    iteration = 0
    while improved and iteration < 500:
        if time.time() > deadline - 2:
            break
        improved = False
        iteration += 1
        
        # Try adding each excluded element
        for x in range(1, n+1):
            if time.time() > deadline - 2:
                break
            if x not in current_solution:
                if can_add(x, current_solution):
                    current_solution.add(x)
                    current_size += 1
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = set(current_solution)
        
        # Try swaps: remove one element, try to add two others
        for x in list(current_solution):
            if time.time() > deadline - 2:
                break
            # Temporarily remove x
            temp_sol = current_solution - {x}
            added = 0
            for y in range(1, n+1):
                if time.time() > deadline - 2:
                    break
                if y not in temp_sol and y != x:
                    if can_add(y, temp_sol):
                        temp_sol.add(y)
                        added += 1
            if added >= 1:
                # Net gain is added - 1
                if added > 1:
                    current_solution = temp_sol
                    current_size = len(current_solution)
                    improved = True
                    if current_size > best_size:
                        best_size = current_size
                        best_solution = set(current_solution)
                elif added == 1:
                    # Same size, but might enable future improvements. Accept with some probability or always.
                    # For simplicity, accept same-size moves occasionally
                    if iteration % 5 == 0:
                        current_solution = temp_sol
                        current_size = len(current_solution)
                        if current_size > best_size:
                            best_size = current_size
                            best_solution = set(current_solution)
    
    # Final: try to add back any elements that can be added
    for x in range(1, n+1):
        if time.time() > deadline - 1:
            break
        if x not in current_solution:
            if can_add(x, current_solution):
                current_solution.add(x)
                current_size += 1
                if current_size > best_size:
                    best_size = current_size
                    best_solution = set(current_solution)
    
    result = sorted(best_solution)
    
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
