"""Optimized approach: Prime-factor-based conflict detection + multi-swap local search + branch-and-bound."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt
from fractions import Fraction
import numpy as np
from itertools import combinations

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    if n < 9:
        return True
    if n % 3 == 0:
        return False
    if n < 25:
        return True
    if n % 5 == 0:
        return False
    if n < 49:
        return True
    if n % 7 == 0:
        return False
    if n < 121:
        return True
    i = 11
    while i <= isqrt(n):
        if n % i == 0:
            return False
        i += 2
    return True

def _get_prime_factors(n):
    """Return list of unique prime factors of n."""
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    N = 512
    
    # Precompute values and prime factors for all numbers 1..511
    values = [Fraction(0)] * N
    float_values = [0.0] * N
    prime_factors = [[] for _ in range(N)]
    is_prime_arr = [False] * N
    
    for a in range(1, N):
        values[a] = Fraction(1, N - a)
        float_values[a] = 1.0 / (N - a)
        prime_factors[a] = _get_prime_factors(a)
        is_prime_arr[a] = _is_prime(a)
        
    # Identify primes
    primes = [a for a in range(1, N) if is_prime_arr[a]]
    
    # Initialize with all primes
    selected = set(primes)
    
    # Build mapping: prime -> list of selected elements that have this prime as a factor
    prime_to_selected = {}
    for b in selected:
        for p in prime_factors[b]:
            if p not in prime_to_selected:
                prime_to_selected[p] = []
            prime_to_selected[p].append(b)
    
    # Marker array to track which elements are currently in conflict
    conflict_mark = [0] * N
    conflict_iter = 0
    
    def find_conflicts(candidates):
        """Find all selected elements that conflict with candidates. Returns list."""
        nonlocal conflict_iter
        conflict_iter += 1
        conflict_list = []
        for c in candidates:
            for p in prime_factors[c]:
                if p in prime_to_selected:
                    for b in prime_to_selected[p]:
                        if conflict_mark[b] != conflict_iter:
                            conflict_mark[b] = conflict_iter
                            conflict_list.append(b)
        return conflict_list
    
    def apply_move(add_candidates, remove_elements):
        """Apply a move: remove elements, add candidates. Update prime_to_selected."""
        for b in remove_elements:
            for p in prime_factors[b]:
                if p in prime_to_selected:
                    lst = prime_to_selected[p]
                    for idx, val in enumerate(lst):
                        if val == b:
                            lst.pop(idx)
                            break
                    if not lst:
                        del prime_to_selected[p]
        for c in add_candidates:
            for p in prime_factors[c]:
                if p not in prime_to_selected:
                    prime_to_selected[p] = []
                prime_to_selected[p].append(c)
        selected.difference_update(remove_elements)
        selected.update(add_candidates)
    
    def get_score():
        return sum(values[a] for a in selected)
    
    current_score = get_score()
    
    # 1-for-k move search with prime-factor-based conflict detection
    improved = True
    while improved:
        if time.time() > deadline:
            break
            
        improved = False
        best_gain = Fraction(0)
        best_add = None
        best_remove = None
        best_float_gain = 0.0
        
        for c in range(1, N):
            if c in selected:
                continue
            
            conflict_list = find_conflicts([c])
            
            float_cost = sum(float_values[b] for b in conflict_list)
            float_gain = float_values[c] - float_cost
            
            if float_gain <= best_float_gain - 1e-15:
                continue
            
            conflict_cost = sum(values[b] for b in conflict_list)
            gain = values[c] - conflict_cost
            
            if gain > best_gain:
                best_gain = gain
                best_float_gain = float_gain
                best_add = [c]
                best_remove = list(conflict_list)
        
        if best_add is not None:
            apply_move(best_add, best_remove)
            current_score += best_gain
            improved = True
    
    # 2-for-k move search
    if time.time() < deadline - 15:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: float_values[x], reverse=True)
        top_composites = composites[:80]
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            best_float_gain = 0.0
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    
                    conflict_list = find_conflicts([c1, c2])
                    
                    float_cost = sum(float_values[b] for b in conflict_list)
                    float_gain = float_values[c1] + float_values[c2] - float_cost
                    
                    if float_gain <= best_float_gain - 1e-15:
                        continue
                    
                    conflict_cost = sum(values[b] for b in conflict_list)
                    gain = values[c1] + values[c2] - conflict_cost
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_float_gain = float_gain
                        best_add = [c1, c2]
                        best_remove = list(conflict_list)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # 3-for-k move search
    if time.time() < deadline - 10:
        composites = [a for a in range(1, N) if not is_prime_arr[a] and a > 1]
        composites.sort(key=lambda x: float_values[x], reverse=True)
        top_composites = composites[:50]
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
                
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            best_float_gain = 0.0
            
            for i in range(len(top_composites)):
                c1 = top_composites[i]
                if c1 in selected:
                    continue
                for j in range(i + 1, len(top_composites)):
                    c2 = top_composites[j]
                    if c2 in selected:
                        continue
                    if gcd(c1, c2) != 1:
                        continue
                    for k in range(j + 1, len(top_composites)):
                        c3 = top_composites[k]
                        if c3 in selected:
                            continue
                        if gcd(c1, c3) != 1 or gcd(c2, c3) != 1:
                            continue
                        
                        conflict_list = find_conflicts([c1, c2, c3])
                        
                        float_cost = sum(float_values[b] for b in conflict_list)
                        float_gain = float_values[c1] + float_values[c2] + float_values[c3] - float_cost
                        
                        if float_gain <= best_float_gain - 1e-15:
                            continue
                        
                        conflict_cost = sum(values[b] for b in conflict_list)
                        gain = values[c1] + values[c2] + values[c3] - conflict_cost
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_float_gain = float_gain
                            best_add = [c1, c2, c3]
                            best_remove = list(conflict_list)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Final pass of 1-for-k to clean up
    if time.time() < deadline - 5:
        improved = True
        while improved:
            if time.time() > deadline - 1:
                break
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            best_float_gain = 0.0
            
            for c in range(1, N):
                if c in selected:
                    continue
                conflict_list = find_conflicts([c])
                
                float_cost = sum(float_values[b] for b in conflict_list)
                float_gain = float_values[c] - float_cost
                
                if float_gain <= best_float_gain - 1e-15:
                    continue
                
                conflict_cost = sum(values[b] for b in conflict_list)
                gain = values[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_float_gain = float_gain
                    best_add = [c]
                    best_remove = list(conflict_list)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Branch-and-bound global optimization pass
    if time.time() < deadline - 20:
        # Build conflict graph
        conflict_sets = [set() for _ in range(N)]
        for a in range(1, N):
            for b in range(a + 1, N):
                if gcd(a, b) != 1:
                    conflict_sets[a].add(b)
                    conflict_sets[b].add(a)
        
        # Sort vertices by value descending for B&B
        all_vertices = list(range(1, N))
        all_vertices.sort(key=lambda x: float_values[x], reverse=True)
        
        # Precompute neighbor bitmasks using Python integers (arbitrary precision)
        # Vertex index: 0..510 maps to value 1..511
        vertex_index = {v: i for i, v in enumerate(all_vertices)}
        n_vertices = len(all_vertices)
        
        # Build adjacency as bitmask sets
        adj_masks = [0] * n_vertices
        for i in range(n_vertices):
            v = all_vertices[i]
            for nb in conflict_sets[v]:
                j = vertex_index[nb]
                adj_masks[i] |= (1 << j)
        
        # Fractional knapsack upper bound: for remaining candidates,
        # sort by value and take greedily, allowing fractional last item
        # This is an upper bound on the MWIS of the induced subgraph
        
        def fractional_knapsack_bound(candidate_indices):
            """Compute fractional knapsack upper bound for a set of vertex indices."""
            if not candidate_indices:
                return 0.0
            # Sort by value descending
            sorted_idx = sorted(candidate_indices, key=lambda i: float_values[all_vertices[i]], reverse=True)
            bound = 0.0
            for idx in sorted_idx:
                bound += float_values[all_vertices[idx]]
            return bound
        
        # Better bound: use the fact that if we pick vertex v, we must exclude its neighbors
        # Upper bound = sum of all candidate values (weak)
        # Stronger: for each candidate, value(v) + sum of non-neighbor candidates
        # Take max over candidates, or use LP relaxation
        
        # Even better: maximum weight independent set bound via greedy coloring
        # For now, use a simple but effective bound:
        # UB = sum of top k values where k is the independence number estimate
        
        # Implement a proper B&B with memoization-like pruning
        best_solution = set(selected)
        best_score_float = sum(float_values[a] for a in selected)
        best_score_exact = current_score
        
        # B&B state
        def bb_search(available_mask, current_weight, current_set, depth=0):
            """Branch and bound for MWIS."""
            nonlocal best_solution, best_score_float, best_score_exact
            
            if time.time() > deadline - 5:
                return
            
            # If no available vertices, update best if improved
            if available_mask == 0:
                if current_weight > best_score_float + 1e-12:
                    best_score_float = current_weight
                    best_score_exact = sum(values[a] for a in current_set)
                    best_solution = set(current_set)
                return
            
            # Upper bound: sum of all available vertex values
            # This is weak but fast
            avail_list = [i for i in range(n_vertices) if (available_mask >> i) & 1]
            if not avail_list:
                return
            
            ub = sum(float_values[all_vertices[i]] for i in avail_list)
            if current_weight + ub <= best_score_float + 1e-12:
                return
            
            # Choose branching vertex: highest value available
            # Or use a heuristic: vertex with highest degree in available subgraph
            # For speed, just pick the highest value
            branch_idx = max(avail_list, key=lambda i: float_values[all_vertices[i]])
            branch_val = all_vertices[branch_idx]
            
            # Branch 1: Include branch_idx
            # Remove it and all its neighbors from available
            new_mask = available_mask & ~(1 << branch_idx)
            # Remove neighbors
            new_mask &= ~adj_masks[branch_idx]
            new_set = current_set | {branch_val}
            bb_search(new_mask, current_weight + float_values[branch_val], new_set, depth + 1)
            
            # Branch 2: Exclude branch_idx
            new_mask2 = available_mask & ~(1 << branch_idx)
            bb_search(new_mask2, current_weight, current_set, depth + 1)
        
        # Start B&B from empty set with all vertices available
        # But this is too slow for 512 vertices. Instead, start from current solution
        # and only consider adding/removing a few vertices
        
        # Better approach: Use the current solution as a starting point
        # and do a limited B&B that only considers swaps involving top candidates
        
        # Identify promising candidates: high-value vertices not in selected
        candidates = [v for v in range(1, N) if v not in selected]
        candidates.sort(key=lambda x: float_values[x], reverse=True)
        top_candidates = candidates[:30]
        
        # For each subset of top candidates (size 1-3), try to add them
        # and remove conflicting selected elements, then do local search
        
        for size in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            for combo in combinations(top_candidates, size):
                if time.time() > deadline - 5:
                    break
                
                # Check if combo is internally coprime
                valid = True
                for i in range(len(combo)):
                    for j in range(i + 1, len(combo)):
                        if gcd(combo[i], combo[j]) != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Find conflicts with current selection
                conflicts = set()
                for c in combo:
                    for b in selected:
                        if gcd(c, b) != 1:
                            conflicts.add(b)
                
                # Calculate gain
                gain = sum(float_values[c] for c in combo) - sum(float_values[b] for b in conflicts)
                if gain <= 0:
                    continue
                
                # Apply move
                new_selected = set(selected)
                new_selected.difference_update(conflicts)
                new_selected.update(combo)
                
                # Do local search from this state
                temp_prime_to_selected = {}
                for b in new_selected:
                    for p in prime_factors[b]:
                        if p not in temp_prime_to_selected:
                            temp_prime_to_selected[p] = []
                        temp_prime_to_selected[p].append(b)
                
                temp_conflict_mark = [0] * N
                temp_conflict_iter = 0
                
                def temp_find_conflicts(candidates_list):
                    nonlocal temp_conflict_iter
                    temp_conflict_iter += 1
                    conflict_list = []
                    for c in candidates_list:
                        for p in prime_factors[c]:
                            if p in temp_prime_to_selected:
                                for b in temp_prime_to_selected[p]:
                                    if temp_conflict_mark[b] != temp_conflict_iter:
                                        temp_conflict_mark[b] = temp_conflict_iter
                                        conflict_list.append(b)
                    return conflict_list
                
                def temp_apply_move(add_list, remove_list):
                    for b in remove_list:
                        for p in prime_factors[b]:
                            if p in temp_prime_to_selected:
                                lst = temp_prime_to_selected[p]
                                for idx, val in enumerate(lst):
                                    if val == b:
                                        lst.pop(idx)
                                        break
                                if not lst:
                                    del temp_prime_to_selected[p]
                    for c in add_list:
                        for p in prime_factors[c]:
                            if p not in temp_prime_to_selected:
                                temp_prime_to_selected[p] = []
                            temp_prime_to_selected[p].append(c)
                    new_selected.difference_update(remove_list)
                    new_selected.update(add_list)
                
                # 1-for-k local search
                temp_improved = True
                while temp_improved and time.time() < deadline - 5:
                    temp_improved = False
                    temp_best_gain = 0.0
                    temp_best_add = None
                    temp_best_remove = None
                    
                    for c in range(1, N):
                        if c in new_selected:
                            continue
                        
                        temp_conflict_list = temp_find_conflicts([c])
                        
                        temp_float_cost = sum(float_values[b] for b in temp_conflict_list)
                        temp_float_gain = float_values[c] - temp_float_cost
                        
                        if temp_float_gain <= temp_best_gain - 1e-15:
                            continue
                        
                        if temp_float_gain > temp_best_gain:
                            temp_best_gain = temp_float_gain
                            temp_best_add = [c]
                            temp_best_remove = list(temp_conflict_list)
                    
                    if temp_best_add is not None:
                        temp_apply_move(temp_best_add, temp_best_remove)
                        temp_improved = True
                
                # Check if we found a better solution
                temp_score = sum(float_values[a] for a in new_selected)
                if temp_score > best_score_float + 1e-12:
                    best_score_float = temp_score
                    best_score_exact = sum(values[a] for a in new_selected)
                    best_solution = set(new_selected)
        
        # Apply the best solution found
        if best_solution != selected:
            selected.clear()
            prime_to_selected.clear()
            selected.update(best_solution)
            for b in selected:
                for p in prime_factors[b]:
                    if p not in prime_to_selected:
                        prime_to_selected[p] = []
                    prime_to_selected[p].append(b)
            current_score = best_score_exact
    
    # Final 1-for-k polish to ensure local optimality
    if time.time() < deadline - 2:
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            best_gain = Fraction(0)
            best_add = None
            best_remove = None
            best_float_gain = 0.0
            
            for c in range(1, N):
                if c in selected:
                    continue
                conflict_list = find_conflicts([c])
                
                float_cost = sum(float_values[b] for b in conflict_list)
                float_gain = float_values[c] - float_cost
                
                if float_gain <= best_float_gain - 1e-15:
                    continue
                
                conflict_cost = sum(values[b] for b in conflict_list)
                gain = values[c] - conflict_cost
                if gain > best_gain:
                    best_gain = gain
                    best_float_gain = float_gain
                    best_add = [c]
                    best_remove = list(conflict_list)
            
            if best_add is not None:
                apply_move(best_add, best_remove)
                current_score += best_gain
                improved = True
    
    # Final Verification
    result = list(selected)
    for a in result:
        if a < 1 or a > 511:
            raise ValueError(f"Out of range: {a}")
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if gcd(result[i], result[j]) != 1:
                raise ValueError(f"Not coprime: {result[i]}, {result[j]}")
    
    return result
# EVOLVE-BLOCK-END
