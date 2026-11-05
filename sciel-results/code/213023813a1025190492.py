import time
from math import gcd, lcm
from collections import defaultdict
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s kill
    
    N = 128
    all_elements = list(range(1, N + 1))
    
    # Pre-compute LCM table for all pairs
    lcm_table = {}
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            lcm_table[(i, j)] = lcm(i, j)
    
    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b) == lcm(a,c) == lcm(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[(a, b)]
            for c in range(b + 1, N + 1):
                if lcm_table[(a, c)] == lab and lcm_table[(b, c)] == lab:
                    bad_triples.append((a, b, c))
    
    if time.time() > deadline:
        return list(range(1, N + 1))
    
    # Build conflict map: for each element v, the set of pairs (u, w) such that {v, u, w} is bad
    conflict_map = defaultdict(set)
    degree = defaultdict(int)
    for a, b, c in bad_triples:
        conflict_map[a].add((b, c))
        conflict_map[b].add((a, c))
        conflict_map[c].add((a, b))
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
    
    # Helper: check if adding element v to selected_set creates a bad triple
    def can_add(v, selected_set):
        for u, w in conflict_map[v]:
            if u in selected_set and w in selected_set:
                return False
        return True

    # Helper: check if selected_set is valid (no bad triples)
    def is_valid(selected_set):
        # Efficient check: for each v in set, check conflicts
        for v in selected_set:
            for u, w in conflict_map[v]:
                if u in selected_set and w in selected_set:
                    return False
        return True

    # Initial Greedy Solution
    def greedy_solve(order):
        selected_set = set()
        for v in order:
            if can_add(v, selected_set):
                selected_set.add(v)
        return selected_set

    # Local Search: 1-for-2 and 2-for-3 swaps
    def local_search(initial_set, time_limit):
        current_set = set(initial_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try 1-for-2
            for v in list(current_set):
                test_set = current_set - {v}
                candidates = [w for w in all_elements if w not in test_set and can_add(w, test_set)]
                if len(candidates) >= 2:
                    # Try to find a pair that can both be added
                    for i in range(len(candidates)):
                        if improved: break
                        for j in range(i + 1, len(candidates)):
                            w1, w2 = candidates[i], candidates[j]
                            if can_add(w1, test_set) and can_add(w2, test_set | {w1}):
                                current_set = test_set | {w1, w2}
                                improved = True
                                break
            if time.time() > time_limit:
                break
            # Try 2-for-3
            # This is more expensive, so we do it less frequently or if time permits
            if time.time() < time_limit - 5:
                # Pick random pairs to remove
                if len(current_set) >= 2:
                    v1, v2 = random.sample(list(current_set), 2)
                    test_set = current_set - {v1, v2}
                    candidates = [w for w in all_elements if w not in test_set and can_add(w, test_set)]
                    if len(candidates) >= 3:
                        # Try to find a triple that can be added
                        # Greedily pick 3
                        found = False
                        for i in range(len(candidates)):
                            if found: break
                            for j in range(i + 1, len(candidates)):
                                if found: break
                                for k in range(j + 1, len(candidates)):
                                    w1, w2, w3 = candidates[i], candidates[j], candidates[k]
                                    if can_add(w1, test_set) and can_add(w2, test_set | {w1}) and can_add(w3, test_set | {w1, w2}):
                                        current_set = test_set | {w1, w2, w3}
                                        improved = True
                                        found = True
                                        break
        return current_set

    # Run Heuristic to get a baseline
    best_set = set()
    best_size = 0
    
    # Sorted by degree
    sorted_elements = sorted(all_elements, key=lambda x: degree[x])
    base_set = greedy_solve(sorted_elements)
    base_set = local_search(base_set, start_time + 10.0)
    best_set = base_set
    best_size = len(base_set)
    
    # Random restarts
    random.seed(42)
    for _ in range(20):
        if time.time() > deadline - 20.0:
            break
        noisy_order = sorted(all_elements, key=lambda x: degree[x] + random.random())
        curr_set = greedy_solve(noisy_order)
        curr_set = local_search(curr_set, start_time + 5.0)
        if len(curr_set) > best_size:
            best_set = curr_set
            best_size = len(best_set)
            
    # Try ILP if time permits
    if time.time() < deadline - 10.0:
        try:
            from scipy.optimize import milp, LinearConstraint, Bounds
            
            if not bad_triples:
                return list(range(1, N + 1))
                
            c = -np.ones(N)
            A = np.zeros((len(bad_triples), N))
            for i, (a, b, c_val) in enumerate(bad_triples):
                A[i, a-1] = 1
                A[i, b-1] = 1
                A[i, c_val-1] = 1
                
            constraints = LinearConstraint(A, ub=2)
            integrality = np.ones(N)
            bounds = Bounds(0, 1)
            
            # Set time limit for the solver
            options = {'time_limit': 10.0, 'mip_rel_gap': 0.0}
            
            res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds, options=options)
            
            if res.status == 0 or res.status == 3: # Optimal or Time Limit Reached with incumbent
                if res.x is not None:
                    selected = [i+1 for i, val in enumerate(res.x) if val > 0.5]
                    if selected and is_valid(set(selected)):
                        if len(selected) > best_size:
                            best_set = set(selected)
                            best_size = len(best_set)
                            
        except Exception as e:
            pass # Fallback to heuristic result

    if not best_set:
        best_set = {1}
        
    return list(best_set)
