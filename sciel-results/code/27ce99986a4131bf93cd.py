import math
import time
from fractions import Fraction
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    N = 512
    MAX_A = 511
    deadline = start_time + 110.0
    
    # Precompute prime factors for each number 1..511
    factors = [set() for _ in range(MAX_A + 1)]
    factors[1] = set()
    for i in range(2, MAX_A + 1):
        if not factors[i]:
            for j in range(i, MAX_A + 1, i):
                factors[j].add(i)
    
    # Weights: w[a] = 1/(512-a), strictly increasing with a
    weights = [0.0] * (MAX_A + 1)
    for a in range(1, MAX_A + 1):
        weights[a] = 1.0 / (N - a)
    
    # Build conflict graph: conflict[a] = set of b > a that share a factor with a
    conflict = [set() for _ in range(MAX_A + 1)]
    for a in range(1, MAX_A + 1):
        for b in range(a + 1, MAX_A + 1):
            if factors[a] & factors[b]:
                conflict[a].add(b)
                conflict[b].add(a)
    
    # Sort vertices by weight descending (a descending since weights increase with a)
    # We'll use 1-indexed a values
    vertices = list(range(MAX_A, 0, -1))
    
    # Greedy initial solution: take vertices in order if coprime to all chosen
    def is_coprime_to_all(a, chosen):
        for b in chosen:
            if b in conflict[a]:
                return False
        return True
    
    chosen = set()
    for a in vertices:
        if is_coprime_to_all(a, chosen):
            chosen.add(a)
    
    best_score = sum(weights[a] for a in chosen)
    best_set = set(chosen)
    
    # Branch and bound for max weight independent set
    # State: (current_set, remaining_candidates, current_weight)
    # We process vertices in order of decreasing weight
    
    # Precompute for each vertex, the list of later vertices that conflict with it
    # This helps with pruning
    
    def branch_and_bound():
        nonlocal best_score, best_set, deadline
        
        # Sort vertices by weight descending
        ordered = sorted(range(1, MAX_A + 1), key=lambda x: -weights[x])
        
        # For each position in the ordered list, precompute which subsequent vertices conflict
        # conflict_later[i] = set of indices j > i such that ordered[i] and ordered[j] conflict
        conflict_later = [set() for _ in range(MAX_A)]
        for i in range(MAX_A):
            vi = ordered[i]
            for j in range(i + 1, MAX_A):
                vj = ordered[j]
                if vi in conflict[vj]:  # conflict is symmetric
                    conflict_later[i].add(j)
        
        # Upper bound: sum of weights of all remaining vertices that don't conflict with chosen
        # Tighter: for each remaining vertex, if it conflicts with any chosen, it can't be added
        
        # Use iterative deepening / best-first search
        # State: (index, chosen_set, current_weight)
        # We'll use a priority queue to explore most promising branches first
        
        import heapq
        
        # Start with the greedy solution
        initial_chosen = set()
        for a in vertices:
            if is_coprime_to_all(a, initial_chosen):
                initial_chosen.add(a)
        
        # Map from a to its index in ordered
        a_to_idx = {a: i for i, a in enumerate(ordered)}
        
        # Initialize chosen set from greedy
        chosen_mask = set(a_to_idx[a] for a in initial_chosen)
        
        # Priority queue: (-weight, index, chosen_mask)
        # We explore the branch where we include ordered[index] first
        pq = []
        heapq.heappush(pq, (-sum(weights[a] for a in initial_chosen), 0, frozenset(chosen_mask)))
        
        explored = 0
        
        while pq and time.time() < deadline - 5.0:
            neg_w, idx, chosen_frozen = heapq.heappop(pq)
            w = -neg_w
            chosen_set_local = set(chosen_frozen)
            
            if w + sum(weights[ordered[j]] for j in range(idx, MAX_A) if j not in chosen_set_local and not any(ordered[j] in conflict[ordered[k]] for k in chosen_set_local)) <= best_score:
                continue
            
            explored += 1
            if explored % 10000 == 0:
                if time.time() > deadline - 10.0:
                    break
            
            # Try including ordered[idx]
            vi = ordered[idx]
            if vi not in chosen_set_local:
                # Check if vi conflicts with any in chosen_set_local
                can_add = True
                for k in chosen_set_local:
                    if vi in conflict[ordered[k]]:
                        can_add = False
                        break
                
                if can_add:
                    new_chosen = chosen_set_local | {vi}
                    new_w = w + weights[vi]
                    
                    if new_w > best_score:
                        best_score = new_w
                        best_set = set(ordered[i] for i in new_chosen)
                    
                    # Push state where we include vi
                    if idx + 1 < MAX_A:
                        heapq.heappush(pq, (-new_w, idx + 1, frozenset(new_chosen)))
            
            # Try excluding ordered[idx]
            if idx + 1 < MAX_A:
                # Upper bound for excluding: current weight + sum of weights of remaining vertices that can still be added
                ub = w
                for j in range(idx + 1, MAX_A):
                    vj = ordered[j]
                    if vj not in chosen_set_local and not any(vj in conflict[ordered[k]] for k in chosen_set_local):
                        ub += weights[vj]
                
                if ub > best_score:
                    heapq.heappush(pq, (-w, idx + 1, chosen_frozen))
    
    branch_and_bound()
    
    # Final local search improvement
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        best_gain = 0.0
        best_swap = None
        
        for x in list(best_set):
            # Try to replace x with a larger y not in best_set
            for y in range(MAX_A, x + 1, -1):
                if y in best_set:
                    continue
                # Check if y is coprime to all in best_set except x
                valid = True
                for z in best_set:
                    if z == x:
                        continue
                    if y in conflict[z]:
                        valid = False
                        break
                if valid:
                    gain = weights[y] - weights[x]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (x, y)
                    break  # Only check the best y for this x
        
        if best_swap:
            x, y = best_swap
            best_set.remove(x)
            best_set.add(y)
            best_score += best_gain
            improved = True
    
    return list(best_set)
