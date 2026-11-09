"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # seconds, safety margin before 60s kill
    
    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    # Map number to index 0..n-1
    # index i corresponds to number 2+i
    
    # Precompute conflict graph
    # conflict[i] is a set of indices j such that lcm(numbers[i], numbers[j]) <= 256
    conflict = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i].add(j)
                conflict[j].add(i)
                
    # Weights: reciprocal sum. w[i] = 1.0 / numbers[i]
    weights = [1.0 / x for x in numbers]
    
    # Greedy Initial Solution
    # Sort by weight descending (i.e., number ascending)
    order = sorted(range(n), key=lambda i: -weights[i])
    
    selected = set()
    for i in order:
        if all(j not in selected for j in conflict[i]):
            selected.add(i)
            
    current_sum = sum(weights[i] for i in selected)
    
    # Local Search
    # Helper to check if a set of nodes can be added to a base set
    def can_add(nodes_to_add, base_set):
        # Check internal conflicts among nodes_to_add
        for u in nodes_to_add:
            for v in nodes_to_add:
                if u != v and v in conflict[u]:
                    return False
        # Check conflicts with base_set
        for u in nodes_to_add:
            for v in base_set:
                if v in conflict[u]:
                    return False
        return True

    # Helper to get list of nodes not in selected
    # We will iterate through all possible swaps
    
    improved = True
    while improved and elapsed() < DEADLINE:
        improved = False
        
        # 1. Try adding any single node not in selected
        # This is likely to fail if greedy was optimal for additions, 
        # but might succeed if we removed nodes in previous steps.
        for i in range(n):
            if i in selected:
                continue
            if can_add({i}, selected):
                selected.add(i)
                current_sum += weights[i]
                improved = True
                # Restart loop to check for more additions
                break
        
        if improved:
            continue
            
        # 2. Try 1-for-1 swaps
        # For each u in selected, for each v not in selected
        # If v can be added to selected \ {u} and weights[v] > weights[u]
        for u in list(selected):
            if elapsed() > DEADLINE:
                break
            base_set = selected - {u}
            # We can optimize by only checking v that don't conflict with base_set
            # But n is small (255), so O(n^2) is fine.
            for v in range(n):
                if v in selected:
                    continue
                # Check if v conflicts with any node in base_set
                if any(x in conflict[v] for x in base_set):
                    continue
                # If no conflict, we can swap u with v
                if weights[v] > weights[u]:
                    selected.remove(u)
                    selected.add(v)
                    current_sum += weights[v] - weights[u]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 3. Try 1-for-2 swaps
        # For each u in selected, find pair {v, w} not in selected
        # such that v, w can be added to selected \ {u}
        # and weights[v] + weights[w] > weights[u]
        for u in list(selected):
            if elapsed() > DEADLINE:
                break
            base_set = selected - {u}
            
            # Candidate pairs must not conflict with base_set
            candidates = []
            for v in range(n):
                if v in selected:
                    continue
                if any(x in conflict[v] for x in base_set):
                    continue
                candidates.append(v)
                
            # Now find pairs in candidates that don't conflict with each other
            # and have combined weight > weights[u]
            best_swap = None
            best_gain = 0
            
            # To speed up, sort candidates by weight desc
            candidates.sort(key=lambda x: -weights[x])
            
            for i_idx in range(len(candidates)):
                v = candidates[i_idx]
                # If even the best possible partner (max weight) doesn't beat u, stop?
                # Max partner weight is weights[candidates[0]] (if not v)
                # Simple pruning: if weights[v] + max_weight_in_candidates < weights[u], break
                # But we need specific pairs.
                
                for j_idx in range(i_idx + 1, len(candidates)):
                    w = candidates[j_idx]
                    # Check conflict between v and w
                    if w in conflict[v]:
                        continue
                    gain = weights[v] + weights[w] - weights[u]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (v, w)
                        
            if best_swap is not None:
                v, w = best_swap
                selected.remove(u)
                selected.add(v)
                selected.add(w)
                current_sum += best_gain
                improved = True
                break

    # Convert indices back to numbers
    result = [numbers[i] for i in selected]
    return result
# EVOLVE-BLOCK-END
