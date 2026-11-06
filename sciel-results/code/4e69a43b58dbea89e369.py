import math
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Build conflict graph: edge (a,b) if lcm(a,b) <= 256
    numbers = list(range(2, 257))
    n = len(numbers)
    num_to_idx = {x: i for i, x in enumerate(numbers)}
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = numbers[i]
        for j in range(i+1, n):
            b = numbers[j]
            if math.lcm(a, b) <= 256:
                adj[i].add(j)
                adj[j].add(i)
    
    # Weights: reciprocal
    weights = [1.0 / numbers[i] for i in range(n)]
    
    def is_valid_independent_set(selected_indices):
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                if selected_indices[j] in adj[selected_indices[i]]:
                    return False
        return True
    
    def weight_of_set(selected_indices):
        return sum(weights[i] for i in selected_indices)
    
    # Greedy initialization: sort by weight descending, add if no conflict
    def greedy_init(order=None):
        if order is None:
            order = sorted(range(n), key=lambda i: -weights[i])
        selected = set()
        for i in order:
            if not any(j in selected for j in adj[i]):
                selected.add(i)
        return selected
    
    # Local search with various swap types
    def local_search(selected, time_limit=None):
        improved = True
        while improved:
            if time_limit and time.time() > time_limit:
                break
            improved = False
            selected_list = list(selected)
            
            # 1-for-1 swap: remove i, add j
            for i in selected_list:
                for j in range(n):
                    if j in selected:
                        continue
                    # Check if j conflicts with any remaining selected
                    if any(k in adj[j] for k in selected if k != i):
                        continue
                    new_set = set(selected)
                    new_set.discard(i)
                    new_set.add(j)
                    if weight_of_set(new_set) > weight_of_set(selected) + 1e-15:
                        selected = new_set
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 2-for-1 swap: remove i1, i2, add j
            for i1 in selected_list:
                for i2 in selected_list:
                    if i1 >= i2:
                        continue
                    for j in range(n):
                        if j in selected:
                            continue
                        # Check if j conflicts with any remaining selected (excluding i1, i2)
                        remaining = [k for k in selected if k != i1 and k != i2]
                        if any(k in adj[j] for k in remaining):
                            continue
                        new_set = set(selected)
                        new_set.discard(i1)
                        new_set.discard(i2)
                        new_set.add(j)
                        if weight_of_set(new_set) > weight_of_set(selected) + 1e-15:
                            selected = new_set
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue
            
            # 1-for-2 swap: remove i, add j1, j2
            for i in selected_list:
                found = False
                for j1 in range(n):
                    if j1 in selected or j1 == i:
                        continue
                    for j2 in range(n):
                        if j2 in selected or j2 == i or j2 <= j1:
                            continue
                        if j2 in adj[j1]:
                            continue
                        # Check if j1, j2 conflict with remaining selected (excluding i)
                        remaining = [k for k in selected if k != i]
                        if any(k in adj[j1] for k in remaining):
                            continue
                        if any(k in adj[j2] for k in remaining):
                            continue
                        new_set = set(selected)
                        new_set.discard(i)
                        new_set.add(j1)
                        new_set.add(j2)
                        if weight_of_set(new_set) > weight_of_set(selected) + 1e-15:
                            selected = new_set
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        return selected
    
    # Branch and bound
    def branch_and_bound():
        # Order vertices by weight descending
        order = sorted(range(n), key=lambda i: -weights[i])
        
        best_set = set()
        best_weight = 0.0
        
        # Greedy upper bound: for remaining vertices, sum of top weights that don't conflict
        def upper_bound(remaining, current_weight):
            # Simple: sum of all remaining weights
            return current_weight + sum(weights[i] for i in remaining)
        
        def branch(idx, current_set, remaining):
            nonlocal best_set, best_weight
            
            if time.time() > deadline - 5.0:
                return
            
            if not remaining:
                if weight_of_set(current_set) > best_weight:
                    best_set = set(current_set)
                    best_weight = weight_of_set(current_set)
                return
            
            # Upper bound check
            ub = current_weight + sum(weights[i] for i in remaining)
            if ub <= best_weight + 1e-15:
                return
            
            # Try adding first vertex in order
            v = order[idx] if idx < len(order) else None
            
            if v is None or v not in remaining:
                # Find next available
                for i in range(idx, len(order)):
                    if order[i] in remaining:
                        v = order[i]
                        idx = i
                        break
                if v is None or v not in remaining:
                    return
            
            # Option 1: include v
            new_remaining = set(remaining)
            new_remaining.discard(v)
            # Remove all neighbors of v
            for nb in adj[v]:
                new_remaining.discard(nb)
            
            branch(idx + 1, current_set | {v}, new_remaining)
            
            # Option 2: exclude v
            new_remaining2 = set(remaining)
            new_remaining2.discard(v)
            branch(idx + 1, current_set, new_remaining2)
        
        # Start branch and bound
        initial_remaining = set(range(n))
        branch(0, set(), initial_remaining)
        
        return best_set
    
    # Strategy: Start with greedy, then local search, then try B&B if time permits
    best_solution = greedy_init()
    best_solution = local_search(best_solution, time_limit=deadline - 10.0)
    best_weight = weight_of_set(best_solution)
    
    # Try multiple random restarts of greedy + local search
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline - 15.0:
            break
        
        # Random perturbation: shuffle order with bias toward high weights
        order = list(range(n))
        random.shuffle(order)
        # Sort by: -weight + random noise
        noisy_order = sorted(order, key=lambda i: -weights[i] + random.random() * 0.01)
        
        candidate = greedy_init(noisy_order)
        candidate = local_search(candidate, time_limit=deadline - 10.0)
        
        if weight_of_set(candidate) > best_weight:
            best_weight = weight_of_set(candidate)
            best_solution = candidate
    
    # Try branch and bound if time remains
    if time.time() < deadline - 20.0:
        bb_solution = branch_and_bound()
        if weight_of_set(bb_solution) > best_weight:
            best_weight = weight_of_set(bb_solution)
            best_solution = bb_solution
    
    # Convert back to numbers
    result = [numbers[i] for i in sorted(best_solution)]
    
    # Final validation
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.lcm(result[i], result[j]) <= 256:
                # Fallback: use the known valid set
                fallback = [2] + [x for x in range(129, 257) if x % 2 != 0]
                return fallback
    
    return result

