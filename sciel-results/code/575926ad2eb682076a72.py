import math
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 256
    max_sum = 2 * n
    
    # Perfect squares up to max_sum
    squares = set()
    i = 1
    while i * i <= max_sum:
        squares.add(i * i)
        i += 1
        
    # Step 1: Filter candidates based on diagonal constraint (2*a not square)
    candidates = [a for a in range(1, n + 1) if 2 * a not in squares]
    cand_set = set(candidates)
    num_candidates = len(candidates)
    
    # Build adjacency list for the conflict graph
    # We map original values to indices 0..num_candidates-1 for efficiency
    val_to_idx = {v: i for i, v in enumerate(candidates)}
    adj = [set() for _ in range(num_candidates)]
    
    # Efficiently build conflicts
    # For each a, check all b > a
    for i, a in enumerate(candidates):
        for j in range(i + 1, num_candidates):
            b = candidates[j]
            if a + b in squares:
                adj[i].add(j)
                adj[j].add(i)
                
    # Degrees for ordering
    degrees = [len(adj[i]) for i in range(num_candidates)]
    
    # Initial Greedy Solution (Min-Degree)
    def greedy_min_degree():
        pool = set(range(num_candidates))
        selected = []
        while pool:
            min_deg_node = None
            min_deg = float('inf')
            for node in pool:
                deg = len(adj[node] & pool)
                if deg < min_deg:
                    min_deg = deg
                    min_deg_node = node
                    if min_deg == 0:
                        break
            if min_deg_node is None:
                break
            selected.append(min_deg_node)
            pool.discard(min_deg_node)
            pool.difference_update(adj[min_deg_node])
        return selected

    best_solution = greedy_min_degree()
    best_size = len(best_solution)
    
    # Randomized Greedy to potentially improve initial bound
    for seed in range(20):
        random.seed(seed)
        pool = set(range(num_candidates))
        selected = []
        while pool:
            degrees_pool = {node: len(adj[node] & pool) for node in pool}
            min_deg = min(degrees_pool.values())
            candidates_for_pick = [node for node, deg in degrees_pool.items() if deg == min_deg]
            node = random.choice(candidates_for_pick)
            selected.append(node)
            pool.discard(node)
            pool.difference_update(adj[node])
        if len(selected) > best_size:
            best_solution = selected
            best_size = len(selected)

    # Branch and Bound
    # Reorder vertices by descending degree to improve coloring bound
    order = sorted(range(num_candidates), key=lambda x: degrees[x], reverse=True)
    # Map old index to new index
    old_to_new = {old: new for new, old in enumerate(order)}
    new_adj = [set() for _ in range(num_candidates)]
    for i in range(num_candidates):
        for j in adj[i]:
            new_adj[old_to_new[i]].add(old_to_new[j])
    
    # Convert to lists for faster iteration if needed, but sets are good for intersection
    # Greedy Coloring for Upper Bound
    # Colors are 1..k. We want to find the minimum number of colors (chromatic number approximation)
    # Actually, for MIS, the bound is the number of colors in a greedy coloring of the *candidate* subgraph.
    # A proper k-coloring implies the independence number is at most k.
    
    def greedy_color_bound(candidates_set):
        # candidates_set is a set of node indices
        if not candidates_set:
            return 0
        
        # Order candidates by degree within the subgraph (descending) for better coloring
        # Approximation: just use global degree or sort by len(adj[node] & candidates_set)
        # Sorting at every step is expensive. We'll use a simple greedy:
        # Pick a node, assign color 1, remove it and its neighbors? No, coloring is different.
        
        # Standard greedy coloring:
        # 1. Order vertices (e.g., by degree in subgraph)
        # 2. For each vertex, assign the smallest available color not used by neighbors already colored.
        
        # To make it fast, we don't re-sort every time. We can just iterate in the fixed 'order' 
        # but filter for those in candidates_set.
        
        colors = {}
        available_colors = [True] * (len(candidates_set) + 1) # Max colors is size of set
        # Actually, max colors is size of set.
        
        # Let's try a simpler bound: 
        # If we color the graph, the max independent set size <= number of colors.
        # Greedy coloring:
        # Sort candidates by degree in subgraph descending
        sub_graph_nodes = list(candidates_set)
        # Sorting is O(N log N). N is at most 256.
        sub_graph_nodes.sort(key=lambda x: len(adj[x] & candidates_set), reverse=True)
        
        color_of = {}
        num_colors = 0
        for node in sub_graph_nodes:
            # Find smallest color not used by neighbors
            used_colors = set()
            for neighbor in adj[node]:
                if neighbor in candidates_set and neighbor in color_of:
                    used_colors.add(color_of[neighbor])
            
            c = 1
            while c in used_colors:
                c += 1
            color_of[node] = c
            if c > num_colors:
                num_colors = c
                
        return num_colors

    # B&B State
    # We maintain a list of candidates.
    # To speed up, we can use bitsets if N <= 256, but Python sets are okay for N=256.
    
    global best_solution_indices
    global best_size_global
    
    best_solution_indices = best_solution
    best_size_global = best_size
    
    def branch_and_bound(candidates_set, current_solution):
        nonlocal best_solution_indices
        nonlocal best_size_global
        
        if time.time() - start_time > deadline:
            return
            
        if not candidates_set:
            if len(current_solution) > best_size_global:
                best_size_global = len(current_solution)
                best_solution_indices = list(current_solution)
            return
            
        # Upper bound
        ub = greedy_color_bound(candidates_set)
        if len(current_solution) + ub <= best_size_global:
            return
            
        # Pick a node to branch on
        # Heuristic: pick node with highest degree in subgraph
        node = None
        max_deg = -1
        for n in candidates_set:
            deg = len(adj[n] & candidates_set)
            if deg > max_deg:
                max_deg = deg
                node = n
                
        # Branch 1: Include node
        # New candidates: candidates_set - {node} - adj[node]
        new_candidates_1 = candidates_set - {node} - adj[node]
        current_solution.append(node)
        branch_and_bound(new_candidates_1, current_solution)
        current_solution.pop()
        
        # Branch 2: Exclude node
        # New candidates: candidates_set - {node}
        new_candidates_2 = candidates_set - {node}
        branch_and_bound(new_candidates_2, current_solution)
        
    # Start B&B
    initial_candidates = set(range(num_candidates))
    branch_and_bound(initial_candidates, [])
    
    # Convert indices back to values
    final_solution = [candidates[i] for i in best_solution_indices]
    
    # Final verification
    sol_set = set(final_solution)
    for a in sol_set:
        if 2 * a in squares:
            return list(range(1, 10)) # Fallback
        for b in sol_set:
            if a <= b and a + b in squares:
                return list(range(1, 10)) # Fallback
                
    return final_solution

