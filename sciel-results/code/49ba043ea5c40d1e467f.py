"""Traveling salesman constructive heuristic with global connectivity-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None


def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)


def _mst_weight(nodes, distance_matrix):
    """Compute MST weight of the given nodes using Prim's algorithm."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    submat = distance_matrix[np.ix_(nodes, nodes)]
    in_mst = [False] * n
    in_mst[0] = True
    min_dist = submat[0].copy()
    mst_weight = 0.0
    
    for _ in range(n - 1):
        min_dist[in_mst] = np.inf
        u = np.argmin(min_dist)
        mst_weight += min_dist[u]
        in_mst[u] = True
        min_dist = np.minimum(min_dist, submat[u])
    
    return mst_weight


def _two_opt_improve(tour, distance_matrix, max_iter=10):
    """Apply a few rounds of 2-opt to improve the tour."""
    n = len(tour)
    if n < 4:
        return tour
    
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                # Check if reversing segment [i+1..j] improves
                old_cost = distance_matrix[a, b] + distance_matrix[c, d]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d]
                if new_cost < old_cost - 1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    improved = True
    
    return tour


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 0.5:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
    
    # For small n, use exact or near-exact evaluation
    if n <= 4:
        best_score = float('inf')
        best_node = candidates[0]
        
        for i, c in enumerate(candidates):
            remaining = np.delete(candidates, i)
            
            # Try all permutations of remaining nodes (small enough)
            from itertools import permutations
            best_completion = float('inf')
            for perm in permutations(remaining):
                total = distance_matrix[current_node, c]
                prev = c
                for node in perm:
                    total += distance_matrix[prev, node]
                    prev = node
                total += distance_matrix[prev, destination_node]
                if total < best_completion:
                    best_completion = total
            
            if best_completion < best_score:
                best_score = best_completion
                best_node = c
        
        return int(best_node)
    
    # For larger n, use a hybrid heuristic:
    # 1. Greedy nearest neighbor with a lookahead penalty
    # 2. Global connectivity via MST lower bound
    # 3. Penalty for creating "bottlenecks" (nodes with high degree in potential subtour)
    
    dist_from_current = distance_matrix[current_node, candidates]
    dist_to_dest = distance_matrix[candidates, destination_node]
    
    # Compute a quick MST-based lower bound for the full remaining set
    full_mst = _mst_weight(candidates, distance_matrix)
    
    best_score = float('inf')
    best_node = candidates[0]
    
    for i, c in enumerate(candidates):
        cost_in = dist_from_current[i]
        
        if n == 2:
            other_node = candidates[1 - i]
            cost_total = distance_matrix[current_node, c] + distance_matrix[c, other_node] + distance_matrix[other_node, destination_node]
            total_score = cost_total
        else:
            remaining = np.delete(candidates, i)
            
            # MST of remaining nodes
            rem_mst = _mst_weight(remaining, distance_matrix)
            
            # Min entry from c to remaining
            min_entry_from_c = np.min(distance_matrix[c, remaining])
            
            # Min exit from remaining to destination
            min_exit = np.min(distance_matrix[remaining, destination_node])
            
            # Penalty: how "central" is c in the remaining structure?
            # If c is far from many remaining nodes, it might create a long detour
            # Use the second-cheapest distance from c to remaining as a proxy for detour cost
            dists_from_c = distance_matrix[c, remaining]
            sorted_dists = np.sort(dists_from_c)
            # If the second closest is much farther than the closest, c might be an outlier
            detour_penalty = max(0.0, sorted_dists[1] - sorted_dists[0]) * 0.3 if len(dists_from_c) >= 2 else 0.0
            
            # Connectivity penalty: if removing c disconnects the MST significantly
            # Approximate by comparing full_mst vs rem_mst + min_entry + min_exit
            # A large gap suggests c is a critical connector
            connectivity_gap = full_mst - (rem_mst + min_entry_from_c + min_exit)
            connectivity_penalty = max(0.0, connectivity_gap) * 0.5
            
            total_score = cost_in + min_entry_from_c + rem_mst + min_exit + detour_penalty + connectivity_penalty
        
        if total_score < best_score:
            best_score = total_score
            best_node = c
    
    return int(best_node)
# EVOLVE-BLOCK-END
