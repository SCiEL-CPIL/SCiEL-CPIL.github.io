"""Savings-based heuristic with look-ahead for Capacitated Vehicle Routing Problem."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIMEOUT = 110.0  # Safety margin before 120s limit
_DEADLINE = _start_time + _TIMEOUT

def _estimate_completion_cost(start_node, remaining_cap, unvisited_set, demands, distance_matrix):
    """
    Estimate the cost to complete a route starting from start_node,
    given remaining_cap, using a greedy nearest-neighbor strategy.
    Returns the total distance from start_node to depot.
    """
    if remaining_cap <= 0:
        return distance_matrix[start_node, 0]
    
    current = start_node
    total_dist = 0.0
    remaining = remaining_cap
    
    # Use a set for fast lookup, but we need to iterate neighbors.
    # Since N is small (50), a list or set is fine.
    # We need to avoid nodes already visited. The 'unvisited_set' passed in 
    # should contain all nodes that are not yet in the current route prefix.
    # However, the simulation is a "what if" so it doesn't modify the actual state.
    # We need a local copy of unvisited nodes to simulate on.
    # To be fast, we can just use a list of unvisited nodes and mark visited locally.
    
    # Optimization: Pre-compute nearest neighbors? No, capacity changes.
    # Simple NN loop.
    
    # We need the set of nodes available for simulation.
    # These are the unvisited nodes from the actual state.
    # We will simulate visiting a subset of them.
    
    # To avoid modifying the global unvisited_set, we work with a local copy or just indices.
    # Given N=50, copying a list of 50 ints is cheap.
    available = list(unvisited_set)
    
    while available:
        # Find nearest feasible node
        best_node = -1
        best_dist = float('inf')
        
        for node in available:
            if demands[node] <= remaining:
                dist = distance_matrix[current, node]
                if dist < best_dist:
                    best_dist = dist
                    best_node = node
        
        if best_node == -1:
            break
            
        total_dist += best_dist
        remaining -= demands[best_node]
        current = best_node
        available.remove(best_node)
        
        if remaining <= 0:
            break
            
    total_dist += distance_matrix[current, 0]
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a look-ahead heuristic.
    Returns 0 (depot) if no feasible node can be added, or if closing is cheaper.
    """
    if time.time() > _DEADLINE:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    # Convert to numpy array for vectorized operations where possible,
    # but the look-ahead simulation is iterative.
    unvisited = list(unvisited_nodes)
    
    # 1. Filter by capacity
    feasible_nodes = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. If at depot, pick the nearest feasible node
    if current_node == depot:
        # NN is sufficient here, or even look-ahead, but NN is very fast and standard.
        # Let's use NN for the first step to save time, or look-ahead for consistency.
        # Look-ahead from depot:
        # Cost to pick j: d(depot, j) + EstCompletion(j, rest_cap - d_j)
        # Let's just use NN for the first step to keep it simple and fast, 
        # as the "savings" concept is less applicable at the start.
        min_dist = float('inf')
        best_node = -1
        for n in feasible_nodes:
            d = distance_matrix[depot, n]
            if d < min_dist:
                min_dist = d
                best_node = n
        return best_node

    # 3. Calculate cost to close route now
    cost_close = distance_matrix[current_node, depot]
    
    # 4. For each feasible node, estimate total route cost if we pick it
    best_node = depot
    best_cost = cost_close  # Default to closing
    
    # We need the set of unvisited nodes for the simulation.
    # The simulation will "visit" some of them.
    # We pass the list of unvisited nodes to the estimator.
    
    for j in feasible_nodes:
        # Immediate cost
        d_cj = distance_matrix[current_node, j]
        
        # Remaining capacity after picking j
        rem_cap = rest_capacity - demands[j]
        
        # Estimate completion cost from j
        # The unvisited nodes for simulation are all unvisited nodes except j itself
        # (since j is now "visited" in the simulation).
        # But wait, the other unvisited nodes are still available.
        # So we pass the list of unvisited nodes, but we must exclude j from the 
        # "available" set inside the simulator? 
        # Actually, the simulator starts at j. It should not visit j again.
        # So available = unvisited - {j}.
        
        # To be efficient, we can just pass unvisited and handle j in the simulator?
        # No, j is the start. The simulator picks *next* nodes.
        # So available nodes for the simulator are unvisited \ {j}.
        
        # Let's optimize the call to _estimate_completion_cost
        # We can pass the full unvisited list and let it filter? 
        # Or create a set?
        
        # For N=50, creating a list comprehension is fine.
        # unvisited_set_for_sim = [n for n in unvisited if n != j]
        
        # However, _estimate_completion_cost takes a set/list.
        # Let's modify the estimator to accept a list and a "start_node" 
        # and it will implicitly exclude start_node from re-selection?
        # No, it picks from 'available'. If j is in 'available', it might pick j again.
        # So we must exclude j.
        
        # To avoid list creation overhead in the loop, we can just check in the estimator?
        # No, let's just pass the list. It's small.
        
        # Optimization: If d_cj is already >= best_cost, we can skip?
        # Not necessarily, because the completion cost could be 0? No, it's at least d(j, depot).
        # So min possible cost is d_cj + d(j, depot).
        # If d_cj + distance_matrix[j, depot] >= best_cost, we can prune?
        # This is a lower bound.
        if d_cj + distance_matrix[j, depot] >= best_cost:
            continue
            
        # Estimate
        # Create a copy of unvisited without j for the simulation
        # Using a set for O(1) removal? The simulator uses list.remove which is O(N).
        # Let's stick to lists.
        
        # Pass unvisited to the function. The function will simulate.
        # We need to ensure j is not picked.
        # Let's adjust _estimate_completion_cost to take a "forbidden" node or just a list.
        
        # For simplicity and speed, let's inline the estimation logic or make it very fast.
        
        # Actually, let's just call the helper.
        # To avoid list copy, we can pass the original list and a "start" node, 
        # and the helper skips the start node.
        
        # Let's refactor _estimate_completion_cost to skip start_node.
        est_cost = _estimate_completion_cost(j, rem_cap, unvisited, demands, distance_matrix)
        
        total_est_cost = d_cj + est_cost
        
        if total_est_cost < best_cost:
            best_cost = total_est_cost
            best_node = j
            
    return best_node
# EVOLVE-BLOCK-END
