"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Computes vertex priorities using an iterative simulated greedy approach.
    
    Instead of a static score, this function simulates the greedy decoding process.
    In each iteration, it selects the vertex that currently has the lowest 'cost' 
    (degree + sum of neighbor degrees), removes it and its neighbors from the active 
    set, and updates the scores of remaining vertices. The final score for each vertex 
    is determined by the step at which it would have been selected (earlier is better).
    
    This adapts to the changing graph structure, which is crucial for Erdos-Renyi graphs 
    where local density fluctuations matter.
    """
    start_time = time.time()
    DEADLINE = 120.0  # Safety margin from 150s limit
    
    # Convert to numpy boolean array for fast operations
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = adjacency.shape[0]
    # Work with a copy of the adjacency matrix as the "active" graph
    active_adj = adjacency.copy()
    
    # Initialize scores. We will assign scores based on selection order.
    # Higher score = selected earlier.
    scores = np.zeros(n, dtype=float)
    unselected = np.ones(n, dtype=bool)
    
    # Precompute initial degrees for speedup if we were doing static, but here we update.
    # To keep it efficient, we maintain a set of active nodes.
    active_nodes = list(range(n))
    active_set = set(active_nodes)
    
    # We need to efficiently find the best vertex at each step.
    # Naive O(N^2) might be too slow for N=200 if done in pure Python loops repeatedly?
    # N=200, N steps. Each step involves scanning N nodes and calculating cost.
    # Cost calculation: deg(u) + sum(deg(v) for v in neighbors(u)).
    # If we do this in Python loops, 200 * 200 * (avg_deg) ~ 200*200*10 = 400k ops. Very fast.
    
    # However, we must update degrees of remaining neighbors when a node is removed.
    # Let's maintain current_degrees array.
    current_degrees = active_adj.sum(axis=1).astype(int)
    
    selected_count = 0
    # We want to assign high scores to early selections.
    # Score = N - step_index.
    
    for step in range(n):
        if time.time() - start_time > DEADLINE:
            # Fallback: assign random scores to unselected to finish
            mask = unselected
            if mask.any():
                scores[mask] = np.random.rand(mask.sum()) * 100
            break
            
        if not unselected.any():
            break
            
        # Find the best vertex among unselected
        best_idx = -1
        best_cost = float('inf')
        
        # Iterate over unselected vertices
        # Optimization: Only check vertices that are still in active_set
        for u in active_nodes:
            if not unselected[u]:
                continue
                
            # Calculate cost for u
            # Cost = current_degrees[u] + sum(current_degrees[v] for v in neighbors(u) if unselected[v])
            deg_u = current_degrees[u]
            
            # Get neighbors of u in the ORIGINAL graph
            # But we only care about neighbors that are still UNSELECTED
            # Note: active_adj is the original graph. 
            # We need to sum degrees of neighbors that are still in the game.
            
            # Vectorized sum of degrees of unselected neighbors
            # neighbors = np.where(active_adj[u])[0]
            # This creates an array each time.
            
            # Alternative: Precompute neighbor lists?
            # For N=200, loop is fine.
            
            # Sum degrees of unselected neighbors
            # We can use the current_degrees array and a mask of unselected neighbors
            # But we need to be careful: current_degrees reflects the degree in the *remaining* graph?
            # No, current_degrees was initialized from the full graph.
            # When we remove a node, we MUST decrement the degrees of its neighbors.
            
            # Let's re-evaluate the state maintenance.
            # State:
            # 1. unselected: bool array
            # 2. current_degrees: int array, degree of each node in the subgraph induced by unselected nodes.
            
            # If current_degrees is maintained correctly:
            # cost(u) = current_degrees[u] + sum(current_degrees[v] for v in neighbors(u) if unselected[v])
            
            # To get sum of degrees of unselected neighbors:
            # We can sum current_degrees[neighbors_u] where neighbors_u are those that are unselected.
            
            # Let's fetch neighbors of u from original adjacency
            # Since we removed nodes, some neighbors might be selected.
            # But current_degrees[selected_node] is 0? 
            # If we decrement neighbors when a node is selected, and set selected node's degree to 0,
            # then summing current_degrees over ALL original neighbors of u will automatically 
            # exclude selected neighbors (since their degree is 0).
            
            # Is this true?
            # If v is selected, current_degrees[v] becomes 0.
            # If v is unselected, current_degrees[v] is its current degree.
            # So sum(current_degrees[original_neighbors_of_u]) == sum(current_degrees[v] for v in neighbors(u) if unselected[v])
            
            # This is a huge optimization. We don't need to mask neighbors.
            
            neighbors_u = np.where(active_adj[u])[0]
            neighbor_deg_sum = current_degrees[neighbors_u].sum()
            
            cost = deg_u + neighbor_deg_sum
            
            if cost < best_cost:
                best_cost = cost
                best_idx = u
                
        if best_idx == -1:
            break
            
        # Select best_idx
        scores[best_idx] = n - step  # Higher score for earlier selection
        
        # Update state
        unselected[best_idx] = False
        current_degrees[best_idx] = 0
        
        # Decrement degrees of neighbors of best_idx that are still unselected
        # Again, we can just decrement all neighbors, but we must not decrement selected ones 
        # (though their degree is already 0, so decrementing would make it negative? 
        # No, we only care about unselected neighbors for future cost calculations?
        # Actually, if we decrement a selected neighbor's degree, it becomes negative.
        # Does that matter? 
        # In the cost calculation for another node w, we sum current_degrees[neighbors_w].
        # If a neighbor v is selected, current_degrees[v] should be 0.
        # If we decrement it, it becomes -1.
        # Then cost(w) includes -1. This is WRONG.
        
        # So we MUST only decrement degrees of UNSELECTED neighbors.
        
        neighbors_best = np.where(active_adj[best_idx])[0]
        # Mask for unselected neighbors
        unselected_neighbors = neighbors_best[unselected[neighbors_best]]
        
        # Decrement their degrees
        current_degrees[unselected_neighbors] -= 1
        
        # Remove best_idx from active_nodes list for iteration speedup?
        # active_nodes is a list. Removing is O(N). N=200 is small.
        # But we are iterating over active_nodes in the inner loop.
        # If we don't remove, we check `if not unselected[u]` which is O(1).
        # So we can keep active_nodes static and just rely on the unselected mask.
        
        selected_count += 1

    # If we didn't finish due to time, the scores for unselected are 0.
    # We should probably give them a low score, which 0 is.
    # But to be safe, if time ran out, we might want to randomize the rest?
    # The deadline guard breaks the loop. Unselected nodes have score 0.
    # This is a valid fallback.
    
    return scores
# EVOLVE-BLOCK-END
