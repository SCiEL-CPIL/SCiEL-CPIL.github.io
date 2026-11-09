import numpy as np

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # If no unvisited nodes, return to depot
    if len(unvisited_nodes) == 0:
        return depot

    # Get demands for unvisited nodes
    # demands is an array where demands[i] is the demand of node i
    # unvisited_nodes is an array/list of node indices
    
    # Filter feasible candidates based on capacity
    # demands[unvisited_nodes] gives the demands of the unvisited nodes
    # A node v is feasible if demands[v] <= rest_capacity
    
    unvisited_arr = np.array(unvisited_nodes)
    demands_unvisited = demands[unvisited_arr]
    feasible_mask = demands_unvisited <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # Calculate distance from current to all feasible nodes
    dists_current = distance_matrix[current_node, feasible_nodes]
    
    # Initialize best cost and best node
    best_cost = np.inf
    best_node = feasible_nodes[0]
    
    # For each feasible node, perform 1-step lookahead
    # Cost = dist(current, v) + min( dist(v, u) for u in feasible(v), or dist(v, depot) if none )
    
    # We can vectorize the "dist(v, u)" part somewhat, but the feasibility of u depends on v's demand.
    # Let's iterate over feasible_nodes. N is small (<=50), so a loop is fine.
    
    for v in feasible_nodes:
        c1 = distance_matrix[current_node, v]
        
        # Check if we can return to depot (always possible if capacity allows, which it does)
        # But we might have more customers.
        
        # New rest capacity after visiting v
        new_rest = rest_capacity - demands[v]
        
        # Find feasible nodes from v
        # Candidates are unvisited nodes excluding v
        # We need to check capacity for them: demands[u] <= new_rest
        
        # To avoid modifying unvisited_nodes (which is passed as arg), we just filter the original list/array
        # We already have unvisited_arr and demands_unvisited
        # We need to exclude v from the candidates for the lookahead step
        
        # Create a mask for nodes u in unvisited_nodes where u != v and demands[u] <= new_rest
        # Note: unvisited_arr contains all unvisited nodes.
        
        # Efficiently check feasibility for all unvisited nodes
        # demands_unvisited <= new_rest
        # Also u != v
        
        # Since we are inside a loop, let's just use the arrays we have
        # We need to be careful with indexing.
        
        # Let's pre-compute distances from v to all unvisited nodes
        dists_v_to_unvisited = distance_matrix[v, unvisited_arr]
        
        # Feasibility mask for lookahead step
        # 1. Capacity: demands_unvisited <= new_rest
        # 2. Not the node v itself
        # We can create a mask
        
        cap_mask = demands_unvisited <= new_rest
        # Exclude v
        # Since unvisited_arr might have duplicates? No, it's a set of unvisited nodes.
        # We need to mask out the index of v in unvisited_arr
        # v is in unvisited_arr. Find its index.
        
        # To make this fast, we can just iterate if N is small, but let's try vectorization.
        # Actually, since N<=50, a simple loop inside the loop is O(N^2) which is 2500 ops.
        # The outer loop is also N. So total O(N^2) per select call.
        # 50 * 50 = 2500 distance lookups and comparisons. Very fast.
        
        min_next_cost = np.inf
        # Iterate through unvisited nodes to find best next step
        # We can just use the vectorized dists_v_to_unvisited and cap_mask
        
        # Create a combined mask
        # We need to handle the case where v is in unvisited_arr.
        # Let's find the index of v in unvisited_arr to exclude it.
        # This is O(N).
        
        # Alternative: Just check all u in unvisited_arr.
        # If u == v, skip.
        # If demands[u] > new_rest, skip.
        # Cost = dist(v, u)
        
        # We can do this with a list comprehension or simple loop
        # Given the small N, a simple loop is clear and fast enough.
        
        for u in unvisited_arr:
            if u == v:
                continue
            if demands[u] > new_rest:
                continue
            cost = distance_matrix[v, u]
            if cost < min_next_cost:
                min_next_cost = cost
        
        # If no feasible u found, we must return to depot
        if min_next_cost == np.inf:
            min_next_cost = distance_matrix[v, depot]
            
        total_cost = c1 + min_next_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = v
            
    return int(best_node)
