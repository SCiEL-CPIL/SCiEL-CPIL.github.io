import time

# EVOLVE-BLOCK-START
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Explicit deadline guard
    # If we are close to the limit, fail safe by closing the route
    if time.time() - _start_time > 145.0:
        return depot

    # Identify feasible nodes
    feasible_nodes = []
    for node in unvisited_nodes:
        if demands[node] <= rest_capacity:
            feasible_nodes.append(node)
            
    if not feasible_nodes:
        return depot

    # Heuristic Selection
    if current_node == depot:
        # Start route: Nearest Neighbor to ensure we enter the "field" efficiently
        best_node = min(feasible_nodes, key=lambda n: distance_matrix[current_node, n])
    else:
        # Continue route: Clarke-Wright Savings Heuristic
        # Maximize savings: d(curr, depot) + d(depot, next) - d(curr, next)
        # This favors nodes that are "in the same direction" from the depot
        dist_curr_depot = distance_matrix[current_node, depot]
        best_node = feasible_nodes[0]
        best_savings = -1.0
        best_dist = distance_matrix[current_node, best_node]
        
        for node in feasible_nodes:
            # Calculate savings
            s = dist_curr_depot + distance_matrix[depot, node] - distance_matrix[current_node, node]
            
            if s > best_savings:
                best_savings = s
                best_node = node
                best_dist = distance_matrix[current_node, node]
            elif s == best_savings:
                # Tie-break: choose closer node
                if distance_matrix[current_node, node] < best_dist:
                    best_node = node
                    best_dist = distance_matrix[current_node, node]
                    
    return int(best_node)
# EVOLVE-BLOCK-END
