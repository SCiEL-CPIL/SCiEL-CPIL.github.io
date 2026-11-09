"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _lagrangian_relaxation_tour(distance_matrix, demands, capacity, max_iters=50):
    """
    Solve relaxed CVRP via Lagrangian relaxation.
    Returns edge weights derived from dual optimization.
    """
    n = distance_matrix.shape[0]
    d0 = distance_matrix[0, :]
    
    # Lagrange multiplier for capacity constraint
    # We penalize routes that exceed capacity
    lambda_dual = 0.0
    step_size = 0.5
    
    best_edge_scores = np.zeros((n, n))
    
    for iteration in range(max_iters):
        # Solve relaxed TSP: minimize distance + lambda * max(0, load - capacity)
        # Approximate by adjusting edge costs based on demand penalty
        # For edge (i,j), the marginal cost includes demand of j
        # Adjusted cost: d[i,j] + lambda * demand[j]
        
        adjusted_costs = distance_matrix.copy()
        # Add demand penalty to each edge based on destination demand
        for j in range(1, n):
            adjusted_costs[:, j] += lambda_dual * demands[j]
        
        # Solve nearest-neighbor tour from depot as approximation of relaxed TSP
        # This gives us a feasible tour structure in the relaxed space
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        tour_edges = []
        
        for _ in range(n - 1):
            # Find nearest unvisited node
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                break
            costs_to_unvisited = adjusted_costs[current, unvisited]
            next_node = unvisited[np.argmin(costs_to_unvisited)]
            tour_edges.append((current, next_node))
            visited[next_node] = True
            current = next_node
        
        # Return to depot
        tour_edges.append((current, 0))
        
        # Score edges based on how often they appear and their quality
        for i, j in tour_edges:
            if i != j:
                # Edge score: lower adjusted cost is better
                # Also consider if this edge helps with capacity
                edge_quality = 1.0 / (adjusted_costs[i, j] + 1e-9)
                best_edge_scores[i, j] += edge_quality
        
        # Update Lagrange multiplier based on capacity violation
        # Approximate total load in the tour
        # For simplicity, estimate if demands are balanced
        avg_demand = np.mean(demands[1:])
        if avg_demand * (n - 1) > capacity:
            lambda_dual = max(0, lambda_dual + step_size)
        else:
            lambda_dual = max(0, lambda_dual - step_size * 0.5)
    
    # Normalize edge scores
    max_score = np.max(best_edge_scores)
    if max_score > 0:
        best_edge_scores = best_edge_scores / max_score
    else:
        best_edge_scores = np.ones((n, n)) / n
    
    return best_edge_scores


def _construct_reference_solution(distance_matrix, demands, capacity):
    """
    Construct a reference solution using greedy insertion with local search.
    """
    n = distance_matrix.shape[0]
    d0 = distance_matrix[0, :]
    
    # Start with all customers unassigned
    unassigned = list(range(1, n))
    routes = []
    
    while unassigned:
        # Start new route from depot
        current = 0
        remaining_cap = capacity
        route = [0]
        
        while unassigned:
            # Find feasible customer with best savings
            best_customer = -1
            best_saving = -np.inf
            best_dist = np.inf
            
            for cust in unassigned:
                if demands[cust] <= remaining_cap:
                    dist = distance_matrix[current, cust]
                    # Savings: d[0,cust] + d[current,0] - d[current,cust]
                    saving = d0[cust] + distance_matrix[current, 0] - dist
                    if saving > best_saving or (saving == best_saving and dist < best_dist):
                        best_saving = saving
                        best_customer = cust
                        best_dist = dist
            
            if best_customer == -1:
                break
            
            route.append(best_customer)
            remaining_cap -= demands[best_customer]
            current = best_customer
            unassigned.remove(best_customer)
        
        # Return to depot
        route.append(0)
        routes.append(route)
    
    return routes


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix using Lagrangian Relaxation-based weights.
    Returns a matrix of the same shape as distance_matrix where larger values
    indicate more attractive edges.
    """
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)

    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    # Base heuristic: inverse distance
    base = 1.0 / (distance_matrix + 1e-9)
    
    # Step 1: Lagrangian Relaxation edge scores
    lr_scores = _lagrangian_relaxation_tour(distance_matrix, demands, capacity, max_iters=30)
    
    # Step 2: Reference solution edge scores
    ref_routes = _construct_reference_solution(distance_matrix, demands, capacity)
    ref_edge_scores = np.zeros((n, n))
    for route in ref_routes:
        for k in range(len(route) - 1):
            i, j = route[k], route[k + 1]
            if i != j:
                ref_edge_scores[i, j] += 1.0 / (distance_matrix[i, j] + 1e-9)
    
    max_ref = np.max(ref_edge_scores)
    if max_ref > 0:
        ref_edge_scores = ref_edge_scores / max_ref
    
    # Step 3: Combine Lagrangian scores with reference solution scores
    # Weight: 60% Lagrangian, 40% reference
    combined_scores = 0.6 * lr_scores + 0.4 * ref_edge_scores
    
    # Step 4: Add demand-aware adjustment
    # Prefer edges to customers with lower demand (more capacity remaining)
    max_demand = np.max(demands[1:]) if n > 1 else 1.0
    if max_demand > 0:
        demands_norm = demands / max_demand
    else:
        demands_norm = np.zeros(n)
    
    # Edges to lower-demand customers are slightly more attractive
    demand_factor = 1.0 + 0.2 * (1.0 - demands_norm[None, :])
    demand_factor[0, :] = 1.0
    demand_factor[:, 0] = 1.0
    
    # Step 5: Final heuristic = base * combined_scores * demand_factor
    heuristic = base * combined_scores * demand_factor
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Verify shape matches input
    assert heuristic.shape == distance_matrix.shape, "Shape mismatch"
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - start_time:
        # Return best available
        pass
    
    return heuristic
# EVOLVE-BLOCK-END
