# EVOLVE-BLOCK-START
import numpy as np
import time

def _construct_nn_routes(distance_matrix, demands, capacity, n):
    """
    Nearest-neighbor constructive heuristic with capacity constraint.
    Returns list of routes (each route is a list of node indices, excluding depot).
    """
    unvisited = set(range(1, n))
    routes = []
    
    while unvisited:
        current = 0
        remaining_cap = capacity
        route = []
        
        while True:
            if not unvisited:
                break
            
            # Find nearest unvisited customer that fits in remaining capacity
            best_node = -1
            best_dist = float('inf')
            for j in unvisited:
                if demands[j] <= remaining_cap:
                    d = distance_matrix[current, j]
                    if d < best_dist:
                        best_dist = d
                        best_node = j
            
            if best_node == -1:
                break
            
            route.append(best_node)
            remaining_cap -= demands[best_node]
            unvisited.remove(best_node)
            current = best_node
        
        if route:
            routes.append(route)
    
    return routes


def _two_opt_route(route, distance_matrix, max_iters=50):
    """
    Apply 2-opt improvement to a single route (sequence of customer nodes).
    Returns improved route.
    """
    if len(route) < 2:
        return route
    
    route = list(route)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(len(route) - 1):
            for j in range(i + 1, len(route)):
                # Compute delta for reversing segment i+1..j
                # Current edges: (route[i], route[i+1]) and (route[j], route[j+1]) if j+1 exists
                # New edges: (route[i], route[j]) and (route[i+1], route[j+1]) if j+1 exists
                a = route[i]
                b = route[i + 1]
                c = route[j]
                d = route[j + 1] if j + 1 < len(route) else None
                
                old_cost = distance_matrix[a, b] + distance_matrix[c, d] if d is not None else distance_matrix[a, b]
                new_cost = distance_matrix[a, c] + distance_matrix[b, d] if d is not None else distance_matrix[a, c]
                
                if new_cost < old_cost - 1e-10:
                    route[i + 1:j + 1] = route[i + 1:j + 1][::-1]
                    improved = True
    
    return route


def _solve_cvrp_reference(distance_matrix, coordinates, demands, capacity, n, time_limit=5.0):
    """
    Solve CVRP using NN construction + 2-opt, with time limit.
    Returns list of routes (each route is a list of customer nodes).
    """
    start_time = time.time()
    routes = _construct_nn_routes(distance_matrix, demands, capacity, n)
    
    # Apply 2-opt to each route
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx, route in enumerate(routes):
            if (time.time() - start_time) >= time_limit:
                break
            new_route = _two_opt_route(route, distance_matrix, max_iters=20)
            if new_route != route:
                routes[idx] = new_route
                improved = True
    
    return routes


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes an edge-attractiveness matrix for ACO-CVRP using a reference-solution
    affinity approach. We solve the CVRP with a fast constructive + local search
    method, then build a heuristic matrix where edges present in the reference
    solution get high attractiveness, and others get scaled down based on distance.
    
    Heuristic[i, j] = base_distance_heuristic * (1 + affinity_boost * in_reference)
    """
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    # Start with a time guard
    t0 = time.time()
    time_budget = 10.0  # seconds for reference solution
    
    # Base heuristic: inverse distance (standard ACO heuristic)
    safe_dist = distance_matrix + eps
    base_heuristic = 1.0 / safe_dist
    
    # Build reference solution
    try:
        routes = _solve_cvrp_reference(distance_matrix, coordinates, demands, capacity, n, time_limit=time_budget)
    except Exception:
        # Fallback: just use base heuristic
        return np.maximum(base_heuristic, 1e-9)
    
    # Build edge set from reference routes
    # For each route, the full tour is: depot -> r1 -> r2 -> ... -> rk -> depot
    edge_set = set()
    for route in routes:
        if not route:
            continue
        # Add edges in the route
        for i in range(len(route)):
            u = route[i]
            v = route[i + 1] if i + 1 < len(route) else 0
            edge_set.add((min(u, v), max(u, v)))
            edge_set.add((v, min(u, v)))  # symmetric
        # Edge from last node back to depot is already added when i = len-1
    
    # Also add depot-to-first and last-to-depot explicitly (already covered above)
    
    # Build affinity matrix
    affinity = np.zeros((n, n))
    for (u, v) in edge_set:
        affinity[u, v] = 1.0
        affinity[v, u] = 1.0
    
    # Combine: base heuristic boosted for reference edges
    # Use a multiplicative boost: reference edges get 3x the base heuristic
    boost_factor = 3.0
    heuristic = base_heuristic * (1.0 + (boost_factor - 1.0) * affinity)
    
    # Ensure symmetry (should already be symmetric, but enforce)
    heuristic = (heuristic + heuristic.T) / 2.0
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    # Final time guard: if we've exceeded a safe margin, return what we have
    # (The computation above is already done, so this is just a formality)
    
    return heuristic
# EVOLVE-BLOCK-END
