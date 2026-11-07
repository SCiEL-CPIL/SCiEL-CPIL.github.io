# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_reference_routes(distance_matrix, demands, capacity):
    """
    Generate high-quality reference routes using a fast constructive heuristic
    (Capacity-Aware Nearest Neighbor) followed by local search (2-Opt).
    Returns a list of routes, where each route is a list of node indices.
    """
    n = distance_matrix.shape[0]
    dm = distance_matrix
    demands_arr = np.asarray(demands, dtype=np.float64)
    
    unvisited = set(range(1, n))
    routes = []
    
    # 1. Constructive Phase: Capacity-Aware Nearest Neighbor
    while unvisited:
        current = 0
        load = 0.0
        route = [0]
        while True:
            best_node = -1
            best_dist = np.inf
            # Find nearest unvisited customer that fits
            for j in unvisited:
                if demands_arr[j] <= capacity - load:
                    dist = dm[current, j]
                    if dist < best_dist:
                        best_dist = dist
                        best_node = j
            
            if best_node == -1:
                break
            
            route.append(best_node)
            load += demands_arr[best_node]
            unvisited.remove(best_node)
            current = best_node
        
        # Return to depot
        route.append(0)
        if len(route) > 2: # Only keep non-empty routes
            routes.append(route)
    
    # 2. Local Search Phase: 2-Opt for each route
    # Note: 2-Opt on a path (depot -> ... -> depot) is slightly different than a tour,
    # but we can treat the segment between depots as a path and optimize the internal order.
    
    def two_opt_route(route, dm):
        """
        Perform 2-Opt on a route defined as [0, c1, c2, ..., 0].
        We only swap internal segments to avoid breaking the depot structure improperly 
        unless we treat it as a full cycle. Since it's a path from 0 to 0, 
        we can reverse segments between i and j.
        """
        if len(route) < 4:
            return route
        
        improved = True
        # Limit iterations to keep it fast, as we have multiple routes
        max_iters = 100
        iter_count = 0
        
        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route) - 1):
                    # Calculate cost change for reversing route[i..j]
                    # Original: ... route[i-1] -> route[i] ... route[j] -> route[j+1] ...
                    # New:      ... route[i-1] -> route[j] ... route[i] -> route[j+1] ...
                    
                    # Edge (i-1, i) and (j, j+1) are removed
                    # Edge (i-1, j) and (i, j+1) are added
                    # (Assuming i-1 and j+1 are valid indices in the route list)
                    
                    if i == 1:
                        # i-1 is 0 (depot)
                        d_old_1 = dm[route[i-1], route[i]]
                    else:
                        d_old_1 = dm[route[i-1], route[i]]
                    
                    d_old_2 = dm[route[j], route[j+1]]
                    
                    d_new_1 = dm[route[i-1], route[j]]
                    d_new_2 = dm[route[i], route[j+1]]
                    
                    if d_new_1 + d_new_2 < d_old_1 + d_old_2 - 1e-9:
                        # Reverse the segment
                        route[i:j+1] = route[i:j+1][::-1]
                        improved = True
        
        return route

    optimized_routes = []
    for route in routes:
        # Apply 2-Opt
        opt_route = two_opt_route(route, dm)
        optimized_routes.append(opt_route)
        
    return optimized_routes

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute heuristic matrix by injecting a strong geometric prior derived from
    a fast constructive heuristic (NN + 2-Opt). This avoids the expensive
    parameter search of the previous strategy.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    dm = np.asarray(distance_matrix, dtype=np.float64)
    n = dm.shape[0]
    
    # 1. Base Heuristic: Inverse Distance
    # Avoid division by zero
    dm_safe = dm + 1e-9
    eta = 1.0 / dm_safe
    
    # Ensure finite and positive
    eta[~np.isfinite(eta)] = 1e-9
    eta[eta <= 0] = 1e-9
    
    # 2. Generate Reference Routes
    # This is fast: O(n^2) for NN, O(n^2 * iters) for 2-Opt
    reference_routes = _generate_reference_routes(dm, demands, capacity)
    
    # 3. Boost Edges in Reference Routes
    # We want to strongly encourage the ACO to use edges that were part of the
    # high-quality constructive solution.
    boost_factor = 10.0
    min_boost = 5.0
    
    for route in reference_routes:
        for i in range(len(route) - 1):
            u = route[i]
            v = route[i+1]
            # Boost the attractiveness of this edge
            # We use a multiplicative boost on the base 1/d value
            # But to be safe, we can just set it to a high value or multiply
            if eta[u, v] > 0:
                eta[u, v] *= boost_factor
            else:
                eta[u, v] = min_boost
    
    # 4. Symmetrize? 
    # The problem is directed in the sense of "from i to j", but distance is symmetric.
    # The ACO uses eta[i, j]. Since dm is symmetric, eta is symmetric.
    # No need to explicitly symmetrize if we updated both directions? 
    # Wait, the loop `for i in range(len(route)-1)` only updates (u, v).
    # Since the tour is a cycle/path, the reverse edge (v, u) might not be in the route 
    # unless the route goes back. But in CVRP, we don't necessarily traverse edges in both directions.
    # However, the heuristic matrix is used for i->j. If the reference solution used i->j, 
    # we boost i->j. This is correct for the direction of travel.
    
    # 5. Final Cleanup
    # Ensure diagonal is small (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    # Check deadline (though this should be very fast)
    if time.time() > deadline:
        pass # Just return what we have
    
    return eta
# EVOLVE-BLOCK-END
