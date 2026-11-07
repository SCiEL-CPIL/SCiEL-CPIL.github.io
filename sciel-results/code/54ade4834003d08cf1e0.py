# EVOLVE-BLOCK-START
import numpy as np
import time

def _optimize_tour_2opt(tour, distance_matrix):
    """
    Performs 2-opt optimization on a given tour.
    tour: list of node indices (customers only).
    """
    n = len(tour)
    if n <= 2:
        return tour
    
    current_tour = list(tour)
    
    improved = True
    max_iterations = 30
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                u = current_tour[i-1] if i > 0 else 0
                v = current_tour[i]
                x = current_tour[j]
                y = current_tour[j+1] if j < n - 1 else 0
                
                # Cost before: d(u,v) + d(x,y)
                # Cost after: d(u,x) + d(v,y)
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-10:
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    improved = True
                    
    return current_tour

def _nearest_neighbor_route(distance_matrix, demands, capacity, start_node, visited, rng):
    """
    Constructs a single route starting from start_node (usually depot) using a 
    randomized nearest neighbor heuristic, respecting capacity.
    Returns the list of nodes in the route and the final node index.
    """
    route = []
    current = start_node
    remaining_capacity = capacity
    local_visited = visited.copy()
    
    while True:
        # Find feasible neighbors: not visited, demand fits
        feasible = []
        for j in range(1, len(demands)):
            if not local_visited[j] and demands[j] <= remaining_capacity:
                feasible.append(j)
                
        if not feasible:
            break
            
        # Randomized selection: pick with probability proportional to 1/distance
        # But to keep it simple and fast, we can just pick the nearest, or a random one from top-k
        # Let's use a simple nearest neighbor for speed, but we are calling this multiple times 
        # with different starting conditions? No, NN is deterministic. 
        # To get diversity, we need to randomize the start or the choice.
        # Actually, the plan is to run multiple NNs from the depot. They will be identical.
        # So we must randomize the choice.
        
        # Calculate distances to feasible
        dists = np.array([distance_matrix[current, j] for j in feasible])
        
        # Inverse distance probabilities
        probs = 1.0 / (dists + 1e-9)
        probs /= probs.sum()
        
        # Sample next node
        next_node = feasible[rng.choice(len(feasible), p=probs)]
        
        route.append(next_node)
        local_visited[next_node] = True
        remaining_capacity -= demands[next_node]
        current = next_node
        
    return route, current

def _solve_cvrp_nn_2opt(distance_matrix, demands, capacity, rng, max_routes=20):
    """
    Solves CVRP using a multi-start Nearest Neighbor + 2-Opt approach.
    Returns a list of solutions, where each solution is a list of routes.
    """
    n = len(demands)
    unvisited = np.ones(n, dtype=bool)
    unvisited[0] = False
    
    solutions = []
    
    for _ in range(max_routes):
        current_unvisited = unvisited.copy()
        routes = []
        
        while np.any(current_unvisited[1:]):
            # Start a new route from depot
            route, end_node = _nearest_neighbor_route(distance_matrix, demands, capacity, 0, current_unvisited, rng)
            
            if not route:
                # Should not happen if there are unvisited nodes that fit, 
                # but if a single demand > capacity, it would loop. 
                # Problem states demands 1-9, cap 50, so always fits.
                break
                
            # Update unvisited
            for node in route:
                current_unvisited[node] = False
                
            # Optimize this route
            opt_route = _optimize_tour_2opt(route, distance_matrix)
            routes.append(opt_route)
            
        if routes:
            solutions.append(routes)
            
    return solutions

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a multi-start ensemble of 
    Nearest Neighbor + 2-Opt solutions. The heuristic value for an edge is 
    inversely proportional to the frequency of that edge appearing in the 
    high-quality solutions, combined with a base inverse-distance signal.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 280.0
    
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # Base heuristic: Inverse distance
    safe_dist = distance_matrix + 1e-9
    h_base = 1.0 / safe_dist
    
    # Initialize edge frequency counter
    edge_count = np.zeros((n, n), dtype=float)
    
    # Run ensemble
    rng = np.random.default_rng(42)
    num_solutions = 50 # Try to generate up to 50 diverse solutions
    
    for _ in range(num_solutions):
        if time.time() > deadline - 20.0:
            break
            
        # Generate a set of routes
        routes = _solve_cvrp_nn_2opt(distance_matrix, demands, capacity, rng, max_routes=1)
        
        if not routes:
            continue
            
        for route in routes:
            # Add depot edges
            prev = 0
            for node in route:
                edge_count[prev, node] += 1
                edge_count[node, prev] += 1
                prev = node
            edge_count[prev, 0] += 1
            edge_count[0, prev] += 1
            
    # Construct heuristic from edge counts
    # We want edges that appear frequently in good solutions to be attractive.
    # However, simple counting might be too sparse. 
    # Let's combine: h = h_base * (1 + alpha * normalized_count)
    
    # Normalize counts: max count is likely 1 or 2 for specific edges in a single solution,
    # but across 50 solutions, popular edges will have higher counts.
    max_count = edge_count.max()
    if max_count > 0:
        norm_count = edge_count / max_count
    else:
        norm_count = np.zeros_like(edge_count)
        
    # Blend base inverse distance with ensemble consensus
    # The ensemble tells us which specific edges are "good".
    # The base tells us general geometry.
    # If an edge is in the top solutions, boost it significantly.
    alpha = 10.0
    h_ensemble = h_base * (1.0 + alpha * norm_count)
    
    # Ensure positivity and finiteness
    h_ensemble = np.where(np.isfinite(h_ensemble) & (h_ensemble > 0), h_ensemble, 1e-9)
    
    # Final safety check
    if not np.all(np.isfinite(h_ensemble)):
        h_ensemble = h_base
        
    return h_ensemble
# EVOLVE-BLOCK-END
