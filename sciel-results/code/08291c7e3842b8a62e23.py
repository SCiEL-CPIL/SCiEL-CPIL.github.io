# EVOLVE-BLOCK-START
import numpy as np
import time

def _optimize_tour_2opt(tour, distance_matrix):
    """
    Performs 2-opt optimization on a given tour.
    tour: list of node indices (excluding depot at start/end, or including? 
          Standard 2-opt works on the sequence of customers. 
          Here we assume tour is the sequence of customers visited in order.
          We will reconstruct the full path with depot to calculate costs.
    """
    n = len(tour)
    if n <= 2:
        return tour
    
    # Create a mutable copy
    current_tour = list(tour)
    
    def _tour_cost(t):
        # t is list of customers
        total = 0.0
        prev = 0
        for node in t:
            total += distance_matrix[prev, node]
            prev = node
        total += distance_matrix[prev, 0]
        return total
    
    improved = True
    max_iterations = 50 # Safety bound for small n
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Reverse segment i to j
                # Original: ... A[i] - A[i+1] ... A[j] ...
                # New: ... A[i] - A[j] ... A[i+1] ...
                
                # Calculate delta cost
                # Nodes involved: 
                # u = current_tour[i-1] if i > 0 else 0
                # v = current_tour[i]
                # x = current_tour[j]
                # y = current_tour[j+1] if j < n-1 else 0
                
                u = current_tour[i-1] if i > 0 else 0
                v = current_tour[i]
                x = current_tour[j]
                y = current_tour[j+1] if j < n - 1 else 0
                
                # Cost before: d(u,v) + d(x,y)
                # Cost after: d(u,x) + d(v,y)
                # Note: The internal edges of the reversed segment remain the same 
                # because distance matrix is symmetric (Euclidean).
                
                delta = (distance_matrix[u, x] + distance_matrix[v, y]) - (distance_matrix[u, v] + distance_matrix[x, y])
                
                if delta < -1e-10:
                    # Reverse the segment
                    current_tour[i:j+1] = current_tour[i:j+1][::-1]
                    improved = True
                    
    return current_tour

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Generates an edge-attractiveness matrix using a Dynamic Cluster Affinity strategy.
    
    1. Construct a high-quality initial tour using a Sweep + 2-Opt approach.
       - Sweep: Sort customers by angle from depot.
       - Partition: Split into routes respecting capacity (greedy packing in angular order).
       - Optimize: Apply 2-opt within each route and potentially across the whole sequence 
         if we treat it as a single TSP-like problem, but for CVRP, we optimize each route 
         independently after initial partitioning. 
       
       Actually, a simpler and robust "expert" heuristic for CVRP:
       - Use Nearest Neighbor with capacity constraints to build a set of routes.
       - Apply 2-opt to each route.
       - This yields a valid, good set of routes.
       
    2. Construct the heuristic matrix H such that:
       - H[i, j] is high if the edge (i, j) is present in the expert solution.
       - H[i, j] is low otherwise.
       - We use a "soft" indicator: H[i, j] = 1.0 if (i,j) or (j,i) is in expert tour, 
         else 1e-9. 
       - To allow some exploration, we can blend with a distance-based fallback:
         H[i, j] = expert_indicator[i, j] * (1.0 / distance_matrix[i, j]) + small_noise
       
       However, pure binary might be too restrictive if the expert solution is suboptimal 
       in a way that requires deviating. 
       A better approach: 
       H[i, j] = 1.0 if edge is in expert tour, else 0.1 * (1/d). 
       This strongly prefers expert edges but allows others.
       
       Let's refine: 
       The ACO uses pheromone * heuristic. 
       If we set H to 1 for expert edges and 0.01 for others, the ACO will likely 
       replicate the expert solution initially, then pheromone accumulation might 
       reinforce it or find better local variations if the expert solution has 
       weaknesses.
       
       Let's implement:
       1. Generate expert tour via NN + 2-Opt per route.
       2. Create a boolean matrix `expert_edges`.
       3. H[i, j] = 10.0 if expert_edge else (1.0 / (distance_matrix[i, j] + 1e-6)).
       
       This ensures that the "expert" moves are 10x more attractive than a typical 
       close move, and much more attractive than a far move.
    """
    n = distance_matrix.shape[0]
    start_time = time.time()
    deadline = start_time + 290.0

    distance_matrix = np.asarray(distance_matrix, dtype=float)
    coordinates = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    # 1. Generate Expert Solution
    # Strategy: Sweep-based partitioning followed by 2-Opt per route.
    
    # Step 1a: Sort customers by angle
    angles = np.zeros(n)
    for k in range(1, n):
        x, y = coordinates[k]
        angles[k] = np.arctan2(y, x)
    
    customer_indices = np.arange(1, n)
    sorted_customers = customer_indices[np.argsort(angles[1:n])]
    
    # Step 1b: Partition into routes using First-Fit Decreasing (by angle order)
    # We iterate through sorted customers and add to current route if it fits,
    # else start new route.
    routes = []
    current_route = []
    current_load = 0.0
    
    for c in sorted_customers:
        d = demands[c]
        if current_load + d <= capacity:
            current_route.append(c)
            current_load += d
        else:
            if current_route:
                routes.append(current_route)
            current_route = [c]
            current_load = d
            
    if current_route:
        routes.append(current_route)
        
    # Step 1c: Optimize each route with 2-Opt
    optimized_routes = []
    for route in routes:
        opt_route = _optimize_tour_2opt(route, distance_matrix)
        optimized_routes.append(opt_route)
        
    # Step 1d: Construct Expert Edge Matrix
    expert_edges = np.zeros((n, n), dtype=bool)
    for route in optimized_routes:
        prev = 0
        for node in route:
            expert_edges[prev, node] = True
            expert_edges[node, prev] = True # Symmetric
            prev = node
        expert_edges[prev, 0] = True
        expert_edges[0, prev] = True
        
    # 2. Construct Heuristic Matrix
    # Base heuristic: Inverse distance (standard ACO heuristic)
    # We avoid division by zero for self-loops, though ants don't visit self.
    dist_safe = distance_matrix.copy()
    np.fill_diagonal(dist_safe, 1e6) # Large value -> small heuristic
    base_heuristic = 1.0 / dist_safe
    
    # Expert boost
    # If edge is in expert solution, multiply by a large factor.
    # Let's use a factor of 100.0 to strongly guide the ACO.
    # H = base_heuristic * (1.0 + 99.0 * expert_edges)
    # This makes expert edges ~100x more attractive than non-expert edges of same length.
    
    heuristic_matrix = base_heuristic * (1.0 + 99.0 * expert_edges)
    
    # Ensure positivity and finiteness
    heuristic_matrix = np.where(heuristic_matrix > 0, heuristic_matrix, 1e-9)
    if not np.all(np.isfinite(heuristic_matrix)):
        heuristic_matrix = np.ones_like(distance_matrix) * 1e-9
        
    return heuristic_matrix
# EVOLVE-BLOCK-END
