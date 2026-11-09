"""Optimized heuristics for ACO CVRP using a SEARCH-based approach with Greedy-2Opt evaluation."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses a search approach: generate multiple candidate heuristic matrices
    based on different weighting strategies and return the one that
    minimizes expected route cost (estimated via a fast greedy + 2-opt simulation).
    """
    start_time = time.time()
    deadline = start_time + 290.0  # 10s safety margin before 300s limit
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    # Validate inputs
    if distance_matrix.shape != (n, n):
        return 1.0 / (distance_matrix + 1e-9)

    # Base heuristic: inverse distance
    # We will vary the exponent to find the best discriminative power
    # Knowledge: Higher exponents (e.g., 1.5) often lead to tighter initial routes.
    
    # Precompute useful arrays
    demand_ratio = demands / max(demands.max(), 1e-9)
    
    # Candidate strategies
    # We focus on exponent variations and slight demand adjustments.
    # The ACO probability is proportional to pheromone * heuristic.
    # Initially pheromone is uniform, so the heuristic drives the first moves.
    # A steeper heuristic (higher exponent) forces ants to take short edges early.
    
    candidates = []
    
    # Exponent variations
    exponents = [0.8, 1.0, 1.2, 1.5, 2.0]
    demand_scales = [0.0, 0.1, 0.2] # Factor for demand penalty: 1/(1 + factor * demand)
    
    for exp in exponents:
        base = 1.0 / (distance_matrix ** exp + 1e-9)
        for ds in demand_scales:
            if ds == 0.0:
                cand = base.copy()
            else:
                # Penalize edges to high-demand nodes slightly, 
                # encouraging load balancing or early visits to heavy nodes?
                # Actually, for CVRP, it's often better to group high demand.
                # Let's try a neutral scaling first, then maybe boost high demand?
                # Let's stick to a simple inverse demand scaling which prefers low demand 
                # (easier to fit later) or just neutral.
                # The prompt says "demand-aware scaling" in inspirations.
                # Let's try: scale by 1/(1 + ds * demand_ratio[j])
                scale_matrix = np.ones((n, n))
                for j in range(1, n):
                    scale_matrix[:, j] = 1.0 / (1.0 + ds * demand_ratio[j])
                cand = base * scale_matrix
            candidates.append(cand)
            
    # Also include a depot-boosted version of the best exponent (1.5)
    # Edges to/from depot are crucial for route completion
    base_15 = 1.0 / (distance_matrix ** 1.5 + 1e-9)
    depot_boost = np.ones((n, n))
    depot_boost[0, :] = 1.2
    depot_boost[:, 0] = 1.2
    candidates.append(base_15 * depot_boost)
    
    # Evaluate candidates using Fast Greedy + 2-Opt
    def evaluate_candidate(heuristic_mat):
        """
        Simulates a greedy construction guided by the heuristic, 
        then applies 2-Opt to estimate the quality.
        Returns the total distance of the polished solution.
        """
        # Greedy Construction
        # At each step, pick the feasible neighbor j that maximizes heuristic_mat[current, j]
        # This mimics the ACO behavior when pheromone is uniform (initial state).
        
        visited = np.zeros(n, dtype=bool)
        visited[0] = True
        current = 0
        load = 0.0
        route = [0]
        
        while True:
            # Find feasible neighbors
            best_j = -1
            best_val = -1e9
            
            # Optimization: Pre-compute feasible list? 
            # n=50 is small, linear scan is fine.
            for j in range(1, n):
                if not visited[j] and load + demands[j] <= capacity:
                    val = heuristic_mat[current, j]
                    if val > best_val:
                        best_val = val
                        best_j = j
                        
            if best_j == -1:
                # No feasible customer, return to depot
                route.append(0)
                current = 0
                load = 0.0
                # Check if all visited
                if np.sum(visited) == n:
                    break
                continue
            
            # Move to best_j
            route.append(best_j)
            visited[best_j] = True
            load += demands[best_j]
            current = best_j
            
        # Calculate raw cost
        raw_cost = 0.0
        for k in range(len(route) - 1):
            raw_cost += distance_matrix[route[k], route[k+1]]
            
        # Apply 2-Opt to improve the route
        # The route is a sequence of nodes. 2-Opt reverses segments to reduce distance.
        # We need to respect the depot breaks.
        # Split route into sub-tours (segments between 0s)
        
        # For simplicity and speed, we apply 2-Opt on the entire sequence 
        # but ensuring we don't cross depot boundaries improperly?
        # Actually, standard 2-Opt on the full path 0->...->0->...->0 is valid 
        # if we treat the depot as a fixed point. 
        # However, reversing a segment that includes the depot might break the route structure 
        # (e.g., 0-A-B-0-C-D-0 -> 0-B-A-0-C-D-0 is fine. 0-A-B-0-C-D-0 -> 0-A-C-B-D-0 is NOT a valid CVRP solution 
        # because it mixes routes).
        
        # So we must apply 2-Opt within each sub-tour (between depots).
        
        # Identify sub-tours
        sub_tours = []
        current_tour = []
        for node in route:
            if node == 0:
                if current_tour:
                    sub_tours.append(current_tour)
                    current_tour = []
            else:
                current_tour.append(node)
        if current_tour:
            sub_tours.append(current_tour)
            
        improved_cost = 0.0
        
        for tour in sub_tours:
            if len(tour) < 3:
                # Cost is just depot->first + last->depot + internal
                if len(tour) > 0:
                    improved_cost += distance_matrix[0, tour[0]] + distance_matrix[tour[-1], 0]
                    for i in range(len(tour)-1):
                        improved_cost += distance_matrix[tour[i], tour[i+1]]
                continue
                
            # 2-Opt on this sub-tour
            # The tour is a path: 0 -> tour[0] -> ... -> tour[-1] -> 0
            # We can reverse segments of 'tour'
            
            improved = True
            # Limit iterations for speed
            max_iters = 10
            iter_count = 0
            while improved and iter_count < max_iters:
                improved = False
                iter_count += 1
                # Calculate current cost of this sub-tour
                curr_cost = distance_matrix[0, tour[0]] + distance_matrix[tour[-1], 0]
                for i in range(len(tour)-1):
                    curr_cost += distance_matrix[tour[i], tour[i+1]]
                
                best_gain = 0
                best_i, best_j = -1, -1
                
                # Try all pairs i < j
                # Gain = (dist[0,i] + dist[j,0]) - (dist[0,j] + dist[i,0]) ? No.
                # Reversing tour[i..j] changes edges:
                # Old: (prev(i) -> i) and (j -> next(j))
                # New: (prev(i) -> j) and (i -> next(j))
                # If i is 0 (start), prev is depot. If j is end, next is depot.
                
                L = len(tour)
                for i in range(L - 1):
                    for j in range(i + 1, L):
                        # Define nodes
                        # i and j are indices in 'tour'
                        # Edge 1: between node before i and i
                        # Edge 2: between node j and node after j
                        
                        # Node before i:
                        if i == 0:
                            prev_node = 0
                            dist_prev_i = distance_matrix[0, tour[0]]
                        else:
                            prev_node = tour[i-1]
                            dist_prev_i = distance_matrix[prev_node, tour[i]]
                            
                        # Node after j:
                        if j == L - 1:
                            next_node = 0
                            dist_j_next = distance_matrix[tour[j], 0]
                        else:
                            next_node = tour[j+1]
                            dist_j_next = distance_matrix[tour[j], next_node]
                        
                        # New edges:
                        # prev_node -> tour[j]
                        new_dist_1 = distance_matrix[prev_node, tour[j]]
                        # tour[i] -> next_node
                        new_dist_2 = distance_matrix[tour[i], next_node]
                        
                        gain = (dist_prev_i + dist_j_next) - (new_dist_1 + new_dist_2)
                        
                        if gain > best_gain + 1e-6:
                            best_gain = gain
                            best_i, best_j = i, j
                
                if best_i != -1:
                    # Reverse segment
                    tour[best_i:best_j+1] = tour[best_i:best_j+1][::-1]
                    improved = True
            
            # Add final cost of this sub-tour
            improved_cost += distance_matrix[0, tour[0]] + distance_matrix[tour[-1], 0]
            for i in range(len(tour)-1):
                improved_cost += distance_matrix[tour[i], tour[i+1]]
                
        return improved_cost

    # Evaluate each candidate
    best_candidate = None
    best_cost = float('inf')
    
    for idx, cand in enumerate(candidates):
        if time.time() > deadline - 5:
            break
        try:
            cost = evaluate_candidate(cand)
            if cost < best_cost:
                best_cost = cost
                best_candidate = cand
        except Exception:
            continue

    # If no valid candidate, fallback to base heuristic (exp 1.0)
    if best_candidate is None:
        best_candidate = 1.0 / (distance_matrix + 1e-9)
    
    # Ensure all entries are positive
    result = np.maximum(best_candidate, 1e-9)
    
    return result
# EVOLVE-BLOCK-END
