import numpy as np
import time

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Heuristic matrix for ant colony optimization of capacitated vehicle routing.
    
    Args:
        distance_matrix: (n, n) matrix of Euclidean distances
        coordinates: (n, 2) array of coordinates in unit square
        demands: (n,) array of demands (node 0 is depot with demand 0)
        capacity: vehicle capacity (50)
    
    Returns:
        (n, n) edge-attractiveness matrix where larger values are more attractive.
        Non-positive entries become 1e-9.
    """
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    dm = np.asarray(demands, dtype=float)
    n = d.shape[0]
    
    if d.shape[0] != d.shape[1]:
        return np.ones((n, n))
    
    # Time budget management
    deadline = time.time() + 270  # Safety margin before 330s limit
    
    # Base inverse distance heuristic with tuned exponent
    heuristic = np.zeros_like(d)
    mask = d > 1e-12
    heuristic[mask] = 1.0 / (d[mask] ** 2.5)
    
    # Demand factor: favor edges to high-demand nodes for vehicle filling
    demand_factor = 1.0 + dm / capacity
    heuristic = heuristic * demand_factor[np.newaxis, :]
    
    # Ensure positive values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    # Clarke-Wright Savings with capacity constraint (existing method)
    depot = 0
    customers = list(range(1, n))
    
    # Compute savings matrix
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            s = d[depot, i] + d[depot, j] - d[i, j]
            savings[i, j] = s
            savings[j, i] = s
    
    # Sort savings in descending order
    pairs = [(savings[i, j], i, j) for i in range(1, n) for j in range(i + 1, n) if savings[i, j] > 0]
    pairs.sort(reverse=True)
    
    # Union-Find with capacity tracking
    parent = list(range(n))
    rank = [0] * n
    route_load = [0.0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        route_load[rx] = route_load[ry] + dm[rx] + dm[ry]
        return True
    
    # Initialize: each customer is its own route
    for i in range(1, n):
        route_load[i] = dm[i]
    
    # Process savings in order with capacity constraint
    good_edges = set()
    for s, i, j in pairs:
        if time.time() > deadline:
            break
        
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        
        new_load = route_load[ri] + route_load[rj]
        if new_load > capacity:
            continue
        
        if union(ri, rj):
            good_edges.add((min(i, j), max(i, j)))
    
    # Boost heuristic for good edges from Clarke-Wright
    boost_factor = 4.0
    for i, j in good_edges:
        heuristic[i, j] *= boost_factor
        heuristic[j, i] *= boost_factor
    
    # Lagrangian Relaxation-based edge identification
    # Solve relaxed TSP (ignore capacity) and identify edges that are locally optimal
    # under a relaxed constraint with penalty for load violations
    if time.time() < deadline - 60:
        # Step 1: Compute relaxed edge weights that penalize load imbalance
        # For each edge (i,j), compute a "relaxed attractiveness" that considers
        # both distance and how well the edge fits into a capacity-respecting route
        
        # Dual variable: penalty for exceeding capacity
        # We'll use a simple approach: for each pair (i,j), compute the marginal cost
        # of adding this edge to a route, considering capacity constraints
        
        # Compute node "density" - how much demand each node has relative to capacity
        node_density = dm / capacity
        
        # For each edge (i,j), compute a Lagrangian-adjusted weight
        # The idea: edges between nodes with complementary demand profiles are more attractive
        # because they can be combined without exceeding capacity
        
        # Compute pairwise "compatibility" based on demand
        # Two nodes are compatible if their combined demand fits in one vehicle
        compatibility = np.ones((n, n))
        for i in range(1, n):
            for j in range(i + 1, n):
                combined_demand = dm[i] + dm[j]
                if combined_demand <= capacity:
                    # Compatible: boost based on how "full" the vehicle would be
                    # Full vehicles are more efficient, so boost edges that lead to fuller loads
                    fill_ratio = combined_demand / capacity
                    compatibility[i, j] = 1.0 + 2.0 * fill_ratio
                    compatibility[j, i] = compatibility[i, j]
                else:
                    # Incompatible: penalize
                    compatibility[i, j] = 0.5
                    compatibility[j, i] = compatibility[i, j]
        
        # Step 2: Solve a relaxed assignment problem to identify "good" edges
        # Use a simple greedy approach: repeatedly pick the best edge that doesn't create cycles
        # and respects a relaxed capacity constraint
        
        # Initialize: each node is in its own component
        comp_parent = list(range(n))
        comp_load = dm.copy()
        
        def comp_find(x):
            while comp_parent[x] != x:
                comp_parent[x] = comp_parent[comp_parent[x]]
                x = comp_parent[x]
            return x
        
        def comp_union(x, y):
            rx, ry = comp_find(x), comp_find(y)
            if rx == ry:
                return False
            comp_parent[ry] = rx
            comp_load[rx] += comp_load[ry]
            return True
        
        # Compute relaxed edge scores: combine distance, demand compatibility, and savings
        relaxed_scores = np.zeros((n, n))
        for i in range(1, n):
            for j in range(i + 1, n):
                # Base score: inverse distance
                base_score = 1.0 / (d[i, j] + 1e-12)
                
                # Savings component
                savings_score = max(0, savings[i, j])
                
                # Compatibility component
                compat_score = compatibility[i, j]
                
                # Combined score with Lagrangian-like weighting
                # The dual variable lambda penalizes capacity violations
                # We approximate this by weighting compatibility more heavily
                lambda_penalty = 1.5
                relaxed_scores[i, j] = base_score * (1.0 + lambda_penalty * compat_score) + 0.5 * savings_score
                relaxed_scores[j, i] = relaxed_scores[i, j]
        
        # Greedy edge selection with relaxed capacity constraint
        # Use a higher capacity threshold to be more permissive
        relaxed_capacity = capacity * 1.2  # Allow 20% over capacity in relaxed solution
        
        relaxed_good_edges = set()
        edge_list = []
        for i in range(1, n):
            for j in range(i + 1, n):
                if relaxed_scores[i, j] > 0:
                    edge_list.append((relaxed_scores[i, j], i, j))
        
        edge_list.sort(reverse=True)
        
        for score, i, j in edge_list:
            if time.time() > deadline:
                break
            
            ri, rj = comp_find(i), comp_find(j)
            if ri == rj:
                continue
            
            # Check relaxed capacity constraint
            new_load = comp_load[ri] + comp_load[rj]
            if new_load > relaxed_capacity:
                continue
            
            if comp_union(ri, rj):
                relaxed_good_edges.add((min(i, j), max(i, j)))
        
        # Boost heuristic for relaxed-good edges
        # These edges are locally optimal under the relaxed constraint
        relaxed_boost = 3.0
        for i, j in relaxed_good_edges:
            heuristic[i, j] *= relaxed_boost
            heuristic[j, i] *= relaxed_boost
        
        # Additional: identify edges that are in BOTH Clarke-Wright and relaxed solutions
        # These are particularly strong candidates
        intersection_edges = good_edges & relaxed_good_edges
        if intersection_edges:
            intersection_boost = 2.0
            for i, j in intersection_edges:
                heuristic[i, j] *= intersection_boost
                heuristic[j, i] *= intersection_boost
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
