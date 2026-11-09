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
    
    # Time budget management
    deadline = time.time() + 270  # Safety margin before 330s limit
    elapsed = lambda: time.time() - (deadline - 270)
    
    # Clarke-Wright Savings with capacity constraint
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
    
    # Boost heuristic for good edges
    boost_factor = 4.0
    for i, j in good_edges:
        heuristic[i, j] *= boost_factor
        heuristic[j, i] *= boost_factor
    
    # Additional local optimization: try 2-opt style improvements on the savings solution
    if time.time() < deadline - 30:
        # Build route structure from union-find
        routes = {}
        for i in range(1, n):
            r = find(i)
            if r not in routes:
                routes[r] = []
            routes[r].append(i)
        
        # For each route, try to improve by swapping edge assignments
        # This is a simplified version - just identify edges that appear in multiple routes
        edge_count = {}
        for i, j in good_edges:
            edge_count[(i, j)] = edge_count.get((i, j), 0) + 1
            edge_count[(j, i)] = edge_count.get((j, i), 0) + 1
        
        # Further boost edges that are used multiple times
        for edge, count in edge_count.items():
            if count > 1:
                heuristic[edge[0], edge[1]] *= (1.0 + 0.5 * (count - 1))
    
    # Final cleanup
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
