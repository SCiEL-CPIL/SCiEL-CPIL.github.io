"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for the entire evaluation run to enforce deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    if time.time() - _start_time > 145:
        # Fallback to simple nearest feasible to avoid timeout
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        return int(feasible[np.argmin(distance_matrix[current_node, feasible])])

    # 1. Identify feasible nodes
    unvisited_array = np.array(unvisited_nodes)
    if len(unvisited_array) == 0:
        return depot
        
    feasible_mask = demands[unvisited_array] <= rest_capacity
    feasible_nodes = unvisited_array[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # 2. Calculate scores for feasible nodes
    # Strategy: Voronoi-based demand consolidation
    # Prioritize nodes that allow the route to exhaust capacity within a localized geographic cluster
    
    # Distances from current node to feasible nodes
    dists = distance_matrix[current_node, feasible_nodes]
    
    # Savings: S(i,j) = d(i,depot) + d(depot,j) - d(i,j)
    d_i_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    savings = d_i_depot + d_depot_j - dists
    
    # Voronoi-based demand score:
    # For each feasible node i, estimate the total demand in its "Voronoi region"
    # A node j is in i's Voronoi region if d(i,j) < d(k,j) for all other feasible k != i
    # We approximate this by counting demand of nodes closer to i than to any other feasible node
    
    n_feasible = len(feasible_nodes)
    idx = feasible_nodes
    
    # Get submatrix of distances between feasible nodes
    sub_dist = distance_matrix[np.ix_(idx, idx)]
    
    # For each node i, find nodes j where i is the nearest feasible node
    # This is a nearest-neighbor assignment problem
    # For each j, find argmin_i d(i,j)
    # Then sum demands of all j assigned to each i
    
    # sub_dist[i,j] = distance from feasible node i to feasible node j
    # For each column j, find the row i that minimizes sub_dist[i,j]
    nearest_to_j = np.argmin(sub_dist, axis=0)  # For each j, which i is closest
    
    # Now, for each i, sum the demands of all j assigned to i
    # We need to map from feasible node indices to actual node indices
    demands_feasible = demands[feasible_nodes]
    
    # Create an array to accumulate demand for each feasible node i
    voronoi_demand = np.zeros(n_feasible)
    
    # For each j, add its demand to the nearest i
    # Use np.add.at for efficient accumulation
    np.add.at(voronoi_demand, nearest_to_j, demands_feasible)
    
    # Normalize voronoi_demand by rest_capacity to get a "capacity utilization" score
    # Higher voronoi_demand means this node covers more remaining demand in its region
    # We want to prioritize nodes that can "cover" more demand
    capacity_utilization = voronoi_demand / (rest_capacity + 1e-9)
    
    # 3. Combine scores
    # We want to minimize distance, maximize savings, maximize voronoi demand coverage
    # Score = dist - alpha * savings - beta * capacity_utilization
    
    # Tuning constants:
    # alpha=1.0 makes savings important.
    # beta=0.5 adds preference for nodes with high Voronoi demand coverage.
    alpha = 1.0
    beta = 0.5
    
    scores = dists - alpha * savings - beta * capacity_utilization
    
    # 4. Select best node
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
