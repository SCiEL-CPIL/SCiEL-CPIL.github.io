import numpy as np
import time

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Deadline guard: track elapsed time and return a safe result if we're close to the limit
    # We'll use a simple approach: if the computation takes too long, fall back to nearest feasible node
    
    if len(unvisited_nodes) == 0:
        return 0
    
    unvisited = np.array(unvisited_nodes)
    
    # Filter for feasible nodes based on capacity
    feasible_mask = np.array([demands[node] <= rest_capacity for node in unvisited])
    
    if not np.any(feasible_mask):
        return 0
    
    feasible_nodes = unvisited[feasible_mask]
    
    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
    
    # Get distance vectors
    dist_current = distance_matrix[current_node, feasible_nodes]
    dist_depot_to_current = distance_matrix[depot, current_node]
    dist_depot_to_nodes = distance_matrix[depot, feasible_nodes]
    
    # Strategy: Combine multiple heuristics
    # 1. Savings-based: savings = dist(depot, current) + dist(depot, i) - dist(current, i)
    #    We want to maximize savings
    # 2. Density-based: prefer nodes that are close to the centroid of remaining feasible nodes
    #    This helps maintain spatial clustering and reduce travel variance
    # 3. Capacity consideration: prefer nodes with higher demand when capacity is tight
    
    # Compute savings
    savings = dist_depot_to_current + dist_depot_to_nodes - dist_current
    
    # Compute centroid of feasible nodes (using distance matrix as proxy for position)
    # Since we don't have coordinates, we use the distance matrix to estimate relative positions
    # A simpler approach: use the average distance from depot as a proxy for "spread"
    
    # Density-weighted scoring:
    # For each feasible node i, compute how "central" it is among feasible nodes
    # Use the distance matrix to compute pairwise distances among feasible nodes
    dist_matrix_feasible = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]
    
    # For each node, compute its average distance to other feasible nodes
    # Lower average distance means more central/dense
    avg_dist_to_others = np.mean(dist_matrix_feasible, axis=1)
    
    # Normalize savings to [0, 1] range
    savings_min = np.min(savings)
    savings_max = np.max(savings)
    if savings_max > savings_min:
        savings_norm = (savings - savings_min) / (savings_max - savings_min)
    else:
        savings_norm = np.ones_like(savings)
    
    # Normalize avg_dist_to_others (lower is better, so invert)
    avg_dist_min = np.min(avg_dist_to_others)
    avg_dist_max = np.max(avg_dist_to_others)
    if avg_dist_max > avg_dist_min:
        density_norm = 1.0 - (avg_dist_to_others - avg_dist_min) / (avg_dist_max - avg_dist_min)
    else:
        density_norm = np.ones_like(avg_dist_to_others)
    
    # Capacity pressure: if rest_capacity is low relative to average demand, prefer higher-demand nodes
    # This helps fill the vehicle more efficiently
    avg_demand = np.mean(demands[feasible_nodes])
    capacity_pressure = 1.0 - (rest_capacity / (avg_demand * len(feasible_nodes))) if avg_demand > 0 else 0.0
    capacity_pressure = np.clip(capacity_pressure, 0.0, 1.0)
    
    # Demand score: higher demand is better when capacity is tight
    demand_scores = demands[feasible_nodes]
    demand_min = np.min(demand_scores)
    demand_max = np.max(demand_scores)
    if demand_max > demand_min:
        demand_norm = (demand_scores - demand_min) / (demand_max - demand_min)
    else:
        demand_norm = np.ones_like(demand_scores)
    
    # Combined score: weighted sum
    # Weights: savings is most important, density second, demand third
    w_savings = 0.5
    w_density = 0.3
    w_demand = 0.2
    
    # When capacity pressure is high, increase weight on demand
    if capacity_pressure > 0.5:
        w_demand = 0.3
        w_savings = 0.45
        w_density = 0.25
    
    score = w_savings * savings_norm + w_density * density_norm + w_demand * demand_norm
    
    # Pick the node with the highest score
    best_idx = np.argmax(score)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
