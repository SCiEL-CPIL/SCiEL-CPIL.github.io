"""CVRP heuristic using savings-based route construction with demand-aware scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_TIMEOUT = 115.0  # Safety margin before 120s limit
_DEADLINE = None

# Precomputed savings matrix cache
_savings_cache = {}


def _compute_savings(distance_matrix, depot):
    """Compute pairwise savings for all node pairs."""
    n = distance_matrix.shape[0]
    savings = np.zeros((n, n))
    for i in range(1, n):
        for j in range(i + 1, n):
            savings[i, j] = distance_matrix[depot, i] + distance_matrix[depot, j] - distance_matrix[i, j]
            savings[j, i] = savings[i, j]
    return savings


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Selects the next node using a savings-inspired heuristic with demand efficiency.
    
    The heuristic scores each feasible candidate node j by:
    1. The savings of adding j to the current route (compared to serving j separately)
    2. Demand efficiency (prefer nodes that use capacity well)
    3. Proximity to current node
    
    Returns 0 (depot) if no feasible node can be added or if the time limit is reached.
    """
    global _start_time, _DEADLINE, _savings_cache
    
    if _start_time is None:
        _start_time = time.time()
        _DEADLINE = _start_time + _TIMEOUT
    
    # Deadline guard
    if time.time() > _DEADLINE:
        return int(depot)
    
    # Convert unvisited_nodes to numpy array
    if isinstance(unvisited_nodes, (list, tuple)):
        unvisited_arr = np.array(unvisited_nodes, dtype=int)
    else:
        unvisited_arr = np.asarray(unvisited_nodes, dtype=int)
    
    # If no unvisited nodes, return depot
    if len(unvisited_arr) == 0:
        return int(depot)
    
    # Find feasible nodes
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return int(depot)
    
    feasible_nodes = unvisited_arr[feasible_mask]
    
    # At depot: start with the node that has the highest "savings potential"
    # i.e., the node farthest from depot (since savings = d(depot,i) + d(depot,j) - d(i,j),
    # starting with a far node creates more opportunity for high-savings merges)
    # But we also want to be practical: pick the one with good savings with other unvisited nodes.
    if current_node == depot:
        # Score: prefer nodes with high distance to depot (more savings potential)
        # but also consider connectivity to other unvisited nodes
        d_depot = distance_matrix[depot, feasible_nodes]
        # Average distance from each feasible node to other unvisited nodes
        # (lower avg dist = better connectivity = more savings opportunities)
        n_unvisited = len(unvisited_arr)
        if n_unvisited > 1:
            # Compute average distance from each feasible node to all unvisited nodes
            avg_dist_to_unvisited = np.array([
                np.mean(distance_matrix[f, unvisited_arr]) for f in feasible_nodes
            ])
            # Normalize
            max_avg = np.max(avg_dist_to_unvisited) if np.max(avg_dist_to_unvisited) > 0 else 1.0
            norm_avg = avg_dist_to_unvisited / max_avg
            
            # Score: combine high depot distance (savings potential) with low avg distance (connectivity)
            max_depot = np.max(d_depot) if np.max(d_depot) > 0 else 1.0
            norm_depot = d_depot / max_depot
            
            # We want high depot distance AND low avg distance to unvisited
            scores = 0.6 * norm_depot - 0.4 * norm_avg
            best_idx = np.argmax(scores)
        else:
            # Only one unvisited node, just pick it
            best_idx = 0
        
        return int(feasible_nodes[best_idx])
    
    # Not at depot: use savings-based scoring
    # For each candidate j, compute the "savings" of visiting j now vs. later
    # Savings of adding j to current route: d(current, j) + d(j, depot) vs d(current, depot) + d(depot, j)
    # But since we're already in a route, the relevant comparison is:
    # Cost to go current->j->depot vs current->depot (if j were served separately)
    # Savings = d(current, depot) + d(depot, j) - d(current, j) - d(j, depot)
    #         = d(current, depot) - d(current, j)
    # So higher savings means d(current, j) is small relative to d(current, depot)
    
    d_current_j = distance_matrix[current_node, feasible_nodes]
    d_current_depot = distance_matrix[current_node, depot]
    d_depot_j = distance_matrix[depot, feasible_nodes]
    
    # Savings from adding j to this route
    savings = d_current_depot - d_current_j
    
    # Demand efficiency: prefer nodes with higher demand (uses capacity better)
    # Normalize demand
    demands_j = demands[feasible_nodes]
    max_demand = np.max(demands_j) if np.max(demands_j) > 0 else 1.0
    norm_demand = demands_j / max_demand
    
    # Also consider: how well does j connect to remaining unvisited nodes?
    # Nodes that are close to other unvisited nodes create more future savings
    n_unvisited = len(unvisited_arr)
    if n_unvisited > 1:
        # Average distance from each candidate to other unvisited nodes (excluding itself)
        # Lower is better (more connectivity)
        avg_conn = np.array([
            np.mean(distance_matrix[f, unvisited_arr]) - distance_matrix[f, f] / max(n_unvisited - 1, 1)
            for f in feasible_nodes
        ])
        max_conn = np.max(avg_conn) if np.max(avg_conn) > 0 else 1.0
        norm_conn = avg_conn / max_conn
    else:
        norm_conn = np.zeros(len(feasible_nodes))
    
    # Combine scores:
    # 1. Savings (higher is better) - primary
    # 2. Demand efficiency (higher is better) - secondary  
    # 3. Connectivity to unvisited (lower is better) - tertiary
    max_savings = np.max(savings) if np.max(savings) > 0 else 1.0
    norm_savings = savings / max_savings
    
    # Ensure norm_savings is in [0, 1] range
    norm_savings = np.clip(norm_savings, 0, 1)
    
    # Weighted score (higher is better)
    w_savings = 0.5
    w_demand = 0.2
    w_conn = 0.3
    
    scores = w_savings * norm_savings + w_demand * norm_demand - w_conn * norm_conn
    
    # Find best candidate
    best_idx = np.argmax(scores)
    best_node = feasible_nodes[best_idx]
    
    # Route closing decision:
    # Check if it's better to return to depot than to visit best_node
    # Cost of visiting best_node then returning: d(current, best_node) + d(best_node, depot)
    # Cost of returning now: d(current, depot)
    cost_visit = d_current_j[best_idx] + d_depot_j[best_idx]
    cost_return = d_current_depot
    
    # If visiting the node costs more than 1.2x just returning, close the route
    # This threshold is conservative to avoid premature route closing
    if cost_visit > cost_return * 1.2:
        return int(depot)
    
    return int(best_node)
# EVOLVE-BLOCK-END
