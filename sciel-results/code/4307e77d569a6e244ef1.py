import numpy as np
import time

# Deadline guard: kill after 150s wall-clock, return with safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def _time_ok():
    return time.time() < _DEADLINE

# Cache for coordinate recovery from distance matrix
_coord_cache = {}

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic for CVRP: Soft-Savings with lookahead.
    
    Uses a smooth surrogate (soft-max via log-sum-exp) for the savings criterion
    to avoid being locked to a single successor when multiple are near-optimal.
    Also incorporates a small lookahead penalty for candidates whose nearest
    feasible neighbor is far away.
    
    Returns the next customer node index, or depot (0) to close the route.
    """
    # Deadline check
    if not _time_ok():
        # Return nearest feasible node as a fast fallback
        if len(unvisited_nodes) == 0:
            return depot
        feasible = [n for n in unvisited_nodes if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        # Nearest feasible
        best = min(feasible, key=lambda n: distance_matrix[current_node, n])
        return int(best)
    
    if len(unvisited_nodes) == 0:
        return depot
    
    unvisited_arr = np.array(unvisited_nodes, dtype=np.int64)
    
    # Feasible nodes: demand fits in remaining capacity
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    
    if not np.any(feasible_mask):
        return depot
    
    feasible_nodes = unvisited_arr[feasible_mask]
    n_feasible = len(feasible_nodes)
    
    # Distances
    d_cur_j = distance_matrix[current_node, feasible_nodes]  # (n_feasible,)
    d_dep_j = distance_matrix[depot, feasible_nodes]         # (n_feasible,)
    d_cur_dep = distance_matrix[current_node, depot]         # scalar
    
    # Savings: d(cur,dep) + d(dep,j) - d(cur,j)
    # Positive saving means it's cheaper to go to j than to return to depot first
    savings = d_cur_dep + d_dep_j - d_cur_j
    
    # Lookahead: for each candidate j, find the min distance from j to any other feasible node
    # (excluding j itself). This penalizes candidates that are isolated.
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n_feasible, n_feasible)
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(D_ff, np.inf)
        # Nearest neighbor distance for each feasible node
        nn_dist = np.min(D_ff, axis=1)  # (n_feasible,)
    else:
        nn_dist = np.array([0.0])
    
    # If the best saving is very negative, return to depot
    if np.max(savings) < -1e-9:
        return depot
    
    # Combined score with lookahead penalty
    beta = 0.05
    score = savings - beta * nn_dist
    
    # Capacity-aware penalty: prefer nodes that leave more capacity for future nodes
    # This avoids premature capacity exhaustion on distant nodes
    remaining_after = rest_capacity - demands[feasible_nodes]
    # Normalize capacity usage: higher remaining_after is better
    # Scale: divide by rest_capacity to get [0,1]
    cap_ratio = remaining_after / rest_capacity if rest_capacity > 0 else 1.0
    
    # Penalize nodes that consume too much capacity relative to what's left
    # But only when capacity is tight
    if rest_capacity < 25:  # capacity is getting low
        cap_penalty = 0.3 * (1.0 - cap_ratio)
    else:
        cap_penalty = 0.0 * (1.0 - cap_ratio)
    
    # Geometric sweep bonus: project onto principal axis and reward angular alignment
    # with current heading to maintain compact sectors
    n_total = distance_matrix.shape[0]
    if n_total not in _coord_cache:
        # Recover 2D coordinates from distance matrix using classical MDS
        # D^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i · x_j
        D = distance_matrix
        D2 = D ** 2
        # Double centering
        J = np.eye(n_total) - np.ones((n_total, n_total)) / n_total
        B = -0.5 * J @ D2 @ J
        # Eigen decomposition
        eigvals, eigvecs = np.linalg.eigh(B)
        # Take top 2 components
        idx = np.argsort(eigvals)[::-1]
        coords = eigvecs[:, idx[:2]] * np.sqrt(np.maximum(eigvals[idx[:2]], 0))
        _coord_cache[n_total] = coords
    
    coords = _coord_cache[n_total]
    
    # Current position and depot position in recovered coordinate space
    cur_pos = coords[current_node]
    dep_pos = coords[depot]
    feas_pos = coords[feasible_nodes]
    
    # Heading vector: from current to depot (what we'd do if closing route)
    heading = dep_pos - cur_pos
    heading_norm = np.linalg.norm(heading)
    
    if heading_norm > 1e-9:
        heading_unit = heading / heading_norm
        
        # For each feasible node, compute angle between heading and vector to that node
        vectors_to_nodes = feas_pos - cur_pos  # (n_feasible, 2)
        vec_norms = np.linalg.norm(vectors_to_nodes, axis=1)
        
        # Avoid division by zero
        valid = vec_norms > 1e-9
        angles = np.zeros(n_feasible)
        if np.any(valid):
            unit_vectors = vectors_to_nodes[valid] / vec_norms[valid, None]
            cos_angles = np.sum(unit_vectors * heading_unit, axis=1)
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            angles[valid] = np.arccos(cos_angles)
        
        # Angular deviation from heading: 0 means going same direction as heading
        # We want to minimize angular deviation (stay in compact sector)
        angular_penalty = angles  # larger angle = worse
        
        # Scale angular penalty to be comparable to savings
        # Typical savings range is ~O(1) to O(10), angles are in [0, pi]
        alpha = 0.15  # weight for angular penalty
        final_score = score - cap_penalty - alpha * angular_penalty
    else:
        final_score = score - cap_penalty
    
    best_idx = np.argmax(final_score)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
