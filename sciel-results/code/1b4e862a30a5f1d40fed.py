"""Capacitated VRP heuristic: curvature-minimizing smooth path with look-ahead."""
# EVOLVE-BLOCK-START
import numpy as np
import math
import time

# Deadline guard: track start time to avoid exceeding wall-clock limit
_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 150s limit

def _elapsed():
    return time.time() - _start_time

def _time_exceeded():
    return _elapsed() > (150.0 - _DEADLINE_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next node to visit using a curvature-minimizing smooth-path heuristic.
    
    Strategy:
    - If no unvisited nodes, return depot.
    - Filter unvisited nodes by capacity feasibility.
    - Among feasible nodes, score each candidate by:
      1. Local curvature: how much the path bends if we go current -> candidate -> next_best
         (lower curvature = smoother path = less backtracking)
      2. Distance from current (to avoid long jumps)
      3. Distance to depot (to encourage compact routes that return easily)
      4. Capacity-tightness penalty: prefer nodes whose demand leaves more room
         for future insertions (avoids dead-ends when capacity is low)
    - If no feasible node, return depot.
    
    The curvature term approximates the turning angle at the candidate node.
    We estimate "next_best" as the nearest feasible neighbor of the candidate
    (or depot if none), then compute the angle at the candidate between
    (current -> candidate) and (candidate -> next_best).
    """
    if len(unvisited_nodes) == 0:
        return 0
    
    # Time guard: if we're running out of time, fall back to simple nearest-neighbor
    if _time_exceeded():
        unvisited = np.array(unvisited_nodes)
        dists = distance_matrix[current_node, unvisited]
        demands_arr = np.array([demands[node] for node in unvisited])
        feasible_mask = demands_arr <= rest_capacity
        feasible_indices = np.where(feasible_mask)[0]
        if len(feasible_indices) == 0:
            return 0
        feasible_nodes = unvisited[feasible_indices]
        feasible_dists = dists[feasible_indices]
        best_idx = np.argmin(feasible_dists)
        return int(feasible_nodes[best_idx])
    
    unvisited = np.array(unvisited_nodes)
    n = len(unvisited)
    
    # Get distances from current node to all unvisited
    dists = distance_matrix[current_node, unvisited]
    
    # Filter by capacity
    demands_arr = np.array([demands[node] for node in unvisited])
    feasible_mask = demands_arr <= rest_capacity
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) == 0:
        return 0
    
    feasible_nodes = unvisited[feasible_indices]
    feasible_dists = dists[feasible_indices]
    feasible_demands = demands_arr[feasible_indices]
    
    # Distance from each feasible node to depot
    dists_to_depot = distance_matrix[depot, feasible_nodes]
    
    # For curvature computation, we need to estimate the "next best" node after each candidate.
    # Approximation: for each candidate c, the next best is the nearest feasible node
    # (excluding c itself) or the depot if no other feasible nodes exist.
    # We compute this efficiently.
    
    n_feasible = len(feasible_nodes)
    
    # Precompute: for each feasible node, find its nearest neighbor among all feasible nodes
    # (or depot). This is O(n_feasible^2) but n_feasible is at most 50, so it's fine.
    # dist_from_candidate[i, j] = distance from feasible_nodes[i] to feasible_nodes[j]
    # Also include depot as a "neighbor" option.
    
    # Build a matrix: rows are feasible nodes, columns are [feasible nodes..., depot]
    # dist_matrix_feasible[i, j] for j in range(n_feasible) = distance(feasible_nodes[i], feasible_nodes[j])
    # dist_matrix_feasible[i, n_feasible] = distance(feasible_nodes[i], depot)
    
    dist_mat_feas = np.zeros((n_feasible, n_feasible + 1))
    for i in range(n_feasible):
        dist_mat_feas[i, :n_feasible] = distance_matrix[feasible_nodes[i], feasible_nodes]
        dist_mat_feas[i, n_feasible] = distance_matrix[feasible_nodes[i], depot]
    
    # For each candidate i, the "next best" is the argmin of dist_mat_feas[i] excluding self
    # But we also want to consider that after going to candidate i, the remaining capacity
    # decreases. However, for the curvature estimate, we just use the nearest neighbor.
    
    # For curvature: we need the angle at candidate i between:
    #   vector from current_node to candidate_i, and
    #   vector from candidate_i to next_best_i
    # 
    # We don't have coordinates, only distances. We can use the law of cosines to
    # compute the angle if we know all three sides of the triangle:
    #   side_a = dist(current, candidate_i)
    #   side_b = dist(candidate_i, next_best_i)  
    #   side_c = dist(current, next_best_i)
    # angle at candidate_i: cos(theta) = (a^2 + b^2 - c^2) / (2*a*b)
    # theta = arccos(clamp(cos_theta, -1, 1))
    # Curvature penalty = theta (in radians), we want to minimize this.
    
    # Actually, for a smooth path, we want the angle to be close to pi (straight line).
    # So we penalize deviation from pi: penalty = pi - theta, or equivalently,
    # we want to maximize cos(theta) (since cos(pi) = -1, cos(0) = 1).
    # Wait: if the path goes current -> candidate -> next_best in a straight line,
    # the angle at candidate is pi (180 degrees). cos(pi) = -1.
    # If the path makes a sharp turn (U-turn), angle is 0, cos(0) = 1.
    # So we want to minimize (1 - cos(theta)) which is 0 for straight line and 2 for U-turn.
    # Or equivalently, maximize cos(theta) (most negative = straightest).
    
    # Let's compute curvature score for each candidate.
    curvature_scores = np.zeros(n_feasible)
    
    for i in range(n_feasible):
        # side_a: dist(current_node, feasible_nodes[i])
        side_a = dists[feasible_indices[i]]
        
        if side_a < 1e-9:
            curvature_scores[i] = 0.0
            continue
        
        # Find next_best: nearest neighbor of feasible_nodes[i] among all feasible + depot
        # Exclude self (index i)
        dist_row = dist_mat_feas[i].copy()
        dist_row[i] = np.inf  # exclude self
        
        next_best_idx = np.argmin(dist_row)
        side_b = dist_row[next_best_idx]
        
        if side_b < 1e-9:
            curvature_scores[i] = 0.0
            continue
        
        # side_c: dist(current_node, next_best_node)
        if next_best_idx < n_feasible:
            next_best_node = feasible_nodes[next_best_idx]
        else:
            next_best_node = depot
        
        side_c = distance_matrix[current_node, next_best_node]
        
        # Law of cosines: cos(theta) = (a^2 + b^2 - c^2) / (2*a*b)
        cos_theta = (side_a**2 + side_b**2 - side_c**2) / (2.0 * side_a * side_b)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        # We want to maximize cos_theta (straighter is better).
        # Curvature penalty = 1 - cos_theta (0 for straight, 2 for U-turn)
        curvature_scores[i] = 1.0 - cos_theta
    
    # Capacity-tightness penalty: prefer nodes that leave more residual capacity.
    # After visiting node i, remaining capacity = rest_capacity - demand[i].
    # We penalize candidates that leave very little capacity, to avoid dead-ends.
    residual_cap = rest_capacity - feasible_demands
    # Normalize: penalty is higher when residual_cap is small relative to rest_capacity
    if rest_capacity > 0:
        cap_penalty = 1.0 - (residual_cap / rest_capacity)
    else:
        cap_penalty = np.ones(n_feasible)
    
    # Combined score:
    # Primary: distance from current (shorter is better)
    # Secondary: curvature (smoother is better)
    # Tertiary: distance to depot (closer is better for compact routes)
    # Quaternary: capacity tightness (more residual is better)
    
    # Normalize each component to [0, 1] range for balanced weighting
    dist_norm = feasible_dists / (np.max(feasible_dists) + 1e-9)
    curv_norm = curvature_scores / (np.max(curvature_scores) + 1e-9)
    depot_norm = dists_to_depot / (np.max(dists_to_depot) + 1e-9)
    
    # Weights: distance is most important, then curvature, then depot proximity, then capacity
    w_dist = 1.0
    w_curv = 0.4
    w_depot = 0.2
    w_cap = 0.15
    
    scores = (w_dist * dist_norm + 
              w_curv * curv_norm + 
              w_depot * depot_norm + 
              w_cap * cap_penalty)
    
    best_idx = np.argmin(scores)
    return int(feasible_nodes[best_idx])
# EVOLVE-BLOCK-END
