"""Rewritten program: TSP constructive heuristic using Geometric Area and Regret."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline enforcement
_start_time = time.time()
_TOTAL_LIMIT = 150.0  # Evaluator limit
_MARGIN = 5.0         # Safety margin

def _is_deadline_passed():
    """Check if we are close to the time limit."""
    return (time.time() - _start_time) > (_TOTAL_LIMIT - _MARGIN)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node to visit in the TSP tour.
    
    Strategy: Geometric Area Minimization with Regret.
    
    1. Immediate cost: d(current, u)
    2. Geometric Tightness: Area of triangle (current, destination, u).
       Minimizing this area encourages the tour to stay close to the 
       "chord" connecting current and destination, preserving local 
       convexity and avoiding wide detours.
    3. Regret: How much more expensive u becomes if we pick a different node next.
    
    score(u) = d(current, u) + w_area * Area(current, dest, u)^2 + w_regret * regret(u)
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return -1
        
    if n == 1:
        return int(unvisited_nodes[0])
        
    # Enforce deadline: if time is running out, fall back to fast 1-step NN
    if _is_deadline_passed():
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    # 1. Distance from current node to all unvisited candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Distance from each unvisited node to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Distance between current and destination (constant for this step)
    d_cd = distance_matrix[current_node, destination_node]
    
    # 4. Compute Area^2 for each candidate u using Heron's formula
    # Sides: a = d_cd, b = d_curr[u], c = d_dest[u]
    # s = (a + b + c) / 2
    # Area^2 = s * (s-a) * (s-b) * (s-c)
    
    # Vectorized calculation
    # b = d_curr (shape n)
    # c = d_dest (shape n)
    # a = d_cd (scalar)
    
    s = (d_cd + d_curr + d_dest) / 2.0
    area_sq = s * (s - d_cd) * (s - d_curr) * (s - d_dest)
    
    # Ensure non-negative due to floating point errors
    area_sq = np.maximum(area_sq, 0.0)
    
    # 5. Compute Regret for each candidate u
    # regret(u) = min_{v != u} [d(current, v) + d(v, u)] - d(current, u)
    
    d_sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    d_sub_T = d_sub.T
    
    # combined_regret[u, v] = d_curr[v] + d_sub_T[u, v]
    combined_regret = d_curr[np.newaxis, :] + d_sub_T
    np.fill_diagonal(combined_regret, np.inf)
    
    min_cost_if_not_u = np.min(combined_regret, axis=1)
    regret = min_cost_if_not_u - d_curr
    
    # 6. Normalize terms to combine them effectively
    # We want to minimize the score.
    
    # Normalize d_curr
    curr_min = np.min(d_curr)
    curr_max = np.max(d_curr)
    if curr_max > curr_min:
        curr_norm = (d_curr - curr_min) / (curr_max - curr_min)
    else:
        curr_norm = np.zeros(n)
        
    # Normalize area_sq
    area_min = np.min(area_sq)
    area_max = np.max(area_sq)
    if area_max > area_min:
        area_norm = (area_sq - area_min) / (area_max - area_min)
    else:
        area_norm = np.zeros(n)
        
    # Normalize regret
    reg_min = np.min(regret)
    reg_max = np.max(regret)
    if reg_max > reg_min:
        regret_norm = (regret - reg_min) / (reg_max - reg_min)
    else:
        regret_norm = np.zeros(n)
        
    # 7. Combine scores
    # Weights:
    # - Immediate distance is the most important.
    # - Area term penalizes detours from the destination.
    # - Regret term ensures we don't leave "stranded" nodes.
    
    w_curr = 1.0
    w_area = 0.5
    w_regret = 0.5
    
    scores = w_curr * curr_norm + w_area * area_norm - w_regret * regret_norm
    
    # 8. Select the candidate with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

# EVOLVE-BLOCK-END
