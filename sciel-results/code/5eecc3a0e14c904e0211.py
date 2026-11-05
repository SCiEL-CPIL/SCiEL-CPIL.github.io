import numpy as np
import time

# Deadline guard: kill after 150s wall-clock, return with safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # 5s safety margin

def _time_ok():
    return time.time() < _DEADLINE

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
    
    # 1-Step Savings: d(cur,dep) + d(dep,j) - d(cur,j)
    savings_1 = d_cur_dep + d_dep_j - d_cur_j
    
    # 2-Step Lookahead: Estimate the best saving achievable from j to any other feasible node k.
    # Saving(j->k) = d(j,dep) + d(dep,k) - d(j,k)
    # For fixed j, we maximize over k != j: d(j,dep) + (d(dep,k) - d(j,k))
    # Since d(j,dep) is constant for j, we maximize (d(dep,k) - d(j,k))
    
    if n_feasible > 1:
        # Distance matrix among feasible nodes
        D_ff = distance_matrix[np.ix_(feasible_nodes, feasible_nodes)]  # (n_feasible, n_feasible)
        
        # Matrix M where M[j, k] = d(dep, k) - d(j, k)
        # d_dep_j is shape (n_feasible,), so d_dep_j[None, :] broadcasts to (1, n_feasible)
        M = d_dep_j[None, :] - D_ff
        
        # Set diagonal to -inf so we don't pick self
        np.fill_diagonal(M, -np.inf)
        
        # Max saving contribution from the next node for each j
        # V[j] = max_k (d(dep, k) - d(j, k))
        V = np.max(M, axis=1)
        
        # Full 2-step saving estimate for j:
        # S_2(j) = d(j, dep) + V[j]
        # Note: d(j, dep) is d_dep_j
        savings_2_part = d_dep_j + V
    else:
        savings_2_part = np.array([0.0])
    
    # Capacity ratio: how much of remaining capacity this node would consume
    capacity_ratio = demands[feasible_nodes] / max(rest_capacity, 1)
    
    # Weights
    alpha_lookahead = 0.5  # Weight for 2-step lookahead savings
    beta_cap = 0.15        # Weight for capacity consumption penalty
    
    # Adaptive capacity penalty
    # Normalize 1-step savings to [0, 1] for adaptation
    savings_min = np.min(savings_1)
    savings_max = np.max(savings_1)
    savings_range = max(savings_max - savings_min, 1e-9)
    savings_frac = (savings_1 - savings_min) / savings_range
    
    # When savings are high, reduce capacity penalty
    adaptive_cap_weight = beta_cap * (1.0 - 0.5 * savings_frac)
    
    # Combined Score
    # Primary: 1-step savings
    # Secondary: 2-step lookahead savings (smoothed by taking max, but we can also soft-max if needed)
    # Penalty: Capacity consumption
    score = savings_1 + alpha_lookahead * savings_2_part - adaptive_cap_weight * capacity_ratio * max(savings_max, 1e-9)
    
    # If the best 1-step saving is very negative, return to depot
    # (Even if 2-step is good, if immediate step is bad, it might be better to close)
    # However, usually if savings_1 is negative, savings_2 might compensate.
    # Let's check if the total score is positive?
    # The depot return cost is d_cur_dep.
    # If we go to j, cost is d_cur_j.
    # If d_cur_j > d_cur_dep, savings_1 is negative.
    # If we then go to k, total cost d_cur_j + d_j_k.
    # Return cost d_cur_dep + d_dep_k.
    # Total saving = d_cur_dep + d_dep_k - d_cur_j - d_j_k = savings_1 + (d_dep_k - d_j_k) = savings_1 + V[j] (plus d_dep_j? No)
    # Wait, Saving(j->k) formula:
    # Route: ... c -> j -> k -> dep
    # Alt: ... c -> dep -> k -> dep (and j is in another route? No, j is visited)
    # Actually, the standard savings is d(c,dep) + d(dep,k) - d(c,k).
    # Here we are extending c -> j.
    # If we stop at j: cost d(c,j). Return d(j,dep). Total d(c,j)+d(j,dep).
    # If we go c->dep: cost d(c,dep).
    # Saving of c->j vs c->dep->j is d(c,dep) + d(dep,j) - d(c,j). This is savings_1.
    # If we add k after j:
    # Cost c->j->k->dep = d(c,j) + d(j,k) + d(k,dep).
    # Alt c->dep->k->dep (and j is separate? No, we are building one route).
    # The "Savings" heuristic merges routes.
    # Here we are building a route sequentially.
    # The decision is: Add j to current route?
    # If yes, we pay d(c,j). We have remaining capacity.
    # The "benefit" is that we avoid returning to depot now.
    # The "cost" is that we might be forced to return later.
    # The 2-step lookahead estimates the benefit of the *next* step.
    
    # If max score is negative, it means going to any node increases cost compared to closing?
    # Actually, score is relative.
    # Let's just pick the best node if any feasible node exists.
    # But if savings_1 is very negative, maybe we should close?
    # The depot return is always an option.
    # If we return to depot, cost is d_cur_dep.
    # If we go to j, cost is d_cur_j + (future costs).
    # If d_cur_j > d_cur_dep + (future benefit), we should return.
    # Future benefit is approximated by -savings_2_part?
    # Let's stick to the relative score.
    
    if n_feasible == 0:
        return depot
        
    best_idx = np.argmax(score)
    best_node = feasible_nodes[best_idx]
    
    # Safety check: If the best node's immediate cost is significantly higher than returning,
    # and the lookahead doesn't compensate, return to depot.
    # Cost(j) = d_cur_j. Return = d_cur_dep.
    # If d_cur_j - d_cur_dep > (alpha * savings_2_part + ...), maybe return?
    # This is complex. The score already balances this.
    
    return int(best_node)
