"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 150.0
_SAFETY_MARGIN = 2.0

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Savings-based selection.
    Strategy:
    1. Filter feasible nodes by capacity.
    2. If none, return depot.
    3. For each feasible candidate, calculate a "savings" score:
       Savings = d(current, depot) + d(depot, candidate) - d(current, candidate)
       This metric is high when visiting the candidate now saves more distance 
       than going back to depot and then visiting it later (or vice versa).
       Actually, standard savings is d(i, depot) + d(depot, j) - d(i, j).
       Here, we are extending a route from current_node.
       If we go to candidate now, cost is d(current, candidate) + future_cost.
       If we go to depot now, cost is d(current, depot) + future_cost (starting new route).
       
       A better proxy for "which node to visit next to minimize total length":
       We want to minimize d(current, candidate) + d(candidate, depot) 
       IF this is likely the last node.
       But if it's not the last, we want to minimize d(current, candidate) + min_d(candidate, others).
       
       Let's use a hybrid:
       Score = d(current, candidate) + lambda * d(candidate, depot)
       Where lambda depends on how many nodes are left or capacity.
       
       Alternative: "Nearest Neighbor with Depot Pull"
       Score = d(current, candidate) + alpha * d(candidate, depot)
       
       Let's try a "Regret" based approach or "Savings" adaptation.
       
       Standard Clarke-Worth Savings for merging routes:
       S(i, j) = d(i, 0) + d(0, j) - d(i, j)
       
       In a single route construction context (greedy insertion/extension):
       We are at node 'current'. We have a set of unvisited 'U'.
       If we pick 'c' next, the immediate cost is d(current, c).
       The "regret" of not picking 'c' is high if d(current, c) is small 
       AND d(c, other) is large (it gets expensive to reach it later).
       
       Let's try: Score = d(current, c) - beta * min_{o in U, o!=c} d(c, o)
       This rewards nodes that are close to current AND close to other unvisited nodes 
       (keeping the cluster tight).
       
       Wait, the current program does: immediate + 0.2*min_other + 0.1*depot.
       It scored 0.99.
       
       Let's try a different family: **Sweep Heuristic** or **Angle-based**.
       Or **Furthest-Insertion** logic applied greedily?
       
       Let's try a **Weighted Nearest Neighbor** where the weight is the demand.
       Nodes with higher demand are "costlier" to serve if left for later because 
       they consume more capacity, potentially forcing earlier returns to depot.
       So, prioritize high demand nodes early?
       
       Score = d(current, c) / (demands[c] + 1)  -> Minimizes distance per unit demand? No.
       
       Let's try: **Minimize d(current, c) + d(c, depot) / rest_capacity**
       
       Actually, let's look at the **Savings** metric more closely.
       If we are building one route at a time (which the evaluator implies by "close a route"),
       the decision is: do we add this node to the current route, or close the route?
       But the function only picks the *next* node. It doesn't decide to close unless no feasible nodes.
       Wait, "return 0 (depot) to close a route".
       The function returns 0 IF we decide to close. 
       The current program returns depot ONLY if no feasible nodes fit.
       
       Is it better to close a route early even if feasible nodes exist?
       Yes, if the remaining feasible nodes are far away, it might be better to close 
       the current route and start a new one from the depot for them.
       
       So the strategy should be:
       1. Calculate the cost of visiting the best feasible node: 
          C_visit = min_{c in Feasible} (d(current, c) + d(c, depot))  # Assuming it's the last one?
          Or C_visit = min_{c in Feasible} d(current, c) + estimated_future_cost
       
       2. Calculate the cost of closing the route now:
          C_close = d(current, depot)
       
       3. If C_close < C_visit (with some margin), return depot.
       4. Else, return the best feasible node.
       
       How to estimate future cost?
       If we pick c, we are at c. The remaining feasible nodes are U \ {c}.
       The best case for the rest is that they are all close to c and each other.
       
       Let's implement a "Lookahead Return" strategy:
       For each feasible candidate c:
         immediate = d(current, c)
         # Heuristic for remaining route length from c:
         # Assume we visit the nearest neighbor from c next, etc. 
         # Or simply: d(c, depot) is the lower bound for closing.
         # A better estimate: d(c, depot) + 0.5 * sum of min distances?
         
       Let's use a simple regret/lookahead:
       Score(c) = d(current, c) + 0.5 * d(c, depot)
       This balances "get there" with "get home".
       
       Compare best Score(c) against d(current, depot).
       If d(current, depot) < best_Score(c) * threshold, return depot.
       
       Threshold: If going to depot is cheaper than going to the best node + coming back, 
       then close.
       Actually, if we go to depot, we pay d(current, depot).
       If we go to c, we pay d(current, c) + [cost to finish route from c].
       Min cost to finish from c is at least d(c, depot).
       So if d(current, depot) < d(current, c) + d(c, depot), it's better to close.
       
       This is equivalent to: d(current, depot) - d(current, c) < d(c, depot).
       
       Let's implement this "Close if it saves distance" logic.
    """
    if len(unvisited_nodes) == 0:
        return depot

    # Time check
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    # Filter feasible
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # Distances from current
    d_cur_to_feas = distance_matrix[current_node, feasible_nodes]
    
    # Distances from feasible to depot
    d_feas_to_depot = distance_matrix[feasible_nodes, depot]
    
    # Cost to close route now
    cost_close = distance_matrix[current_node, depot]
    
    # For each feasible node, estimate the "total cost" of picking it next.
    # Lower bound on remaining cost after picking c is d(c, depot).
    # So total cost approx = d(current, c) + d(c, depot)
    # If this is greater than cost_close, we should close.
    
    # However, if there are many nodes, we won't go directly to depot after c.
    # But d(c, depot) is a hard lower bound.
    # If d(current, c) + d(c, depot) > cost_close, it is strictly better to close 
    # (assuming we can restart from depot for the rest).
    
    # Let's find the candidate that minimizes this "Round Trip Estimate"
    round_trip_est = d_cur_to_feas + d_feas_to_depot
    
    best_idx = np.argmin(round_trip_est)
    best_candidate = feasible_nodes[best_idx]
    best_rt_cost = round_trip_est[best_idx]
    
    # Decision: Close or Visit?
    # If closing is cheaper than the best candidate's round-trip estimate, close.
    # Add a small penalty to encourage visiting if close? 
    # Or use a threshold: Close if cost_close < best_rt_cost * 0.9
    # Actually, if cost_close < best_rt_cost, closing is better in the 2-node case.
    # In the N-node case, the "future" cost of the remaining nodes is shared.
    # But if we close, the remaining nodes start from depot.
    # If we visit c, the remaining nodes start from c.
    
    # Let's stick to the simple rule:
    # If d(current, depot) < min_c (d(current, c) + d(c, depot)), return depot.
    # This is a "Greedy Savings" decision.
    
    if cost_close < best_rt_cost:
        return depot
    
    # Otherwise, return the best candidate
    return int(best_candidate)

# EVOLVE-BLOCK-END
