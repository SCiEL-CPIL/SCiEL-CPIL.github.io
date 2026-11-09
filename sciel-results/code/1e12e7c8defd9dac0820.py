import numpy as np
import time

_start_time = time.time()
TIME_LIMIT = 145.0

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    # Time guard
    if time.time() - _start_time > TIME_LIMIT:
        if not unvisited_nodes:
            return 0
        un_list = list(unvisited_nodes)
        feasible = [u for u in un_list if demands[u] <= rest_capacity]
        if not feasible:
            return 0
        return int(min(feasible, key=lambda u: distance_matrix[current_node][u]))

    # Convert to numpy
    if not isinstance(unvisited_nodes, np.ndarray):
        unvisited_nodes = np.array(unvisited_nodes, dtype=int)
    if not isinstance(demands, np.ndarray):
        demands = np.array(demands)
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)

    n = len(unvisited_nodes)
    if n == 0:
        return 0

    # Feasibility
    node_demands = demands[unvisited_nodes]
    feasible_mask = node_demands <= rest_capacity
    feasible_nodes = unvisited_nodes[feasible_mask]
    n_feas = len(feasible_nodes)

    if n_feas == 0:
        return 0

    if n_feas == 1:
        return int(feasible_nodes[0])

    # --- Structural Change: Multi-step lookahead with capacity-aware simulation ---
    # Instead of a single-step score, we simulate the completion of the current route
    # for the top K candidates and pick the one that minimizes the total route length
    # (including the return to depot).

    # Get candidates: top K by distance from current, but also ensure diversity
    # If n_feas is small, evaluate all.
    K = 5
    if n_feas <= K:
        candidates = feasible_nodes
    else:
        dists = distance_matrix[current_node, feasible_nodes]
        idx_sorted = np.argsort(dists)
        candidates = feasible_nodes[idx_sorted[:K]]

    best_candidate = -1
    best_cost = float('inf')

    # Precompute submatrix for feasible nodes for fast simulation
    # We only need distances among feasible_nodes
    # Note: simulation will consume nodes, so we need a mask of remaining feasible nodes

    for cand in candidates:
        # Simulate route: current -> cand -> ... -> depot
        # State:
        #   curr: current position (starts at 'current_node')
        #   remaining_mask: boolean array over feasible_nodes indicating which are unvisited
        #   cap: remaining capacity
        
        # Reset state
        # We represent the set of remaining feasible nodes as a list or set?
        # For speed with N=50, a list/set of indices into the feasible_nodes array is good.
        # But distance lookups are O(1) via matrix.
        
        # Let's use a boolean mask over feasible_nodes for speed in numpy ops?
        # Actually, a simple list of remaining node IDs might be easier for the greedy loop.
        
        remaining = list(feasible_nodes)
        remaining.remove(cand) # cand is visited first
        
        cap = rest_capacity - demands[cand]
        curr = cand
        cost = distance_matrix[current_node, cand]
        
        while remaining:
            # Find next node from 'remaining' that fits in cap
            # If none, break (route closes, go to depot)
            
            # Filter remaining by capacity
            # demands is a global array, so demands[remaining] works
            valid_mask = np.array([demands[u] <= cap for u in remaining])
            valid_remaining = np.array(remaining)[valid_mask]
            
            if len(valid_remaining) == 0:
                break
            
            # Pick nearest neighbor from valid_remaining
            # dists from curr to valid_remaining
            d = distance_matrix[curr, valid_remaining]
            next_idx = np.argmin(d)
            next_node = valid_remaining[next_idx]
            
            cost += d[next_idx]
            curr = next_node
            cap -= demands[curr]
            
            # Remove next_node from remaining
            # Since remaining is a list, we can use remove, but it's O(N).
            # For N=50, this is fine.
            remaining.remove(next_node)
            
        # End of simulation loop
        # Add distance to depot
        cost += distance_matrix[curr, depot]
        
        if cost < best_cost:
            best_cost = cost
            best_candidate = cand

    return int(best_candidate)
