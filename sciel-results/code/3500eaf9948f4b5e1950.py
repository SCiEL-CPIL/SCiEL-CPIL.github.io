"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time

    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()

    # Safety margin: return depot if we've exceeded 145s (5s buffer before 150s limit)
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot

    if not unvisited_nodes:
        return depot

    unvisited = np.array(unvisited_nodes, dtype=int)
    n = len(unvisited)

    if n == 0:
        return depot

    # Filter by capacity
    dm = np.asarray(distance_matrix)
    cur = int(current_node)

    # Feasible nodes: demand fits in rest_capacity
    feasible_mask = demands[unvisited] <= rest_capacity
    feasible_nodes = unvisited[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    # Clarke-Wright savings heuristic for route construction
    # Savings for visiting node j after node i (instead of going depot->i->depot and depot->j->depot):
    # saving(i, j) = dist(depot, i) + dist(depot, j) - dist(i, j)
    # We want to maximize savings, so pick the node with the highest "savings" relative to current path.
    
    # For a greedy construction, at each step we pick the next node that gives the best marginal savings.
    # The marginal savings of adding node j after current_node (and eventually returning to depot) vs.
    # the alternative of closing the route now and starting a new one for j:
    # 
    # If we go to j then depot: cost = dist(cur, j) + dist(j, depot)
    # If we go to depot now and then to j in a new route: cost = dist(cur, depot) + dist(depot, j)
    # 
    # Savings = dist(cur, depot) + dist(depot, j) - dist(cur, j) - dist(j, depot) ... wait that's not right.
    # 
    # Actually, the standard savings approach:
    # Route: depot -> ... -> cur -> j -> depot
    # vs. two routes: depot -> ... -> cur -> depot and depot -> j -> depot
    # The savings of merging is: dist(cur, depot) + dist(depot, j) - dist(cur, j)
    # But we also need to account for the fact that we're already at cur.
    # 
    # The marginal cost of going to j and then depot vs. going to depot now:
    # Option A (go to j then depot): dist(cur, j) + dist(j, depot)
    # Option B (go to depot now): dist(cur, depot)
    # Then in a new route, we'd visit j: dist(depot, j)
    # Total for Option B: dist(cur, depot) + dist(depot, j)
    # 
    # So the "savings" of choosing j now is:
    # savings = (dist(cur, depot) + dist(depot, j)) - (dist(cur, j) + dist(j, depot))
    #         = dist(cur, depot) + dist(depot, j) - dist(cur, j) - dist(j, depot)
    #         = dist(cur, depot) - dist(cur, j)  [since dist(depot, j) == dist(j, depot)]
    
    # So we want to maximize: dist(cur, depot) - dist(cur, j)
    # Which is equivalent to minimizing: dist(cur, j) - dist(cur, depot)
    # Or: we prefer nodes j where dist(cur, j) is small relative to dist(cur, depot)
    
    # Actually, let me reconsider. The savings heuristic says:
    # When building a route, the "savings" of connecting i and j (instead of having them as separate routes)
    # is s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
    # 
    # In a greedy construction starting from the depot:
    # - Start at depot
    # - At each step, among all unvisited feasible nodes, pick the one that maximizes the savings
    #   of adding it to the current route.
    # 
    # The savings of adding node j to the end of the current route (which ends at cur):
    # If we add j, the route becomes: depot -> ... -> cur -> j -> depot
    # The additional cost is: d(cur, j) + d(j, depot) - d(cur, depot)
    # (we replace the leg cur->depot with cur->j->depot)
    # 
    # So we want to minimize: d(cur, j) + d(j, depot) - d(cur, depot)
    # Which is equivalent to minimizing: d(cur, j) + d(j, depot)
    # (since d(cur, depot) is constant for all j)
    
    # So the heuristic is: pick j that minimizes d(cur, j) + d(j, depot)
    
    dist_cur_to_feasible = dm[cur, feasible_nodes]
    dist_feasible_to_depot = dm[feasible_nodes, depot]
    
    # Score: d(cur, j) + d(j, depot)
    scores = dist_cur_to_feasible + dist_feasible_to_depot
    
    # Pick the node with the minimum score
    best_idx = np.argmin(scores)
    best_node = feasible_nodes[best_idx]
    
    return int(best_node)
# EVOLVE-BLOCK-END
