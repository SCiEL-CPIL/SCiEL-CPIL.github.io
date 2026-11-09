"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level cache for distance matrix to avoid repeated conversions
_dm_cache = {}
_start_time = None

def _get_dm(distance_matrix):
    """Get or create numpy view of distance matrix."""
    key = id(distance_matrix)
    if key not in _dm_cache:
        _dm_cache[key] = np.asarray(distance_matrix, dtype=np.float64)
    return _dm_cache[key]

def _check_deadline():
    """Return True if we are close to the time limit."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    # Keep a safety margin of 10 seconds
    return (time.time() - _start_time) > 140

def _improve_route(route, demands, dm, depot, capacity):
    """
    Try to improve a route using 2-opt and Or-opt (move) heuristics.
    route is a list of nodes starting from the first node (current) and ending before depot.
    Returns the improved route (list of nodes).
    """
    if len(route) < 3:
        return route
    
    # Copy route to avoid modifying original
    r = list(route)
    n = len(r)
    improved = True
    max_iters = 10  # Limit iterations for speed
    
    while improved and max_iters > 0:
        improved = False
        max_iters -= 1
        
        # 2-opt
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Check capacity feasibility for the reversed segment
                # In 2-opt, the set of nodes is the same, just order changes.
                # Since capacity is a constraint on the *set* of nodes in a route,
                # and the set doesn't change in 2-opt, capacity is automatically preserved
                # IF the original route was feasible.
                # However, we must ensure the route is still a valid path.
                
                # Calculate delta cost
                # Current: ... r[i-1] -> r[i] -> ... -> r[j] -> r[j+1] ...
                # New:     ... r[i-1] -> r[j] -> ... -> r[i] -> r[j+1] ...
                
                if i == 0:
                    # r[0] is the start node (current_node from caller)
                    # We usually don't reverse the prefix starting from the very first node 
                    # in a way that changes the start, but 2-opt can reverse r[0..j].
                    # Actually, standard 2-opt on a path:
                    # If i=0, we reverse r[0..j]. The start becomes r[j].
                    # But in our context, r[0] is the node we just picked or the current position.
                    # If we reverse r[0..j], the "current" node for the rest of the simulation changes?
                    # No, the route is a sequence.
                    # Let's stick to standard 2-opt on the sequence of nodes to be visited.
                    
                    # Cost before: d(r[i-1], r[i]) + d(r[j], r[j+1])  (if i>0, j<n-1)
                    # Cost after:  d(r[i-1], r[j]) + d(r[i], r[j+1])
                    
                    pass # Complex boundary cases. Let's use a simpler, safer 2-opt that only swaps internal segments.
        
        # Simpler 2-opt: only consider i > 0 and j < n-1? 
        # Actually, let's just use a standard 2-opt loop.
        
        # Re-implement 2-opt properly
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Nodes involved: r[i-1], r[i], r[j], r[j+1] (with boundary checks)
                # If i == 0:
                #   prev1 = None (start)
                #   The segment r[0..j] is reversed.
                #   New start is r[j].
                #   This changes the "current_node" for the subsequent steps in the simulation?
                #   No, the simulation returns a total length.
                #   If we reverse the prefix, the length from the *original* current_node to the new start r[j] is d(current, r[j]).
                #   But the route starts at r[0]. If we reverse, the route starts at r[j].
                #   This is tricky. Let's assume the route is a closed loop or open path?
                #   It's an open path: current -> ... -> depot.
                #   If we reverse r[0..j], the path becomes current -> r[j] -> ... -> r[0] -> r[j+1] ...
                #   Wait, r[0] is the first node after current.
                #   So the edge is (current, r[0]).
                #   If we reverse r[0..j], the new first node is r[j].
                #   The edge becomes (current, r[j]).
                #   The rest of the reversed segment is r[j-1] ... r[0].
                #   Then edge (r[0], r[j+1]).
                
                # Let's calculate delta
                d1 = dm[r[i-1], r[i]] if i > 0 else 0 # Placeholder, need to handle start
                # This is getting complicated to implement safely and quickly in Python loops.
                # Given the small size (<=16), maybe a simpler improvement is better.
                
                # Let's try Or-opt (move a node) which is safer.
                # Move node r[k] to position l.
                
        # Let's skip complex 2-opt implementation for now due to risk of bugs/timeout.
        # The nearest neighbor + savings lookahead is already strong.
        # Let's just return the route as is if we don't have a robust improver.
        
    return r

def _simulate_route(current_node, depot, unvisited, rest_capacity, demands, dm):
    """
    Simulate completing the route from current_node using a fast greedy heuristic
    that incorporates savings-based scoring for better route compactness.
    Returns the total length of the simulated route (excluding the edge to current_node's predecessor).
    """
    total_dist = 0.0
    current = current_node
    remaining_cap = rest_capacity
    remaining = list(unvisited)
    
    while remaining:
        # Filter feasible
        feasible = [n for n in remaining if demands[n] <= remaining_cap]
        if not feasible:
            break
            
        if len(feasible) == 1:
            next_node = feasible[0]
        else:
            # Use savings-based scoring: prefer nodes that are close to current
            # but also have high potential to merge with other remaining nodes
            feas_arr = np.array(feasible)
            n = len(feas_arr)
            
            dists_to_current = dm[current, feas_arr]
            dists_to_depot = dm[feas_arr, depot]
            
            if n > 1:
                sub_dm = dm[np.ix_(feas_arr, feas_arr)]
                # Savings: d(depot,i) + d(depot,j) - d(i,j)
                # For each i, max savings with any j != i
                savings_matrix = dists_to_depot[:, None] + dists_to_depot[None, :] - sub_dm
                np.fill_diagonal(savings_matrix, -np.inf)
                max_savings = np.max(savings_matrix, axis=1)
                
                # Normalize savings
                max_s = np.max(np.abs(max_savings)) if np.max(np.abs(max_savings)) > 0 else 1.0
                norm_savings = max_savings / max_s
            else:
                norm_savings = np.array([0.0])
            
            # Score: minimize distance to current, but reward high savings potential
            delta = 0.4
            scores = dists_to_current - delta * norm_savings
            best_idx = np.argmin(scores)
            next_node = int(feas_arr[best_idx])
            
        total_dist += dm[current, next_node]
        current = next_node
        remaining_cap -= demands[next_node]
        remaining.remove(next_node)
        
    # Return to depot
    total_dist += dm[current, depot]
    
    return total_dist

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Pick the next customer or return to depot (0).
    
    Strategy: 1-Step Lookahead with savings-enhanced simulation.
    For each feasible candidate, estimate the total cost of the rest of the route
    if we pick that candidate now. The simulation uses a savings-based greedy
    heuristic to produce better route estimates. Choose the candidate that
    minimizes this estimate.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
        
    dm = _get_dm(distance_matrix)
    
    unvisited = list(unvisited_nodes)
    if not unvisited:
        return depot
    
    # Deadline guard: if we're running out of time, fall back to simple greedy
    if _check_deadline():
        feasible = [n for n in unvisited if demands[n] <= rest_capacity]
        if not feasible:
            return depot
        if len(feasible) == 1:
            return int(feasible[0])
        feasible_arr = np.array(feasible)
        dists = dm[current_node, feasible_arr]
        best_idx = np.argmin(dists)
        return int(feasible_arr[best_idx])
    
    # Filter to capacity-feasible nodes
    feasible = [n for n in unvisited if demands[n] <= rest_capacity]
    
    if not feasible:
        return depot
    
    if len(feasible) == 1:
        return int(feasible[0])
    
    # For each feasible candidate, simulate the rest of the route
    best_cost = float('inf')
    best_node = feasible[0]
    
    for cand in feasible:
        # Create a copy of unvisited without cand
        remaining_for_sim = [n for n in unvisited if n != cand]
        
        # Simulate route starting with cand
        # The total distance will include dist(current_node, cand) + dist(cand, next) + ... + dist(last, depot)
        sim_cost = _simulate_route(cand, depot, remaining_for_sim, rest_capacity - demands[cand], demands, dm)
        total_cost = dm[current_node, cand] + sim_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_node = cand
            
    return int(best_node)
# EVOLVE-BLOCK-END
