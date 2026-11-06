"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for deadline tracking
_start_time = None
_deadline = 145.0  # 5s buffer before 150s limit

def _two_opt_improve(route, dist):
    """
    Perform 2-opt improvement on a given route (list of nodes).
    The route is a sequence of nodes. We assume the start is fixed (index 0)
    and we can reverse subpaths to reduce total length.
    Returns the improved route.
    """
    n = len(route)
    if n <= 3:
        return route
    
    # Calculate current total length
    # Route is [start, n1, n2, ..., nk]
    # Length = d(start, n1) + d(n1, n2) + ... + d(nk-1, nk)
    # Note: In our context, 'route' passed to this function usually includes the current node at the end.
    # We are improving the path from the start (depot or previous depot) to the current node.
    
    best_route = route[:]
    best_len = 0.0
    # Calculate initial length
    for i in range(n - 1):
        best_len += dist[best_route[i], best_route[i+1]]
        
    improved = True
    # Limit iterations to avoid infinite loops or excessive time on very long routes
    # 50 nodes max, so O(N^2) is 2500 ops. Even with iterations, it's fast.
    max_iterations = 100
    iter_count = 0
    
    while improved and iter_count < max_iterations:
        improved = False
        iter_count += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                # Reverse segment from i to j
                # New path: ... -> route[i-1] -> route[j] -> ... -> route[i] -> ...
                # Old segment: route[i-1] -> route[i] ... route[j] -> route[j+1]
                # Delta = (d(route[i-1], route[j]) + d(route[i], route[j+1])) - (d(route[i-1], route[i]) + d(route[j], route[j+1]))
                
                old_cost = dist[best_route[i-1], best_route[i]] + dist[best_route[j], best_route[j+1]]
                new_cost = dist[best_route[i-1], best_route[j]] + dist[best_route[i], best_route[j+1]]
                
                if new_cost < old_cost - 1e-10:
                    # Perform reversal
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    best_len += new_cost - old_cost
                    improved = True
    
    return best_route

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking on first call
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded deadline
    elapsed = time.time() - _start_time
    if elapsed > _deadline:
        return depot
    
    if len(unvisited_nodes) == 0:
        return depot
    
    # Check feasible nodes (demand <= rest_capacity)
    unvisited_arr = np.array(unvisited_nodes)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot
    
    # If only one feasible node, take it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])
        
    # Strategy:
    # 1. Estimate the "quality" of adding each candidate to the current route.
    # 2. To do this robustly, we simulate adding the candidate to the route ending at current_node,
    #    apply 2-opt to the resulting segment, and check the total cost.
    # 3. However, we don't have the full route history in the state passed to this function.
    #    We only have current_node, depot, unvisited, etc.
    #    
    #    Wait, the function signature is:
    #    select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)
    #    
    #    We do NOT have the list of previously visited nodes in the current route.
    #    This makes it impossible to run 2-opt on the *actual* partial route because we don't know it.
    #    
    #    Therefore, the "2-opt on partial route" strategy is NOT feasible with this stateless interface
    #    unless we maintain global state of the current route.
    #    
    #    Let's maintain global state for the current route.
    #    
    #    Re-evaluating the directive: "Make a STRUCTURAL change".
    #    The current program is stateless (except for time).
    #    If I introduce state, I must be careful to reset it or handle multiple instances correctly.
    #    The problem says "eight generated 50-customer instances".
    #    Usually, these evaluators run the function sequentially for one instance, then reset.
    #    Or do they? "Score: mean total route length on eight generated 50-customer instances".
    #    If the evaluator calls select_next_node for instance 1, then instance 2, my global route state
    #    from instance 1 will corrupt instance 2.
    #    
    #    I need a way to detect when a new instance starts.
    #    Common heuristic: If current_node is depot (0) and unvisited_nodes contains all nodes (or a large set),
    #    it's likely a new instance start.
    #    Or, if rest_capacity is at max (50), it's a new instance.
    #    
    #    Let's assume rest_capacity == 50 indicates a new route start.
    #    And we can store the current route in a global variable.
    
    global _current_route
    if not hasattr(select_next_node, '_current_route'):
        select_next_node._current_route = [depot]
        
    # Detect new instance/route
    # If we are at depot and capacity is full, it's a new route
    if current_node == depot and rest_capacity == 50:
        select_next_node._current_route = [depot]
    else:
        # Append current node if it's not already the last element (safety)
        if select_next_node._current_route[-1] != current_node:
            select_next_node._current_route.append(current_node)
            
    # Now we have the partial route: select_next_node._current_route
    # We want to choose the next node from feasible_nodes.
    
    # For each candidate, we can:
    # 1. Append it to the route.
    # 2. Run 2-opt on the entire route (or just the tail) to see if it improves.
    # 3. Calculate the total length of the improved route.
    # 4. Choose the candidate that minimizes the total length.
    
    # Optimization: 2-opt on the whole route for every candidate is O(N^3) per step?
    # Route length is up to 50. 2-opt is O(N^2). Candidates are up to 50.
    # 50 * 2500 = 125,000 ops per step. 50 steps = 6.25M ops.
    # This is very fast in Python if vectorized or simple loops.
    
    best_candidate = None
    best_score = float('inf')
    
    # We only need to consider the tail of the route for 2-opt?
    # Actually, 2-opt can swap any part. But since we only added one node,
    # only the tail is affected. However, 2-opt might find improvements in the middle
    # that were missed previously? No, we assume previous steps were optimized.
    # But to be safe and simple, let's run 2-opt on the whole route.
    # Since N is small, it's fine.
    
    # Pre-calculate distances for speed
    # We will simulate each candidate
    
    for c in feasible_nodes:
        # Create a copy of the route with c appended
        simulated_route = select_next_node._current_route[:] + [c]
        
        # Run 2-opt improvement
        improved_route = _two_opt_improve(simulated_route, distance_matrix)
        
        # Calculate total length
        total_len = 0.0
        for i in range(len(improved_route) - 1):
            total_len += distance_matrix[improved_route[i], improved_route[i+1]]
            
        # We might also want to consider the cost to return to depot?
        # The score is "mean total route length".
        # Total route length usually includes return to depot.
        # Let's add d(last, depot) to the score to encourage closing routes appropriately.
        total_len += distance_matrix[improved_route[-1], depot]
        
        if total_len < best_score:
            best_score = total_len
            best_candidate = c
            
    # Update the global route with the best candidate
    # Note: We should update the route to the IMPROVED version?
    # If we update to the improved version, the next step will start from the new last node.
    # But the interface expects us to return the node we are moving to.
    # The evaluator will then call select_next_node with that node as current_node.
    # If we internally reordered the route, the "current_node" passed by the evaluator
    # will be the node we returned.
    # If our internal route ends with a different node (due to 2-opt reversal),
    # there will be a mismatch.
    
    # Example:
    # Route: [0, 1, 2]
    # Candidate: 3
    # Simulated: [0, 1, 2, 3]
    # 2-opt improves to: [0, 1, 3, 2] (reversed 2,3)
    # Total len calculated based on [0, 1, 3, 2].
    # We return 3 (the candidate we picked).
    # Evaluator sets current_node = 3.
    # Next call: select_next_node(3, ...).
    # Our global route is still [0, 1, 2] (or we update it to [0, 1, 2, 3]?).
    # If we update to [0, 1, 2, 3], the next step sees current=3.
    # But the "true" optimal path might have been [0, 1, 3, 2].
    # If we return 3, we are committed to visiting 3 next.
    # The 2-opt improvement [0, 1, 3, 2] implies visiting 3 then 2.
    # So if we return 3, the next step will be 2.
    # This works IF we update our internal route to reflect the choice.
    
    # However, 2-opt on the simulated route might reverse a segment that doesn't include the new node?
    # Unlikely if the route was previously optimized.
    # But if it does, we have a problem.
    # Simplest approach: Only allow 2-opt improvements that involve the newly added node?
    # Or, just use the simulated route's length as a proxy, but update the actual route
    # by simply appending the candidate.
    # This means we use 2-opt for SCORING, but not for STATE UPDATE.
    # This is a "Lookahead 2-opt" heuristic.
    
    # Let's do that: Score based on 2-opt improved length, but state update is simple append.
    
    # Wait, if the 2-opt improvement is significant, ignoring it in the state means
    # we lose that benefit in the actual execution.
    # But we are constrained by the sequential interface.
    # We can only say "go to node X".
    # If the optimal next step is to go to X, but the optimal *path* so far should have been different,
    # we can't fix the past.
    # So, scoring based on "what if we had optimized the past" is a valid lower bound on the cost.
    # It helps distinguish candidates that fit well into the *existing* structure.
    
    # Let's proceed with:
    # 1. Simulate append.
    # 2. 2-opt improve simulation.
    # 3. Score = length of improved simulation + dist(last, depot).
    # 4. Pick best candidate.
    # 5. Update global route by appending the chosen candidate.
    
    # One issue: The global route might drift from the "true" optimal path.
    # But since we are constructing it sequentially, this is the best we can do.
    
    # Optimization: If the route is long, 2-opt on the whole thing is expensive.
    # We can limit 2-opt to the last K nodes (e.g., 10).
    # This is a "Tail 2-opt".
    
    # Let's implement a tail 2-opt for speed and locality.
    
    best_candidate = None
    best_score = float('inf')
    
    for c in feasible_nodes:
        simulated_route = select_next_node._current_route[:] + [c]
        
        # Tail 2-opt: only consider reversals in the last 10 nodes
        tail_start = max(1, len(simulated_route) - 10)
        
        # Apply 2-opt on the tail
        n = len(simulated_route)
        improved = True
        iter_count = 0
        while improved and iter_count < 20: # Fewer iterations for tail
            improved = False
            iter_count += 1
            for i in range(tail_start, n - 2):
                for j in range(i + 1, n - 1):
                    old_cost = distance_matrix[simulated_route[i-1], simulated_route[i]] + distance_matrix[simulated_route[j], simulated_route[j+1]]
                    new_cost = distance_matrix[simulated_route[i-1], simulated_route[j]] + distance_matrix[simulated_route[i], simulated_route[j+1]]
                    
                    if new_cost < old_cost - 1e-10:
                        simulated_route[i:j+1] = reversed(simulated_route[i:j+1])
                        improved = True
        
        # Calculate score: length of simulated_route + dist(last, depot)
        total_len = 0.0
        for i in range(len(simulated_route) - 1):
            total_len += distance_matrix[simulated_route[i], simulated_route[i+1]]
        total_len += distance_matrix[simulated_route[-1], depot]
        
        if total_len < best_score:
            best_score = total_len
            best_candidate = c
            
    # Update global route
    select_next_node._current_route.append(best_candidate)
    
    return int(best_candidate)

# EVOLVE-BLOCK-END
