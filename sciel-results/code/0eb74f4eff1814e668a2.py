"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track deadline
_start_time = None

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    global _start_time
    
    # Initialize deadline tracking
    if _start_time is None:
        _start_time = time.time()
    
    # Safety margin: stop if we've exceeded 145s (5s buffer before 150s limit)
    elapsed = time.time() - _start_time
    if elapsed > 145:
        return depot

    if len(unvisited_nodes) == 0:
        return depot

    unvisited_arr = np.array(unvisited_nodes)
    
    # Check feasible nodes (demand <= rest_capacity)
    feasible_mask = demands[unvisited_arr] <= rest_capacity
    feasible_nodes = unvisited_arr[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return depot

    # If only one feasible node, return it
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Distances
    d_depot_current = distance_matrix[depot, current_node]
    d_depot_cand = distance_matrix[depot, feasible_nodes]
    d_current_cand = distance_matrix[current_node, feasible_nodes]
    
    # Clarke-Wright savings for linking current and candidate
    # s = d(depot, current) + d(depot, cand) - d(current, cand)
    savings = d_depot_current + d_depot_cand - d_current_cand
    
    # We want to maximize savings, but also consider the "cost" of the detour.
    # A pure savings heuristic can be greedy and lead to long routes if it
    # keeps adding nodes that are "cheap" to add relative to separate routes
    # but actually extend the route significantly.
    
    # A better heuristic for sequential construction is often:
    # Minimize: d(current, cand) + lambda * d(cand, depot)
    # Or use a combined score.
    
    # Let's try a weighted combination of savings and distance.
    # High savings is good. Low distance is good.
    
    # Normalize savings and distance to make them comparable?
    # Or just use a simple linear combination.
    
    # Let's test: Score = savings - alpha * d_current_cand
    # If alpha is large, it behaves like nearest neighbor.
    # If alpha is 0, it behaves like pure savings.
    
    # Another approach:
    # The "regret" of not picking a node now.
    
    # Let's try a different perspective.
    # If we pick candidate c, the immediate cost is d(current, c).
    # The future cost is at least d(c, depot) if we close, but likely more.
    # The "opportunity cost" of picking c is that we can't pick other nodes.
    
    # A robust heuristic for CVRP is often:
    # Minimize d(current, c) / d(depot, c) ? No.
    
    # Let's stick to the savings family but refine it.
    # The standard savings s(i,j) = d(i,0) + d(j,0) - d(i,j).
    # When extending a route ending at i, we pick j that maximizes s(i,j).
    # This is what the current code does.
    
    # Why might it fail?
    # 1. It doesn't account for capacity well (only feasibility).
    # 2. It might create very long routes that are hard to close.
    
    # Let's add a penalty for nodes that are far from the depot if the route is already long?
    # We don't know the route length directly, but we can approximate.
    # Actually, d_depot_current is a proxy for how far we are.
    
    # Let's try: Score = savings - beta * d_current_cand
    # We need to tune beta.
    # If d_current_cand is large, the savings might be small anyway?
    # s = d0 + dc - dcc.
    # If dcc is large, s is small. So maximizing s already penalizes large dcc.
    # Wait, s = d0 + dc - dcc.
    # If dcc is large, s is small.
    # If dcc is small, s is large.
    # So maximizing s is equivalent to minimizing dcc - dc?
    # No.
    
    # Let's look at the gradient.
    # We want to pick c.
    
    # Let's try a simple improvement:
    # If the savings is negative, it's bad to link them.
    # But we must pick someone if we want to continue.
    # If all savings are negative, we should probably close the route?
    # But the interface requires us to pick a node or depot.
    # If we pick depot, the route closes.
    # So, if max(savings) < 0, maybe return depot?
    # Let's check if that's valid.
    # "return 0 (depot) to close a route".
    # Yes.
    
    # So, if the best savings is negative, it means linking current and any candidate
    # is more expensive than having them in separate routes.
    # In that case, it might be better to close the current route and start a new one.
    
    best_savings = np.max(savings)
    
    # Threshold for closing the route
    # If best_savings is very low, close.
    # What is a good threshold?
    # If d(current, depot) is small, and d(cand, depot) is large, savings might be low.
    
    # Let's try: if best_savings < 0: return depot
    # But this might be too aggressive.
    
    # Let's try a softer approach.
    # Score = savings
    # If max score < threshold, return depot.
    
    # Let's experiment with the threshold.
    # For now, let's just maximize savings, but if the best savings is negative,
    # we might consider closing.
    
    # Actually, if we are at current, and we close, we pay d(current, depot).
    # If we go to cand, we pay d(current, cand) + ...
    # The savings formula compares:
    # Route 1: ... -> current -> depot
    # Route 2: ... -> cand -> depot (separate)
    # Combined: ... -> current -> cand -> depot
    # Cost(Combined) = d(current, cand) + d(cand, depot)
    # Cost(Separate) = d(current, depot) + d(cand, depot)
    # Savings = Cost(Separate) - Cost(Combined) = d(current, depot) - d(current, cand)
    # Wait, my previous formula was:
    # s = d(depot, current) + d(depot, cand) - d(current, cand)
    # This is the standard savings for merging two routes (depot->current) and (depot->cand).
    # But here, current is already in a route.
    
    # Let's re-derive.
    # Current state: Route ends at current.
    # Option A: Close route. Cost added: d(current, depot).
    # Option B: Go to cand. Cost added: d(current, cand). (Plus future costs).
    # If we assume the future costs from cand are similar regardless,
    # we should pick cand if d(current, cand) < d(current, depot) + (future cost penalty).
    
    # The standard savings heuristic for insertion is:
    # Inserting k between i and j saves: d(i,j) - d(i,k) - d(k,j).
    # Here, we are appending to the end.
    # It's like inserting k between current and depot.
    # Savings = d(current, depot) - d(current, k) - d(k, depot).
    # We want to maximize this savings.
    # If savings > 0, it's beneficial to insert k.
    
    # Let's use this formula!
    # s_insert = d(current, depot) - d(current, cand) - d(cand, depot)
    
    d_current_depot = distance_matrix[current_node, depot]
    s_insert = d_current_depot - d_current_cand - d_depot_cand
    
    # We want to maximize s_insert.
    # If max(s_insert) < 0, it means inserting any candidate increases the cost
    # compared to closing the route now.
    # So we should return depot.
    
    best_idx = np.argmax(s_insert)
    best_s = s_insert[best_idx]
    
    # If the best insertion savings is negative, close the route.
    # But maybe we should allow a small negative savings if we have to visit more nodes?
    # No, if it's negative, it's strictly worse to add that node than to close.
    # Unless capacity forces us? No, capacity is on the route, not the depot.
    # If we close, we start a new route. The new route will have its own savings.
    
    # So, if best_s < 0, return depot.
    
    # However, there is a nuance.
    # What if we have many nodes left and few routes?
    # Closing a route early might lead to many short routes, which is bad.
    # But the savings heuristic naturally handles this?
    # If we close, the next route starts fresh.
    
    # Let's try this heuristic.
    # Threshold: if best_s < 0: return depot
    
    # Let's add a small epsilon or tune the threshold.
    # Maybe if best_s is very close to 0, it's a tie.
    
    # Let's try: if best_s < 0: return depot
    # Otherwise, return the node with max s_insert.
    
    if best_s < 0:
        return depot
    
    return int(feasible_nodes[best_idx])

# EVOLVE-BLOCK-END
