"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time once at module load
_START_TIME = time.time()
_DEADLINE = 150.0  # seconds
_SAFETY_MARGIN = 2.0  # return safe result this far before deadline

def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEALINE - _SAFETY_MARGIN) if False else (time.time() - _START_TIME) > (_DEALINE - _SAFETY_MARGIN)

def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix):
    """
    Heuristic: Minimum Incremental Cost Insertion.
    
    Strategy:
    Instead of greedily picking the nearest feasible node to append to the current route,
    we simulate inserting each feasible unvisited node into the current route at its
    best possible position (minimizing the increase in total route length).
    
    We then select the node that has the MINIMUM "penalty" for being left unvisited 
    or, more practically, we select the node that, if we were to build the route 
    via insertion, would be the most "difficult" to place later? 
    
    Actually, a standard constructive heuristic is:
    1. Start with an empty route (or current partial route).
    2. For the next step, we don't just append. We consider the set of feasible unvisited nodes.
    3. We calculate the cost of inserting node `j` into the current path 
       (which is currently just the sequence of visited nodes ending at `current_node`).
       Wait, the interface only gives `current_node`, not the full history of visited nodes.
       
    CRITICAL CONSTRAINT CHECK:
    The interface `select_next_node(current_node, depot, unvisited_nodes, rest_capacity, demands, distance_matrix)`
    does NOT provide the list of previously visited nodes in the current route.
    It only provides `current_node`.
    
    This means we CANNOT perform true "insertion into the middle of the route" because we don't know the route structure.
    We only know the last node.
    
    Therefore, the "Insertion" strategy must be adapted:
    We can only "insert" at the end (append) or we can assume the route is just a chain ending at current_node.
    If we can only append, we are back to nearest neighbor variants.
    
    HOWEVER, we can use a **Lookahead** strategy that simulates the next 2-3 steps to approximate the benefit of the choice, 
    or we can use a **Savings-based** approach that doesn't require full history, only the current node and the depot.
    
    Let's re-evaluate the "Different Strategy" requirement.
    Current: Greedy scoring (proximity, savings, return).
    Inspiration 1 & 2: Clarke-Wright Savings.
    
    A distinct structural change given the limited state (no full route history) is to use a **Beam Search / Limited Lookahead** 
    or a **Randomized Restart** strategy, or a **Sweep-Angle** based heuristic (which uses geometric properties rather than pure distance sums).
    
    Let's try **Sweep-Angle Heuristic**:
    1. Calculate the angle of each feasible node relative to the depot.
    2. Calculate the angle of the current node relative to the depot.
    3. Prefer nodes that are "close" in angle to the current node, encouraging contiguous sectors.
    4. This is structurally different from distance-minimization because it prioritizes angular continuity, which often correlates with lower total path length in Euclidean spaces by reducing crossings.
    
    Wait, the previous program (p66) used "Sweep-Angle Priority" and scored 1.0328.
    The current program (p71) uses "Multi-Objective Adaptive Scoring" and scored 1.0331.
    They are very close.
    
    Let's try a **2-Step Lookahead** strategy.
    1. For each feasible candidate `c1`:
       a. Simulate moving to `c1`.
       b. From `c1`, find the best next candidate `c2` (nearest feasible remaining node).
       c. Calculate the cost: `dist(cur, c1) + dist(c1, c2)`.
       d. Also consider the cost of closing: `dist(cur, c1) + dist(c1, depot)` if we run out of capacity or nodes.
    2. Select `c1` that minimizes this lookahead cost.
    
    This is a different *search method* (depth-2 lookahead vs depth-1 greedy).
    """
    if len(unvisited_nodes) == 0:
        return depot

    # If time is running out, fall back to simple nearest feasible
    if _time_exceeded():
        feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
        feasible_nodes = unvisited_nodes[feasible_mask]
        if len(feasible_nodes) == 0:
            return depot
        return int(feasible_nodes[np.argmin(distance_matrix[current_node, feasible_nodes])])

    # Filter by capacity
    feasible_mask = np.array([demands[i] <= rest_capacity for i in unvisited_nodes])
    feasible_nodes = unvisited_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return depot

    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # 2-Step Lookahead Strategy
    # For each candidate c1 in feasible_nodes:
    # 1. Cost to go from current to c1: d_cur_c1
    # 2. Simulate being at c1.
    # 3. Remaining capacity: rest_capacity - demands[c1]
    # 4. Remaining unvisited: unvisited_nodes excluding c1
    # 5. Find the best next step c2 from c1 among remaining feasible nodes.
    #    - If no nodes fit in remaining capacity, the "next step" is returning to depot.
    #    - Cost of this path segment: d_c1_c2 (or d_c1_depot)
    # 6. Total lookahead cost: d_cur_c1 + min_next_cost
    # 7. Select c1 with minimum total lookahead cost.
    
    n_feasible = len(feasible_nodes)
    d_cur_cands = distance_matrix[current_node, feasible_nodes]
    
    best_score = float('inf')
    best_node = feasible_nodes[0]
    
    # Pre-compute distances from depot for potential closing costs
    d_depot_cands = distance_matrix[depot, feasible_nodes]
    
    for i, c1 in enumerate(feasible_nodes):
        cost_step1 = d_cur_cands[i]
        
        # Simulate state after visiting c1
        new_rest_cap = rest_capacity - demands[c1]
        
        # Determine next best move from c1
        # If new_rest_cap is 0, we MUST return to depot (or if no other nodes fit)
        # We need to check which other nodes fit in new_rest_cap
        
        # Optimization: If n_feasible is large, this O(N^2) loop might be slow.
        # 50 nodes max. 50*50 = 2500 operations. Very fast.
        
        # Find feasible nodes from c1's perspective
        # We can reuse the full distance matrix
        # Nodes that are feasible from c1 with new_rest_cap
        # Note: c1 is removed from unvisited.
        
        # To avoid creating new arrays in the loop, we can compute min distance to any feasible node or depot.
        
        # Option A: Return to depot
        cost_return = distance_matrix[c1, depot]
        min_next_cost = cost_return
        
        # Option B: Go to another node c2
        if new_rest_cap > 0:
            # Find candidates c2 from the original feasible_nodes, excluding c1
            # And check if demands[c2] <= new_rest_cap
            
            # Vectorized check for c2
            # c2_indices are indices in feasible_nodes
            # We need to exclude index i (c1)
            
            # Create mask for c2 feasibility
            # demands of feasible_nodes
            feas_demands = demands[feasible_nodes]
            
            # Mask: demand <= new_rest_cap AND index != i
            mask_c2 = (feas_demands <= new_rest_cap)
            mask_c2[i] = False
            
            if np.any(mask_c2):
                c2_indices = np.where(mask_c2)[0]
                c2_nodes = feasible_nodes[c2_indices]
                
                # Distances from c1 to c2
                d_c1_c2 = distance_matrix[c1, c2_nodes]
                
                # The "best" next step is the nearest one? 
                # Or should we apply a similar heuristic?
                # For lookahead, nearest is a good proxy for immediate cost.
                # However, to be more robust, we can just take the min distance.
                # This makes it a "Nearest Neighbor Lookahead".
                
                min_d_c1_c2 = np.min(d_c1_c2)
                
                # Compare with returning to depot
                if min_d_c1_c2 < cost_return:
                    min_next_cost = min_d_c1_c2
                    # Note: This lookahead cost doesn't account for the fact that going to c2 might lead to a dead end.
                    # But depth-2 is a standard heuristic improvement over depth-1.
                    
        total_lookahead_cost = cost_step1 + min_next_cost
        
        # Tie-breaking: If costs are equal, prefer the one with lower distance to depot (easier to close later?)
        # Or just keep the first one found.
        
        if total_lookahead_cost < best_score:
            best_score = total_lookahead_cost
            best_node = c1

    return int(best_node)

# EVOLVE-BLOCK-END
