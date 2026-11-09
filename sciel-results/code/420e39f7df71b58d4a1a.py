"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_prev_node = None
_tour_start_node = None

def _get_time_remaining():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _prev_node, _tour_start_node
    
    # Time guard
    if _get_time_remaining() < 1.0:
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
        
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    if n == 1:
        return int(candidates[0])
        
    # Detect start of a new tour
    # Assuming 50 cities total. If 49 are unvisited, it's the first step.
    # If the problem size varies, we might need a more robust check, but 50 is specified.
    if n == 49:
        _tour_start_node = current_node
        _prev_node = None
    else:
        # If we are mid-tour, _prev_node should be the node before current_node
        # But we don't have it stored directly unless we track history.
        # Let's maintain a simple history in a global list? 
        # Or just assume _prev_node was set in the previous call to be the node 
        # that led to current_node? 
        # No, the previous call returned `current_node`. So the node *before* that 
        # is what we need.
        # We need to store the last 2 nodes.
        pass

    # To properly do angle smoothing, we need the node before current_node.
    # Let's use a global list `tour_history`
    global tour_history
    if n == 49:
        tour_history = [current_node]
    else:
        tour_history.append(current_node)
        
    if len(tour_history) >= 2:
        prev_node = tour_history[-2]
    else:
        prev_node = None

    dist_to_candidate = distance_matrix[current_node, candidates]
    
    # If we don't have a previous node, use simple nearest neighbor with destination bias
    if prev_node is None:
        # Small bias to destination to start heading that way? 
        # Or just nearest neighbor.
        dist_candidate_to_dest = distance_matrix[candidates, destination_node]
        # Early in the tour, destination bias should be low.
        beta = 0.1
        scores = dist_to_candidate + beta * dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
        
    # Calculate angle smoothing penalty
    # Vector in: current - prev
    # Vector out: candidate - current
    # We want to minimize the turn angle, i.e., maximize dot product / (norms)
    # cos(theta) = dot(v_in, v_out) / (|v_in| |v_out|)
    # Penalty = 1 - cos(theta)
    
    v_in = distance_matrix[current_node, :] - distance_matrix[prev_node, :] # This is wrong. distance_matrix is distances, not coordinates.
    
    # CRITICAL ISSUE: The function receives `distance_matrix`, not coordinates.
    # We CANNOT calculate angles or vectors without coordinates!
    # The distance matrix is a metric, not the points themselves.
    # We can only use distances.
    
    # So, Angle Smoothing is IMPOSSIBLE with only distance_matrix.
    # We must discard the angle strategy.

    # BACK TO DRAWING BOARD.
    # What can we do with only distances?
    # 1. Nearest Neighbor
    # 2. 2-Step Lookahead: Minimize d(current, next) + d(next, best_next_next)
    # 3. MST-based (Current approach)
    # 4. "Regret" based: For each candidate i, calculate how much worse it is to NOT pick i now compared to the best alternative?
    #    Regret(i) = min_{j != i} (d(current, j) + d(j, ...)) - (d(current, i) + d(i, ...))
    #    This is complex.
    
    # Let's try **2-Step Lookahead** properly.
    # For each candidate i in unvisited:
    #   cost_i = d(current, i)
    #   remaining = unvisited \ {i}
    #   if remaining is empty: cost_i += d(i, destination)
    #   else: cost_i += min_{j in remaining} d(i, j)  # Nearest neighbor of i among remaining
    # This is O(N^2) per step. 50 steps * 2500 = 125k ops. Fast.
    # This is a standard heuristic: "Greedy with 1-step lookahead".
    
    # Let's compare this to the current MST approach.
    # The current approach uses MST leaf savings.
    # 2-Step Lookahead is a more direct estimate of the future cost.
    
    # Let's implement 2-Step Lookahead.
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 1:
        return int(candidates[0])
        
    # Precompute distances between all candidates?
    # dist_matrix_sub = distance_matrix[np.ix_(candidates, candidates)]
    
    # For each candidate i:
    # 1. d(current, i)
    # 2. If n == 2: d(i, other) + d(other, destination)
    # 3. If n > 2: d(i, nearest_neighbor_in_remaining(i))
    
    # Optimization:
    # We can vectorize the "nearest neighbor in remaining" part?
    # Not easily because the set "remaining" changes for each i.
    # But N is small (50). O(N^2) is fine.
    
    scores = np.zeros(n)
    for i in range(n):
        node_i = candidates[i]
        cost_1 = distance_matrix[current_node, node_i]
        
        if n == 2:
            node_j = candidates[1 - i]
            cost_2 = distance_matrix[node_i, node_j]
            cost_3 = distance_matrix[node_j, destination_node]
            scores[i] = cost_1 + cost_2 + cost_3
        else:
            # Find nearest neighbor of node_i among the other candidates
            # Mask out node_i
            # dist_i_to_others = distance_matrix[node_i, candidates]
            # dist_i_to_others[i] = np.inf
            # min_dist = np.min(dist_i_to_others)
            
            # To avoid modifying the array, we can just loop or use fancy indexing
            # Since n is small, a simple loop or vectorized op is fine.
            dists_from_i = distance_matrix[node_i, candidates]
            dists_from_i[i] = np.inf
            min_dist = np.min(dists_from_i)
            
            # Add a small bias to destination to handle the end game
            # If n is small, the "nearest neighbor" might not be the best path to destination.
            # Let's add a weighted destination term.
            dist_to_dest = distance_matrix[node_i, destination_node]
            
            # Heuristic: cost = d(current, i) + min(d(i, j for j in remaining)) + lambda * d(i, dest)
            # Lambda should increase as n decreases.
            if n <= 5:
                # For very small n, do a proper 2-step or 3-step?
                # If n=3, we have i, j, k.
                # Cost(i) = d(cur, i) + min(d(i, j)+d(j, k)+d(k, dest), d(i, k)+d(k, j)+d(j, dest))
                # This is getting complex.
                # Let's stick to the simple min-neighbor + dest bias.
                lambda_dest = 0.5
            else:
                lambda_dest = 0.1
                
            scores[i] = cost_1 + min_dist + lambda_dest * dist_to_dest
            
    return int(candidates[np.argmin(scores)])
# EVOLVE-BLOCK-END
