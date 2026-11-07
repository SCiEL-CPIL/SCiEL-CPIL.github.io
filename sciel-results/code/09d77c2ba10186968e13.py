"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 325.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    n = edge_distance.shape[0]
    tour = np.asarray(local_opt_tour)
    
    if len(tour) < 3:
        return edge_distance.copy()
        
    # Strategy: Deterministic Best-Edge Removal (Exact Perturbation)
    #
    # Instead of a heuristic "field" that biases the search, we identify the
    # single best 2-opt move available in the current tour (based on true distances)
    # and force the 2-opt on the modified matrix to execute it.
    #
    # Mechanism:
    # 1. Find the pair of edges in the tour whose removal and reconnection yields
    #    the largest decrease in tour length (the "best" 2-opt move).
    # 2. Create a modified distance matrix where:
    #    - The two new edges of this best move have their true distance.
    #    - ALL other edges not in the current tour have a huge penalty (1e6 * max_dist).
    #    - The edges in the current tour (excluding the two being removed) keep their true distance.
    #
    # Why this works:
    # The 2-opt algorithm on the modified matrix will see that the only way to
    # improve the tour (which is currently a local optimum on true distances) is
    # to remove one of the two edges involved in the best move. Because all other
    # non-tour edges are extremely expensive, the 2-opt will be forced to take
    # the specific move we identified. This ensures a deterministic, non-random
    # escape from the local optimum, moving the tour to a specific, better
    # (in terms of true distance) configuration.
    
    # Step 1: Identify the best 2-opt move in the current tour
    # We iterate over all pairs of edges in the tour to find the one with the max improvement.
    # Tour edges are (tour[i], tour[i+1]) for i in 0..n-1 (with wrap-around).
    
    best_improvement = -np.inf
    best_i, best_j = -1, -1
    
    # Precompute tour edge lengths for speed
    tour_edges = np.roll(tour, -1, axis=0)
    # edge i is (tour[i], tour_edges[i])
    
    # To find the best 2-opt move, we check pairs of non-adjacent edges.
    # Removing edges i and j (where i and j are not adjacent in the tour) and reconnecting
    # them in the other way changes the tour length by:
    # delta = dist(tour[i], tour[j]) + dist(tour_edges[i], tour_edges[j]) - dist(tour[i], tour_edges[i]) - dist(tour[i-1], tour[i])
    # Wait, the standard 2-opt move removes edges (a,b) and (c,d) and adds (a,c) and (b,d) or (a,d) and (b,c).
    # Let's stick to the standard definition:
    # Edges are e_i = (tour[i], tour[i+1])
    # Remove e_i and e_j (i < j, not adjacent).
    # New edges are (tour[i], tour[j]) and (tour[i+1], tour[j+1]).
    # (Assuming the tour segment between i and j is reversed).
    # Improvement = dist(tour[i], tour[i+1]) + dist(tour[j], tour[j+1]) - dist(tour[i], tour[j]) - dist(tour[i+1], tour[j+1])
    
    # We need to be careful with indices.
    # Let's iterate i from 0 to n-1, and j from i+2 to n-1 (excluding adjacent and same).
    # Also handle the wrap-around case where i=0 and j=n-1 are adjacent.
    
    # For n=100, O(n^2) is 10,000 operations, which is very fast.
    
    tour_len = len(tour)
    for i in range(tour_len):
        # Edge i is (tour[i], tour[(i+1)%tour_len])
        a_i = tour[i]
        b_i = tour[(i+1) % tour_len]
        d_ei = edge_distance[a_i, b_i]
        
        for j in range(i+2, tour_len):
            # Skip if j is the last edge and i is the first edge (they are adjacent in the cycle)
            if i == 0 and j == tour_len - 1:
                continue
                
            # Edge j is (tour[j], tour[(j+1)%tour_len])
            a_j = tour[j]
            b_j = tour[(j+1) % tour_len]
            d_ej = edge_distance[a_j, b_j]
            
            # New edges would be (a_i, a_j) and (b_i, b_j)
            # OR (a_i, b_j) and (b_i, a_j)
            # The standard 2-opt reversal connects a_i to a_j and b_i to b_j.
            
            d_new1 = edge_distance[a_i, a_j]
            d_new2 = edge_distance[b_i, b_j]
            
            improvement = d_ei + d_ej - d_new1 - d_new2
            
            if improvement > best_improvement:
                best_improvement = improvement
                best_i = i
                best_j = j
    
    # If no improving move was found (shouldn't happen if local_opt_tour is truly local opt,
    # but maybe due to floating point or if it's not strictly local opt),
    # or if best_improvement is <= 0, we can just return the original matrix
    # or pick the least bad move. But the contract says local_opt_tour is improved by 2-opt,
    # so there should be no improving 2-opt move on true distances.
    # Wait, the evaluator runs 2-opt on true distances AFTER our function.
    # So local_opt_tour IS a 2-opt local optimum on true distances.
    # Therefore, best_improvement will likely be <= 0.
    
    # If best_improvement <= 0, we are at a local optimum.
    # The standard GLS idea is to penalize edges to FORCE a move.
    # If we force a move, it will be a WORSE move on true distances.
    # However, the evaluator runs 2-opt on true distances afterwards.
    # So if we force a specific "bad" move, the subsequent 2-opt might find a new local optimum
    # that is better than the current one (a "kick").
    
    # Which move to force?
    # The "best" worsening move is the one with the smallest negative improvement (i.e., closest to 0).
    # This minimizes the immediate damage, hoping the subsequent 2-opt can recover and find a new basin.
    
    if best_improvement > 0:
        # This case should rarely happen if local_opt_tour is truly local opt.
        # But if it does, we force the best improving move.
        pass
    else:
        # We want the move with the maximum improvement (which is negative or zero).
        # My loop above finds the MAX improvement.
        # If all are negative, best_improvement is the least negative one.
        # This is exactly what we want: the "least bad" move.
        pass
        
    # If no move was found (n < 4?), return original
    if best_i == -1:
        return edge_distance.copy()
        
    # Construct the modified matrix
    # We want 2-opt to perform the move (best_i, best_j).
    # The move removes edges (tour[best_i], tour[best_i+1]) and (tour[best_j], tour[best_j+1])
    # and adds edges (tour[best_i], tour[best_j]) and (tour[best_i+1], tour[best_j+1]).
    
    # Strategy:
    # 1. Start with a copy of edge_distance.
    # 2. Set all non-diagonal elements to a huge penalty (e.g., 1e6).
    # 3. Set the edges of the CURRENT tour to their true distances (so 2-opt can see the current structure).
    # 4. Set the TWO NEW edges of the best move to their true distances.
    # 5. Set the TWO OLD edges of the best move to a HUGE penalty (so 2-opt MUST remove them).
    #    Actually, if we set them to huge penalty, 2-opt will remove them.
    #    But we also need to ensure it adds the specific new edges.
    #    If all other non-tour edges are huge, and the two new edges are true distance,
    #    2-opt will be forced to use those two new edges to reconnect the tour.
    
    modified_dist = np.full_like(edge_distance, 1e6 * np.max(edge_distance))
    
    # Restore true distances for all edges in the current tour
    # This allows 2-opt to evaluate the current tour's cost correctly (mostly).
    # Actually, we want to force the specific move.
    # Let's try a simpler approach:
    # Only allow edges that are in the current tour OR are the two new edges of the best move.
    # All other edges are penalized.
    
    # Initialize with huge penalty
    mod = np.full_like(edge_distance, 1e6 * np.max(edge_distance))
    
    # Add back current tour edges
    for k in range(tour_len):
        u = tour[k]
        v = tour[(k+1) % tour_len]
        mod[u, v] = edge_distance[u, v]
        mod[v, u] = edge_distance[v, u]
        
    # Add back the two new edges of the best move
    a_i = tour[best_i]
    b_i = tour[(best_i+1) % tour_len]
    a_j = tour[best_j]
    b_j = tour[(best_j+1) % tour_len]
    
    new_edge_1 = (a_i, a_j)
    new_edge_2 = (b_i, b_j)
    
    mod[new_edge_1[0], new_edge_1[1]] = edge_distance[new_edge_1[0], new_edge_1[1]]
    mod[new_edge_1[1], new_edge_1[0]] = edge_distance[new_edge_1[0], new_edge_1[1]]
    
    mod[new_edge_2[0], new_edge_2[1]] = edge_distance[new_edge_2[0], new_edge_2[1]]
    mod[new_edge_2[1], new_edge_2[0]] = edge_distance[new_edge_2[0], new_edge_2[1]]
    
    # Crucially, we want to FORBID the two old edges so they MUST be removed.
    # If we leave them as true distance, 2-opt might just keep them.
    # But wait, if we forbid them, the tour becomes disconnected unless the new edges are used.
    # The new edges ARE available.
    # So 2-opt will see that the current tour (with old edges) is valid, but
    # the move that removes old edges and adds new edges is also valid.
    # Will it take it?
    # The cost of the current tour on `mod` is sum of true distances of tour edges.
    # The cost of the new tour (with new edges) is sum of true distances of new tour edges.
    # Since we picked the "best" (least worsening) move, the new tour is only slightly worse (or better)
    # than the current one on true distances.
    # However, 2-opt on `mod` doesn't know about true distances, it only sees `mod`.
    # In `mod`, the old edges have true distance, and new edges have true distance.
    # So 2-opt sees the improvement as:
    # delta_mod = d(old1) + d(old2) - d(new1) - d(new2)
    # This is the same as delta_true.
    # If delta_true is negative (worsening), 2-opt will NOT take the move!
    # 2-opt only takes moves that improve the objective function.
    # The objective function for 2-opt on `mod` is the sum of distances in `mod`.
    # So if the move worsens the true distance, it also worsens the `mod` distance (since we set them to true distances).
    # Therefore, 2-opt will NOT execute the move. It will stay at the local optimum.
    
    # To FORCE the move, we must make the old edges EXPENSIVE in `mod`.
    # Let's set the two old edges to a huge penalty in `mod`.
    mod[a_i, b_i] = 1e6 * np.max(edge_distance)
    mod[b_i, a_i] = 1e6 * np.max(edge_distance)
    mod[a_j, b_j] = 1e6 * np.max(edge_distance)
    mod[b_j, a_j] = 1e6 * np.max(edge_distance)
    
    # Now, the current tour has a huge cost in `mod` (because it contains the two penalized edges).
    # The new tour (with new edges) does NOT contain the penalized edges.
    # So the new tour has a much lower cost in `mod`.
    # 2-opt will see a massive improvement and will execute the move.
    # Will it stop there?
    # After the move, the tour consists of:
    # - All other original tour edges (true distance in mod)
    # - The two new edges (true distance in mod)
    # Are there any other improving moves in `mod`?
    # Any other move would involve removing an edge with true distance and adding an edge with huge penalty (1e6).
    # So no other move is improving.
    # Thus, 2-opt will perform exactly this one move and stop.
    
    # This ensures the tour is perturbed by exactly this specific move.
    # Then the evaluator runs 2-opt on TRUE distances, which will optimize from this new starting point.
    
    # Ensure diagonal is 0
    np.fill_diagonal(mod, 0)
    
    return mod
# EVOLVE-BLOCK-END
