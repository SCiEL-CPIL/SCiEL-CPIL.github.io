"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import deque

# Cache for MST lower bounds per (frozenset of unvisited) is too heavy for 50 nodes.
# Instead, we use a lightweight geometric heuristic.

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node for a TSP tour construction.
    
    Strategy:
    1. Compute a composite score for each unvisited node.
    2. The score combines:
       - Distance from current node (greedy).
       - Distance from destination node (lookahead to close the loop).
       - Angle penalty: Prefer nodes that continue the current direction of travel
         to minimize sharp turns and potential edge crossings (geometric flow).
    3. Return the node with the best (lowest) score.
    """
    if len(unvisited_nodes) == 0:
        # Should not happen if called correctly, but safe fallback
        return destination_node
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Vectorized operations for speed
    # d_cur[i] = distance from current to unvisited[i]
    d_cur = distance_matrix[current_node, unvisited_nodes]
    # d_dest[i] = distance from destination to unvisited[i]
    d_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # We need coordinates to compute angles. 
    # The problem statement implies Euclidean instances. 
    # However, we are only given distance_matrix. 
    # We cannot recover coordinates exactly from distance matrix without heavy computation (multidimensional scaling) which is too slow per step.
    
    # Alternative: Use distance matrix properties to approximate "turning cost".
    # Triangle inequality: d(cur, dest) <= d(cur, u) + d(u, dest)
    # If d(cur, u) + d(u, dest) is close to d(cur, dest), the node u is "on the way" to destination.
    # This is a proxy for alignment.
    
    d_cur_dest = distance_matrix[current_node, destination_node]
    
    # Sum of distances if we go through u
    d_via = d_cur + d_dest
    
    # "Detour" cost: How much longer is the path cur->u->dest compared to cur->dest?
    # Low detour means u is aligned with the direct path to destination.
    detour = d_via - d_cur_dest
    
    # Combine metrics:
    # We want to minimize d_cur (greedy) but also avoid creating a long detour that makes closing the tour expensive.
    # A common heuristic is: min( d_cur + alpha * d_dest )
    # But we can do better by penalizing nodes that are "behind" us relative to the destination.
    
    # Let's try a weighted sum:
    # Score = d_cur + beta * d_dest
    # beta balances local greediness vs global closure.
    # For TSP, a beta around 0.5-1.0 often works well.
    
    # Let's add a penalty for large detours to encourage "straight" paths.
    # Penalty = max(0, detour - epsilon)
    
    # To make it robust, let's normalize or just use linear combination.
    # Let's test: Score = d_cur + 0.5 * d_dest
    # This is the "greedy with lookahead" heuristic.
    
    # Even better: The "Furthest Insertion" or "Nearest Neighbor with Angle" is hard without coords.
    # Let's stick to distance-based lookahead.
    
    # Empirically, for uniform random points, minimizing d_cur + d_dest / 2 or similar works better than pure NN.
    
    # Let's try: Score = d_cur + 0.8 * d_dest
    # Why 0.8? It's a compromise. If we pick a node far from dest, the final jump will be huge.
    
    # Let's refine: 
    # If we are close to the end (few unvisited), we should prioritize d_dest more.
    # If we are at the start, prioritize d_cur.
    
    n_unvisited = len(unvisited_nodes)
    # Adaptive weight
    # Early on: weight_dest is small. Late on: weight_dest is large.
    # weight_dest = 0.5 + 0.5 * (1 - n_unvisited / 50) ? 
    # Or simply: weight_dest = 1.0 / (1.0 + n_unvisited)
    
    # Let's try a fixed weight first, it's simpler and often robust.
    # Research suggests "Nearest Neighbor with Lookahead" (min d(i,j) + d(j,k)) is a good baseline.
    # Here k is destination.
    # So Score = d_cur + d_dest.
    
    # Let's try minimizing (d_cur + d_dest) directly.
    scores = d_cur + d_dest
    
    # To break ties or add slight variety/geometric constraint without coords:
    # We can penalize nodes that are very close to current but far from dest? No, that's covered by sum.
    
    # Let's add a small penalty for "detour" to encourage straight lines.
    # detour = d_cur + d_dest - d_cur_dest
    # If detour is large, it's a bad turn.
    # Score = d_cur + d_dest + lambda * max(0, detour - d_cur_dest)
    # Actually, detour is always >= 0.
    
    # Let's try: Score = d_cur + 0.5 * d_dest
    # This is a standard improvement over NN.
    
    # Let's implement a dynamic weight:
    # If few nodes left, we must consider d_dest heavily.
    # If many nodes left, d_cur is more important to keep tour tight locally.
    
    # Let's try: weight = 0.5
    # Score = d_cur + weight * d_dest
    
    # Wait, the MST knowledge suggests global connectivity matters.
    # But we can't compute MST per step.
    
    # Let's stick to the lookahead heuristic:
    # Minimize d(current, u) + d(u, destination)
    
    idx = np.argmin(scores)
    
    return int(unvisited_nodes[idx])

# EVOLVE-BLOCK-END
