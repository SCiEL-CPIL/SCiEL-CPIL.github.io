"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and potentially phase, though we can infer phase from unvisited_nodes length
_start_time = None

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    # Deadline guard
    # 150s limit. We have ~50 calls. 
    # If we are getting close to the limit, we should just return NN to be safe.
    # But NN is fast. The expensive part is if we did heavy search. 
    # This heuristic is O(N) per call. 50 * 50 = 2500 ops. Very fast.
    # No need for complex deadline logic unless we add heavy search.
    
    # Strategy: Phase-Aware Hybrid
    # 
    # Insight: 
    # 1. The "exit cost" heuristic (current program) tends to create "islands" or 
    #    suboptimal local structures because it prioritizes having a cheap exit 
    #    over global connectivity. It often scores worse than pure NN on uniform 
    #    Euclidean instances in the early/mid phase.
    # 2. However, pure NN suffers from a long closing edge (distance from last 
    #    node to destination).
    # 3. Therefore, we use Pure Nearest Neighbor for the first ~70-80% of the 
    #    tour to build a compact, low-cost path.
    # 4. For the last ~20-30% of the tour, we switch to a destination-aware 
    #    heuristic to steer the tour back towards the start/destination.
    
    # Threshold: Start considering destination when fewer than 15 nodes remain.
    # This allows the tour to "wind up" towards the destination.
    threshold = 15
    
    if n_unvisited > threshold:
        # Phase 1: Expansion. Pure Nearest Neighbor.
        # This minimizes the immediate edge cost, which is the dominant factor 
        # for the bulk of the tour length.
        d_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
        best_idx = np.argmin(d_current_to_unvisited)
        return int(unvisited_nodes[best_idx])
    else:
        # Phase 2: Closing. Destination-Aware Scoring.
        # We want to minimize the total cost of:
        # 1. Moving from current to candidate.
        # 2. Moving from candidate to destination (since destination is the 
        #    final step for the remaining nodes, or close to it).
        # Actually, a better score for the closing phase is:
        # cost(current -> candidate) + cost(candidate -> destination)
        # This effectively tries to form a triangle current-candidate-destination
        # with minimum perimeter, steering the tour back.
        
        d_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
        d_unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Score = entry cost + exit cost (to destination)
        scores = d_current_to_unvisited + d_unvisited_to_dest
        
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
