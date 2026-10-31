"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Selects the next node using a minimax bottleneck heuristic.
    
    Strategy:
    Instead of minimizing the sum of distances (standard TSP objective), we frame
    the construction as a graph traversal problem where we aim to minimize the
    maximum edge weight in the final tour (bottleneck TSP intuition).
    
    The score for a candidate node is:
    1. The distance from the current node to the candidate (the edge we are adding).
    2. Plus a penalty term that reflects the "bottleneck" potential: 
       the minimum possible maximum edge weight if we were to connect the 
       remaining unvisited nodes (excluding the candidate) in a star-like or 
       path-like structure. 
       
    A simpler, effective proxy for "minimizing the maximum edge" in a constructive
    heuristic is to avoid creating an edge that is significantly longer than the
    current "typical" edge length, or more directly, to choose the next node such
    that the resulting tour segment doesn't force a huge jump later.
    
    However, the directive specifically suggests: "selecting the next node that 
    minimizes the distance to the farthest unvisited city (minimax strategy)".
    
    Let's interpret this carefully. If we pick node X next, we add edge (current, X).
    The "bottleneck" of the *entire* tour will be at least max(current_max_edge, dist(current, X)).
    But we can't see the future.
    
    A different interpretation of "minimax strategy" in a constructive context:
    We want to keep the "diameter" of the remaining problem small.
    If we pick a node X that is far from many unvisited nodes, we might be forced
    to make long jumps later.
    
    Let's try a score that combines:
    1. dist(current, candidate): The immediate cost.
    2. A "regret" or "bottleneck" term: The maximum distance from the candidate 
       to any other unvisited node? No, that would favor central nodes heavily.
       
    Actually, the prompt says: "minimizes the distance to the farthest unvisited city".
    This phrasing is slightly ambiguous. Does it mean:
    - Minimize (dist(current, candidate))? No, that's NN.
    - Minimize (max_{u in unvisited \ candidate} dist(candidate, u))? This is a centrality measure (1-center).
    - Or does it mean we look at the edge we are adding, and compare it to the "bottleneck" of the rest?
    
    Let's look at the "bottleneck TSP" literature. The bottleneck TSP minimizes the largest edge in the tour.
    A heuristic for this often involves avoiding the longest possible edges.
    
    Let's try a strategy that explicitly penalizes long edges relative to the current state.
    Score(candidate) = dist(current, candidate) + lambda * (max_{u in unvisited} dist(candidate, u) - min_{u in unvisited} dist(candidate, u))
    This penalizes candidates that are "outliers" relative to the remaining set, which could lead to long edges later.
    
    Alternatively, a pure "minimax" step might be:
    We want to ensure that the edge we pick is not the maximum edge in the final tour.
    Since we don't know the final tour, we approximate by ensuring the edge we pick is not larger than the "typical" distance in the remaining cluster.
    
    Let's try a different structural change: **Edge Swapping / 2-Opt Lookahead**.
    The current program is a greedy constructive heuristic. A different family is **Lookahead Heuristic**.
    For each candidate, simulate the next 1-2 steps (greedy) and evaluate the resulting path length.
    This is structurally different from a single-step scoring function.
    
    However, the directive explicitly asks for: "Model the TSP construction as a graph traversal problem where the objective is to minimize the maximum edge weight... by selecting the next node that minimizes the distance to the farthest unvisited city (minimax strategy)."
    
    Okay, I will follow the directive's specific suggestion.
    "minimizes the distance to the farthest unvisited city"
    This sounds like: For each candidate c, calculate D_c = max_{u in unvisited, u!=c} dist(c, u).
    Then pick c that minimizes D_c?
    This would pick the most central node in the remaining set.
    But we also must consider the connection from `current_node`.
    If the most central node is very far from `current_node`, picking it is bad.
    
    So the score should be a combination:
    Score(c) = dist(current, c) + w * max_{u in unvisited, u!=c} dist(c, u)
    
    This creates a bias towards nodes that are both close to the current position AND central to the remaining unvisited nodes.
    This is a "bottleneck-aware" nearest neighbor.
    
    Let's implement this.
    """
    if len(unvisited_nodes) == 0:
        return -1
        
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_total = len(distance_matrix)
    n_remaining = len(unvisited_nodes)
    
    # 1. Distance from current to each candidate
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. For each candidate, the max distance to any other unvisited node.
    # This is the "bottleneck" potential if we were to connect this candidate to the rest.
    # sub_matrix shape: (n_remaining, n_remaining)
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # We want max distance to *other* nodes. The diagonal is 0.
    # To find max distance to others, we can just take the max of the row, 
    # but we must exclude the self-distance (0) which is min, so max is unaffected by 0 
    # unless all distances are 0 (impossible for distinct points).
    # So row_max = np.max(sub_matrix, axis=1) gives the max distance to any node in the set (including self=0).
    # Since distances are positive, the max will be to some other node.
    max_dist_to_unvisited = np.max(sub_matrix, axis=1)
    
    # 3. Dynamic weighting
    # Early on, we might care less about the bottleneck of the *entire* remaining set 
    # because the set is large and the max distance is large for everyone.
    # Late on, the set is small, and the max distance is more specific.
    
    # Let's use a weight that increases as we get closer to the end, 
    # to ensure we don't leave a "long edge" for the final connections.
    # Or maybe keep it constant?
    
    # Let's try a constant weight first to see if the "centrality" aspect helps.
    # The scale of dists_from_current and max_dist_to_unvisited might differ.
    # dists_from_current is a single edge.
    # max_dist_to_unvisited is the diameter of the sub-cluster from that node.
    
    # To balance them, we can normalize by the mean distance in the sub_matrix or similar.
    # But simple addition might work if scales are similar.
    # In a uniform 50-point instance, distances are O(1).
    
    w_bottleneck = 0.5  # Weight for the maximin term
    
    scores = dists_from_current + w_bottleneck * max_dist_to_unvisited
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
