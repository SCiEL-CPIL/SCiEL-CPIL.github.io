"""Traveling salesman constructive heuristic with multi-step lookahead and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = None
_DEADLINE = 120.0  # seconds


def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME


def _time_exceeded():
    return (time.time() - _START_TIME) > (_DEADLINE - 2.0)  # 2s safety margin


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node using a composite heuristic that balances:
    1. Distance to candidate node
    2. Distance from candidate to destination (lookahead)
    3. Distance from candidate to its own nearest neighbor (connectivity cost)
    4. 2-step lookahead: min distance from candidate's neighbors to destination
    
    Weights are tuned to favor paths that lead to shorter overall tours.
    """
    if len(unvisited_nodes) == 0:
        return int(current_node)
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    if _time_exceeded():
        # Fallback to simple nearest neighbor for safety
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Get distances from current to all unvisited
    d_to_candidates = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from candidates to destination
    d_candidates_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # For each candidate, find the minimum distance from that candidate to any OTHER unvisited node
    # This approximates the "connectivity cost" of choosing this node
    # To keep it fast, we use a sampled approach or direct computation for small sets
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited <= 30:
        # Direct computation: for each candidate i, min distance to other unvisited nodes
        # dist_matrix between unvisited nodes
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Set diagonal to inf so we don't pick self
        np.fill_diagonal(sub_dm, np.inf)
        min_dist_to_other = np.min(sub_dm, axis=1)
    else:
        # For larger sets, use a cheaper approximation: 
        # just use the distance to the nearest few candidates
        # Sample up to 10 other candidates for each
        min_dist_to_other = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            # Exclude self
            other_indices = np.delete(np.arange(n_unvisited), i)
            if len(other_indices) > 10:
                # Sample 10
                sample_idx = np.random.choice(other_indices, 10, replace=False)
            else:
                sample_idx = other_indices
            min_dist_to_other[i] = np.min(distance_matrix[unvisited_nodes[i], unvisited_nodes[sample_idx]])
    
    # 2-step lookahead: for each candidate, find the best (min) distance from 
    # candidate's nearest neighbor to the destination
    # Approximation: for each candidate, look at its 2 nearest neighbors among unvisited,
    # and take the min distance from those neighbors to destination
    if n_unvisited <= 30:
        sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # For each candidate i, find the 2 nearest other candidates
        two_step_scores = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            # Get distances from candidate i to all others
            dists_i = sub_dm[i].copy()
            dists_i[i] = np.inf
            # Get the 2 smallest
            k = min(2, n_unvisited - 1)
            two_nearest = np.argpartition(dists_i, k)[:k]
            # Distance from those neighbors to destination
            two_step_scores[i] = np.min(d_candidates_to_dest[two_nearest])
    else:
        # Cheaper: just use the candidate's distance to destination as a proxy
        # (less accurate but faster)
        two_step_scores = d_candidates_to_dest.copy()
    
    # Composite score: lower is better
    # Normalize each component to [0, 1] for stable weighting
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    score1 = normalize(d_to_candidates)
    score2 = normalize(d_candidates_to_dest)
    score3 = normalize(min_dist_to_other)
    score4 = normalize(two_step_scores)
    
    # Weights: emphasize immediate distance and 2-step lookahead
    w1 = 0.40  # distance to candidate
    w2 = 0.25  # distance from candidate to destination
    w3 = 0.20  # connectivity cost (min dist to other unvisited)
    w4 = 0.15  # 2-step lookahead
    
    composite = w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4
    
    best_idx = np.argmin(composite)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
