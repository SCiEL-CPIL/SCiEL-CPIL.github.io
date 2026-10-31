"""Traveling salesman constructive heuristic: hybrid MST-aware + nearest neighbour."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: track start time for the entire evaluation
_EVAL_START_TIME = None


def _get_elapsed():
    """Return elapsed time since evaluation start."""
    global _EVAL_START_TIME
    if _EVAL_START_TIME is None:
        _EVAL_START_TIME = time.time()
    return time.time() - _EVAL_START_TIME


def _mst_weight(nodes, distance_matrix):
    """Compute MST weight of the subgraph induced by nodes using Prim's algorithm.
    
    Args:
        nodes: array of node indices
        distance_matrix: full distance matrix
        
    Returns:
        Total weight of the MST
    """
    if len(nodes) <= 1:
        return 0.0
    
    n = len(nodes)
    # Use Prim's algorithm starting from the first node
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    
    mst_weight = 0.0
    
    for _ in range(n):
        # Find the unvisited node with minimum distance
        unvisited_mask = ~in_mst
        if not np.any(unvisited_mask):
            break
        
        # Set visited nodes to inf for argmin
        candidate_dists = min_dist.copy()
        candidate_dists[in_mst] = np.inf
        u = int(np.argmin(candidate_dists))
        
        in_mst[u] = True
        mst_weight += min_dist[u]
        
        # Update distances for neighbors
        for v in range(n):
            if not in_mst[v]:
                d = distance_matrix[nodes[u], nodes[v]]
                if d < min_dist[v]:
                    min_dist[v] = d
    
    return mst_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Select the next node using a hybrid heuristic combining nearest neighbour
    with MST-awareness.
    
    The score for each candidate is:
        alpha * d(current, candidate) + beta * (MST(remaining_after) - MST(remaining_before))
    
    This balances immediate step cost with the structural impact on the remaining
    graph's connectivity.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        # Should not happen, but return destination as fallback
        return int(destination_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Deadline check: if we're running out of time, fall back to nearest neighbour
    if _get_elapsed() > 110.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, compute full MST-aware score
    if n_unvisited <= 20:
        # Compute MST of current unvisited set (baseline)
        baseline_mst = _mst_weight(unvisited_nodes, distance_matrix)
        
        # For each candidate, compute the MST of the remaining set after removing it
        scores = np.zeros(n_unvisited)
        dists = distance_matrix[current_node, unvisited_nodes]
        
        for i in range(n_unvisited):
            # Remaining nodes after picking unvisited_nodes[i]
            remaining = np.delete(unvisited_nodes, i)
            if len(remaining) <= 1:
                remaining_mst = 0.0
            else:
                remaining_mst = _mst_weight(remaining, distance_matrix)
            
            # Score: immediate distance + change in MST weight
            # We want to minimize this
            scores[i] = dists[i] + 0.5 * (remaining_mst - baseline_mst)
        
        return int(unvisited_nodes[np.argmin(scores)])
    
    else:
        # For larger sets, use a faster approximation
        # Combine nearest neighbour with a simple structural penalty
        
        # Get distances from current node
        dists = distance_matrix[current_node, unvisited_nodes]
        
        # Compute a simple "isolation" penalty: for each candidate,
        # the max distance to other unvisited nodes (minimax centrality)
        # This is expensive, so use a subsampled version for large sets
        
        if n_unvisited <= 50:
            # Compute pairwise distances among unvisited nodes
            sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
            
            # For each candidate, compute max distance to others (isolation risk)
            # Use the row max as a proxy
            max_dists = np.max(sub_matrix, axis=1)
            
            # Also compute mean distance as a connectivity measure
            mean_dists = np.mean(sub_matrix, axis=1)
            
            # Hybrid score: immediate distance + weighted isolation penalty
            alpha = 1.0
            beta = 0.1
            gamma = 0.05
            
            scores = alpha * dists + beta * max_dists + gamma * mean_dists
        else:
            # For very large sets, just use nearest neighbour with a small lookahead
            # Pick the 5 nearest and evaluate them more carefully
            k = min(5, n_unvisited)
            nearest_idx = np.argsort(dists)[:k]
            nearest_nodes = unvisited_nodes[nearest_idx]
            
            # Compute a simple structural score for these candidates
            scores = np.zeros(k)
            for i, node_idx in enumerate(nearest_idx):
                # Distance from current
                d_current = dists[node_idx]
                
                # Average distance from this candidate to other unvisited nodes
                # (sample if needed)
                if n_unvisited > 30:
                    # Sample up to 20 other nodes
                    sample_idx = np.random.choice(n_unvisited - 1, size=min(20, n_unvisited - 1), replace=False)
                    # Adjust indices to exclude the candidate
                    sample_idx = sample_idx[sample_idx >= node_idx] + 1
                    sample_idx = sample_idx[sample_idx < n_unvisited]
                    if len(sample_idx) > 0:
                        avg_dist = np.mean(distance_matrix[unvisited_nodes[node_idx], unvisited_nodes[sample_idx]])
                    else:
                        avg_dist = 0.0
                else:
                    mask = np.ones(n_unvisited, dtype=bool)
                    mask[node_idx] = False
                    avg_dist = np.mean(distance_matrix[unvisited_nodes[node_idx], unvisited_nodes[mask]])
                
                scores[i] = d_current + 0.1 * avg_dist
            
            best_local = np.argmin(scores)
            return int(unvisited_nodes[nearest_idx[best_local]])
        
        return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
