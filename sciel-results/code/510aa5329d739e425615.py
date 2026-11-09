"""TSP constructive heuristic with MST-leaf pruning and global connectivity awareness."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None
_DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _check_deadline():
    """Return True if deadline is approaching."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return (time.time() - _start_time) > (150.0 - _DEADLINE_MARGIN)


def _prim_mst_weight(dm_sub):
    """Compute MST weight of a small distance matrix using Prim's algorithm."""
    n = dm_sub.shape[0]
    if n <= 1:
        return 0.0
    if n == 2:
        return float(dm_sub[0, 1])
    
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    mst_weight = 0.0
    
    for _ in range(n):
        candidates = np.where(~in_mst)[0]
        if len(candidates) == 0:
            break
        min_idx = candidates[np.argmin(min_dist[candidates])]
        mst_weight += min_dist[min_idx]
        in_mst[min_idx] = True
        # Update distances
        for j in range(n):
            if not in_mst[j] and dm_sub[min_idx, j] < min_dist[j]:
                min_dist[j] = dm_sub[min_idx, j]
    
    return mst_weight


def _compute_mst_leaf_costs(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Compute MST-based leaf pruning scores for each unvisited candidate.
    
    For each candidate, we compute:
    1. The MST weight of the subgraph induced by {candidate} ∪ remaining_unvisited ∪ {destination}
    2. The degree of the candidate in that MST (to identify if it's a leaf)
    3. The connection cost from current_node to candidate
    
    The idea: prefer candidates that are leaves of the MST (natural endpoints of paths)
    with low connection cost from current_node.
    """
    n_unvisited = len(unvisited_nodes)
    
    # Precompute the full set of relevant nodes
    all_nodes = np.concatenate([unvisited_nodes, [destination_node]])
    n_all = len(all_nodes)
    
    # Build the distance matrix for all relevant nodes + current_node
    # We need: current_node, all unvisited, destination
    relevant = np.concatenate([[current_node], unvisited_nodes, [destination_node]])
    n_rel = len(relevant)
    dm_rel = distance_matrix[np.ix_(relevant, relevant)]
    
    # Indices in dm_rel: 0=current, 1..n_unvisited=unvisited, n_unvisited+1=destination
    dest_idx = n_unvisited + 1
    
    scores = np.zeros(n_unvisited)
    
    for i in range(n_unvisited):
        cand_idx = i + 1  # index in dm_rel
        cand = unvisited_nodes[i]
        
        # Direct cost from current to candidate
        d_current_to_cand = dm_rel[0, cand_idx]
        
        # Build subgraph: candidate + remaining unvisited + destination
        # In dm_rel indices: cand_idx, all unvisited except i, dest_idx
        sub_indices = []
        for j in range(n_unvisited):
            if j != i:
                sub_indices.append(j + 1)
        sub_indices.append(dest_idx)
        
        # Add candidate
        sub_indices.insert(0, cand_idx)
        
        n_sub = len(sub_indices)
        
        if n_sub <= 1:
            mst_w = 0.0
            is_leaf = True
        elif n_sub == 2:
            mst_w = float(dm_rel[sub_indices[0], sub_indices[1]])
            is_leaf = True  # Both nodes are leaves in a 2-node MST
        else:
            dm_sub = dm_rel[np.ix_(sub_indices, sub_indices)]
            mst_w = _prim_mst_weight(dm_sub)
            
            # Determine if candidate (index 0 in sub) is a leaf in the MST
            # Re-run Prim's but track edges
            n_s = n_sub
            in_mst = np.zeros(n_s, dtype=bool)
            min_dist = np.full(n_s, np.inf)
            min_dist[0] = 0.0
            parent = np.full(n_s, -1, dtype=int)
            
            for _ in range(n_s):
                candidates = np.where(~in_mst)[0]
                if len(candidates) == 0:
                    break
                min_idx = candidates[np.argmin(min_dist[candidates])]
                in_mst[min_idx] = True
                # Update distances and track parent
                for j in range(n_s):
                    if not in_mst[j] and dm_sub[min_idx, j] < min_dist[j]:
                        min_dist[j] = dm_sub[min_idx, j]
                        parent[j] = min_idx
            
            # Count degree of node 0 (candidate) in MST
            degree = 0
            for j in range(n_s):
                if parent[j] == 0:
                    degree += 1
            # Also check if 0's parent is someone (0 is root, so parent[0]=-1)
            # In Prim's from 0, node 0 is root, so its degree is just how many children it has
            is_leaf = (degree <= 1)
        
        # Score: prefer low connection cost, and bonus for being a leaf
        # If candidate is a leaf in MST, it's a natural path endpoint
        # Weight: primary is connection cost, secondary is MST weight, tertiary is leaf bonus
        w_connect = 1.0
        w_mst = 0.15
        w_leaf_bonus = 0.1
        
        score = w_connect * d_current_to_cand + w_mst * mst_w
        if not is_leaf:
            score += w_leaf_bonus  # Penalty for not being a leaf
        
        scores[i] = score
    
    return scores


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select next node using MST-leaf pruning heuristic.
    
    The key insight: in an optimal TSP tour, each city has exactly two neighbors.
    When building the tour incrementally, the "next" city should ideally be one that
    can be naturally appended to the path. MST leaf pruning identifies cities that
    are leaves in the minimum spanning tree of the remaining subgraph - these are
    natural endpoints for path extension.
    
    We score each candidate by:
    1. Direct connection cost from current node (primary)
    2. MST weight of the remaining subgraph after picking this candidate (secondary)
    3. Whether the candidate is a leaf in that MST (bonus for leaves)
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    if len(unvisited_nodes) == 0:
        return int(destination_node)
    
    # If only one node left, pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if close, fall back to simple nearest neighbor
    if _check_deadline():
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    # For very small instances, use the full MST-leaf heuristic
    # For larger instances, use a hybrid approach
    if n_unvisited <= 15:
        # Full MST-leaf pruning
        scores = _compute_mst_leaf_costs(current_node, destination_node, unvisited_nodes, distance_matrix)
        best_idx = np.argmin(scores)
        return int(unvisited_nodes[best_idx])
    
    else:
        # Hybrid: use a reduced candidate set for MST computation
        # First, get top-k nearest neighbors as candidates
        k = min(n_unvisited, 8)
        dists = distance_matrix[current_node, unvisited_nodes]
        top_k_indices = np.argsort(dists)[:k]
        candidates = unvisited_nodes[top_k_indices]
        
        # Also include the node closest to destination (to maintain connectivity)
        dest_dists = distance_matrix[destination_node, unvisited_nodes]
        closest_to_dest = unvisited_nodes[np.argmin(dest_dists)]
        if closest_to_dest not in candidates:
            candidates = np.append(candidates, closest_to_dest)
        
        # Compute MST-leaf scores for the reduced candidate set
        scores = _compute_mst_leaf_costs(current_node, destination_node, candidates, distance_matrix)
        best_idx = np.argmin(scores)
        return int(candidates[best_idx])
# EVOLVE-BLOCK-END
