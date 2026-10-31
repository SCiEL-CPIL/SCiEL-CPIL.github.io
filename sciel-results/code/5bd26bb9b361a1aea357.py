"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = None
_DEADLINE_BUFFER = 10.0  # seconds safety margin before 150s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_BUFFER
    
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        # If only two left, pick the one that creates the shorter edge from current
        # (The other will be picked next, and then we close to dest)
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        # We want to minimize the edge we are creating now.
        # However, the edge to destination is fixed for the last node.
        # Actually, if we pick i, next is j, then edge is (current->i) + (i->j) + (j->dest)
        # If we pick j, next is i, then edge is (current->j) + (j->i) + (i->dest)
        # Since dist(i,j) == dist(j,i), we compare d0 + d_dest[0] vs d1 + d_dest[1]
        # Wait, d_dest is distance to destination_node.
        # Let's just compute both options.
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        cost_pick_0 = d0 + d_cross + d_dest_1
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])

    # --- Core Strategy: Convex Hull Peeling / Geometric Boundary Focusing ---
    
    # 1. Compute distances from current node to all unvisited
    d_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Compute distances from all unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Estimate "Hull-ness" or "Centrality"
    # We want to visit nodes that are on the "outside" of the remaining cluster.
    # Approximation: Distance from the centroid of the unvisited set.
    # Since we don't have coordinates, we use distance matrix properties.
    # A node is "central" if it is close to all others.
    # A node is "peripheral" (hull) if it is far from many others.
    
    # Sub-matrix of distances among unvisited nodes
    sub_dist = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # For each node, compute the mean distance to other unvisited nodes.
    # High mean distance -> likely on the hull/periphery.
    # Low mean distance -> likely in the center.
    mean_dist_to_others = np.mean(sub_dist, axis=1)
    
    # Normalize to 0..1 range for combining
    min_mean = np.min(mean_dist_to_others)
    max_mean = np.max(mean_dist_to_others)
    if max_mean > min_mean:
        hull_score = (mean_dist_to_others - min_mean) / (max_mean - min_mean)
    else:
        hull_score = np.zeros(n_unvisited)
        
    # 4. Combine Scores
    # We want a balance:
    # - Not too far from current (local continuity)
    # - Not too close to current (avoid zig-zagging if we are deep inside)
    # - High hull score (visit boundary nodes first)
    # - Good connection to destination (if few nodes left)
    
    # Normalize d_current
    min_d_curr = np.min(d_current)
    max_d_curr = np.max(d_current)
    if max_d_curr > min_d_curr:
        norm_d_curr = (d_current - min_d_curr) / (max_d_curr - min_d_curr)
    else:
        norm_d_curr = np.zeros(n_unvisited)
        
    # Normalize d_dest
    min_d_dest = np.min(d_dest)
    max_d_dest = np.max(d_dest)
    if max_d_dest > min_d_dest:
        norm_d_dest = (d_dest - min_d_dest) / (max_d_dest - min_d_dest)
    else:
        norm_d_dest = np.zeros(n_unvisited)

    # Heuristic weights:
    # As n_unvisited decreases, we care more about closing the loop (d_dest).
    # Early on, we care more about hull structure.
    
    if n_unvisited > 20:
        w_local = 0.3
        w_hull = 0.5
        w_dest = 0.2
    elif n_unvisited > 10:
        w_local = 0.4
        w_hull = 0.4
        w_dest = 0.2
    else:
        w_local = 0.5
        w_hull = 0.2
        w_dest = 0.3

    # Penalty for being "too close" to current? 
    # Actually, standard TSP heuristics usually prefer closer nodes.
    # But "Convex Hull Peeling" suggests we should prefer nodes that are far from the *center* of the remaining set,
    # even if they are not the absolute nearest to current, provided they are reasonable.
    
    # Let's refine the "Local" term. 
    # If we just use d_current, we get Nearest Neighbor.
    # If we use hull_score, we get Farthest from Center.
    
    # Combined Score (lower is better)
    # We want to MINIMIZE distance to current.
    # We want to MAXIMIZE hull_score (so subtract it).
    # We want to MINIMIZE distance to destination (especially later).
    
    # To prevent pure "Farthest from center" which might jump across the map,
    # we add a penalty for large jumps from current.
    
    # Let's try a composite score:
    # Score = w_local * norm_d_curr - w_hull * hull_score + w_dest * norm_d_dest
    
    # Wait, if w_hull is high, we pick the node FARTHEST from the centroid.
    # This might be a node far away from current.
    # We need to constrain this.
    
    # Alternative: "Angular Momentum" or "Sweep"
    # But without coordinates, angle is hard.
    
    # Let's stick to the Hull Score but constrain it.
    # Only consider the top K closest nodes from current, then among those, pick the one with highest hull score.
    # This ensures local continuity while enforcing global structure.
    
    top_k = min(10, n_unvisited)
    
    # Find indices of the top_k closest nodes to current
    # We want the k smallest values in d_current
    # np.argpartition gives the indices of the k smallest elements
    if top_k < n_unvisited:
        k_smallest_indices = np.argpartition(d_current, top_k)[:top_k]
    else:
        k_smallest_indices = np.arange(n_unvisited)
        
    # Among these candidates, score them
    cand_d_curr = d_current[k_smallest_indices]
    cand_hull = hull_score[k_smallest_indices]
    cand_d_dest = d_dest[k_smallest_indices]
    
    # Normalize within candidates for stability? No, global normalization is better for consistency.
    
    # Score for candidates
    # We want to pick the one that balances being close (cand_d_curr) and being on the hull (cand_hull).
    
    # Let's use a simple weighted sum on the candidates.
    # Since cand_d_curr is already small (top k), the variation is small.
    # cand_hull variation might be larger.
    
    # To make weights comparable, let's normalize the candidate features to 0..1
    min_cd = np.min(cand_d_curr)
    max_cd = np.max(cand_d_curr)
    if max_cd > min_cd:
        norm_cd = (cand_d_curr - min_cd) / (max_cd - min_cd)
    else:
        norm_cd = np.zeros(len(cand_d_curr))
        
    min_ch = np.min(cand_hull)
    max_ch = np.max(cand_hull)
    if max_ch > min_ch:
        norm_ch = (cand_hull - min_ch) / (max_ch - min_ch)
    else:
        norm_ch = np.zeros(len(cand_hull))
        
    min_cd_d = np.min(cand_d_dest)
    max_cd_d = np.max(cand_d_dest)
    if max_cd_d > min_cd_d:
        norm_cd_d = (cand_d_dest - min_cd_d) / (max_cd_d - min_cd_d)
    else:
        norm_cd_d = np.zeros(len(cand_d_dest))

    # We want to MINIMIZE local dist, MINIMIZE dest dist.
    # We want to MAXIMIZE hull score (so we minimize -hull score).
    
    # Score = w_local * norm_cd - w_hull * norm_ch + w_dest * norm_cd_d
    
    # Note: norm_ch is 0 for central, 1 for hull. So -norm_ch favors hull.
    
    scores = w_local * norm_cd - w_hull * norm_ch + w_dest * norm_cd_d
    
    best_idx_in_candidates = np.argmin(scores)
    
    # Map back to original index
    best_orig_idx = k_smallest_indices[best_idx_in_candidates]
    
    return int(unvisited_nodes[best_orig_idx])
# EVOLVE-BLOCK-END
