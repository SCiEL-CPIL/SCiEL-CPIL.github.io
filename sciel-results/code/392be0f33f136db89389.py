"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses efficient gain computation and time limit.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    # Compute initial cut size efficiently
    diff_mask = (labels[:, None] != labels[None, :])
    current_cut = int((adj * diff_mask).sum() // 2)
    
    # Precompute degree
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Order vertices by degree (descending) for better search
    vertex_order = np.argsort(-degrees)
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in vertex_order:
            if time.time() - start_time > time_limit:
                break
            # Gain from flipping v = edges_to_same - edges_to_opposite
            # = (degree[v] - edges_to_opposite) - edges_to_opposite
            # = degree[v] - 2 * edges_to_opposite
            lv = labels[v]
            opposite_count = int(adj[v][labels != lv].sum())
            gain = degrees[v] - 2 * opposite_count
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def _two_flip_search(labels, adjacency, time_limit, rng):
    """
    Try 2-vertex swaps to escape local optima.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    degrees = adj.sum(axis=1).astype(np.int32)
    
    diff_mask = (labels[:, None] != labels[None, :])
    current_cut = int((adj * diff_mask).sum() // 2)
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n):
                if time.time() - start_time > time_limit:
                    break
                # Only consider swapping if they have different labels
                if labels[i] == labels[j]:
                    continue
                # Compute gain of swapping i and j
                # For vertex i: currently label li, neighbors with label li contribute to same, others to cut
                # After swap: i has label lj, j has label li
                # This is complex; approximate by checking if swap helps
                li = labels[i]
                lj = labels[j]
                
                # Count edges from i to vertices with label lj (will become same after swap)
                i_to_lj = int(adj[i][labels == lj].sum())
                # Count edges from i to vertices with label li (will become cut after swap)  
                i_to_li = int(adj[i][labels == li].sum())
                
                # Count edges from j to vertices with label li (will become same after swap)
                j_to_li = int(adj[j][labels == li].sum())
                # Count edges from j to vertices with label lj (will become cut after swap)
                j_to_lj = int(adj[j][labels == lj].sum())
                
                # Edge between i and j: if adj[i,j], it was cut (different labels), after swap still cut
                # So no change from the i-j edge itself
                
                # Net change: 
                # i loses i_to_li from cut, gains i_to_lj to cut
                # j loses j_to_lj from cut, gains j_to_li to cut
                # But we must be careful not to double count the i-j edge
                # Actually, let's just compute the change directly
                gain = (i_to_lj - i_to_li) + (j_to_li - j_to_lj)
                # Subtract 2 if adj[i,j] because that edge is counted in both i_to_lj and j_to_li
                # Wait, if labels[i] != labels[j], then adj[i,j] is a cut edge
                # i_to_lj includes j (since labels[j] == lj), so adj[i,j] is counted
                # j_to_li includes i (since labels[i] == li), so adj[i,j] is counted
                # After swap, i has lj, j has li, so adj[i,j] is still cut
                # So the i-j edge doesn't change, but it was counted in both gains
                # We need to subtract 2*adj[i,j] if it's 1
                if adj[i, j]:
                    gain -= 2
                
                if gain > 0:
                    labels[i] = lj
                    labels[j] = li
                    current_cut += gain
                    improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def partition(adjacency):
    n = len(adjacency)
    # Deadline guard
    global_deadline = time.time() + 110  # 110 seconds out of 120s, leave safety margin
    rng = np.random.default_rng(seed=42)
    
    best_labels = None
    best_cut = -1
    
    # Strategy: Multiple random restarts with local search + 2-flip refinement.
    
    num_restarts = 0
    max_restarts = 200  # Upper bound, but time will limit it
    
    while time.time() < global_deadline - 5 and num_restarts < max_restarts:
        # Generate diverse initial labels
        if num_restarts == 0:
            # First start: alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Second start: all zeros
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 2:
            # Third start: all ones
            labels = np.ones(n, dtype=int)
        elif num_restarts == 3:
            # Fourth start: based on degree (high degree -> 1, low -> 0)
            deg = adjacency.sum(axis=1)
            median_deg = np.median(deg)
            labels = (deg > median_deg).astype(int)
        else:
            # Random with bias toward 50/50
            labels = rng.integers(0, 2, size=n)
        
        # Run local search with a time limit
        remaining_time = max(1.0, (global_deadline - 5) - time.time())
        # Give each restart a portion of the remaining time
        ls_time_limit = max(0.05, remaining_time / (num_restarts + 20))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        # Try 2-flip refinement if we have time
        if time.time() < global_deadline - 10:
            two_flip_time = min(1.0, max(0.1, (global_deadline - 10) - time.time()))
            refined_labels, cut = _two_flip_search(refined_labels, adjacency, two_flip_time, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        num_restarts += 1
    
    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
