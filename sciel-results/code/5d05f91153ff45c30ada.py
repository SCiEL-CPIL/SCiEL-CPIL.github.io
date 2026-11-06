"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses a time limit to stop before exceeding the budget.
    Optimized: maintain cut value incrementally.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int32)
    
    # Compute initial cut size: count edges crossing the partition
    # For each vertex i, count neighbors j with different label
    # Cut = 0.5 * sum_i sum_j adj[i,j] * (labels[i] != labels[j])
    # Use vectorized computation
    diff = labels[:, None] != labels[None, :]
    current_cut = int((adj * diff).sum() // 2)
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track number of edges from each vertex to the opposite set
    # opp_count[v] = number of neighbors of v with label != labels[v]
    # This can be updated incrementally when a vertex flips
    
    # Initialize opp_count: for each v, count neighbors u where labels[u] != labels[v]
    opp_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_count[v] = adj[v][labels != labels[v]].sum()
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            # Gain from flipping v:
            # Currently, opp_count[v] edges from v go to opposite set (these are cut)
            # After flipping, those edges become non-cut, and (degrees[v] - opp_count[v]) become cut
            # New cut contribution from v's edges: degrees[v] - opp_count[v]
            # Old cut contribution from v's edges: opp_count[v]
            # Change in cut = (degrees[v] - opp_count[v]) - opp_count[v] = degrees[v] - 2*opp_count[v]
            gain = degrees[v] - 2 * opp_count[v]
            if gain > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update opp_count for v and its neighbors
                # For v: its opp_count becomes degrees[v] - opp_count[v]
                opp_count[v] = degrees[v] - opp_count[v]
                
                # For each neighbor u of v:
                # Before flip: if labels[u] != labels[v], then u was in v's opposite set
                # After flip: labels[v] changed, so the relationship inverts for all neighbors
                # For neighbor u:
                #   If labels[u] == old_labels[v] (i.e., u was in same set as v before),
                #     then after flip, u is in opposite set of v, so u's opp_count increases by 1
                #   If labels[u] != old_labels[v] (i.e., u was in opposite set of v before),
                #     then after flip, u is in same set as v, so u's opp_count decreases by 1
                # 
                # Since labels[v] has already been flipped, we can check:
                #   For neighbor u: if labels[u] == labels[v] (new), then u was in opposite before -> opp_count[u] -= 1
                #   For neighbor u: if labels[u] != labels[v] (new), then u was in same before -> opp_count[u] += 1
                
                neighbors = np.where(adj[v] > 0)[0]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        # u and v are now in same set, so u's edge to v is no longer cut
                        opp_count[u] -= 1
                    else:
                        # u and v are in different sets, so u's edge to v is now cut
                        opp_count[u] += 1
                
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
    
    # Strategy: Multiple random restarts with local search.
    # Also try a simple alternating start.
    
    num_restarts = 0
    max_restarts = 200  # Upper bound, but time will limit it
    
    while time.time() < global_deadline - 5 and num_restarts < max_restarts:
        # Generate random initial labels
        if num_restarts == 0:
            # First start: alternating
            labels = np.arange(n) % 2
        elif num_restarts == 1:
            # Second start: all zeros (will be improved by local search)
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 2:
            # Third start: all ones
            labels = np.ones(n, dtype=int)
        else:
            labels = rng.integers(0, 2, size=n)
        
        # Run local search with a time limit that shrinks as we approach the deadline
        remaining_time = max(1.0, (global_deadline - 5) - time.time())
        # Give each restart a portion of the remaining time, but at least 0.05s
        ls_time_limit = max(0.05, remaining_time / (num_restarts + 10))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        num_restarts += 1
    
    # Fallback if no labels found (should not happen)
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
