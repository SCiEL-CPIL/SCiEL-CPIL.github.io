"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    # Fast local search with incremental gain updates and time guard.
    # We use a gain array to track the benefit of flipping each vertex.
    # Gain[v] = (number of neighbors in opposite set) - (number of neighbors in same set)
    # If we flip v, the cut changes by Gain[v].
    # When we flip v, for each neighbor u, Gain[u] changes by -2*Gain[u]'s contribution...
    # Actually: if v flips from 0 to 1:
    #   For each neighbor u of v:
    #     If u was in 0 (same as v before), now u is in 0 and v is in 1 -> edge (u,v) now crosses.
    #       Previously it didn't cross (both 0). So cut increases by 1 for this edge.
    #       But Gain[u] was counting: same-side neighbors vs opposite-side neighbors.
    #       Before flip: v was same-side as u (both 0). After: v is opposite.
    #       So Gain[u] increases by 2 (one less same, one more opposite).
    #     If u was in 1 (opposite to v before), now u is in 1 and v is in 1 -> edge (u,v) no longer crosses.
    #       Previously it crossed. So cut decreases by 1.
    #       Gain[u] decreases by 2.
    
    # More carefully: Gain[v] = sum_{u in N(v)} (1 if label[u] != label[v] else -1)
    # = (# opposite neighbors) - (# same neighbors)
    # = 2*(# opposite neighbors) - degree(v)
    
    # If we flip v:
    #   New Gain[v] = -Old Gain[v]
    #   For each neighbor u of v:
    #     If label[u] == old label[v] (i.e., same side as v before flip):
    #       After flip, label[u] != new label[v], so this edge now contributes +1 to Gain[u] instead of -1.
    #       Gain[u] changes by +2.
    #     If label[u] != old label[v] (i.e., opposite side to v before flip):
    #       After flip, label[u] == new label[v], so this edge now contributes -1 to Gain[u] instead of +1.
    #       Gain[u] changes by -2.
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adj = np.array(adjacency, dtype=np.int8)
    # Use time guard
    deadline = time.time() + 140  # safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple random restarts to escape local optima
    n_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Initialize labels
        if restart == 0:
            labels = np.arange(n) % 2
        else:
            labels = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Compute initial gains
        # Gain[v] = 2 * (# opposite neighbors) - degree(v)
        # # opposite neighbors = sum of adj[v, :] where labels != labels[v]
        # Let's compute: for each v, count neighbors with same label
        # same_count[v] = sum over u of adj[v,u] * (1 if labels[u] == labels[v] else 0)
        # We can do: same_count = (adj * (labels[None, :] == labels[:, None])).sum(axis=1)
        # But that's O(n^2) memory. For n=100, it's fine.
        
        # Alternative: 
        # labels_one_hot = (labels[:, None] == labels[None, :])  # n x n boolean
        # same_count = (adj * labels_one_hot).sum(axis=1)
        # opposite_count = adj.sum(axis=1) - same_count
        # gains = 2 * opposite_count - adj.sum(axis=1)
        
        degrees = adj.sum(axis=1).astype(np.int32)
        
        def compute_gains(labels):
            # Same-side neighbor count for each vertex
            # For vertex v, same-side neighbors are u where adj[v,u]=1 and labels[u]==labels[v]
            # We can compute this as: for each v, sum adj[v,:] * (labels == labels[v])
            # Vectorized: create matrix M where M[v,u] = 1 if labels[u] == labels[v] else 0
            # Then same_count = (adj * M).sum(axis=1)
            # M[v,u] = (labels[u] == labels[v]) = (labels[None, :] == labels[:, None])[v, u]
            # So M = (labels[None, :] == labels[:, None])
            M = (labels[None, :] == labels[:, None])
            same_count = (adj * M).sum(axis=1).astype(np.int32)
            opposite_count = degrees - same_count
            gains = 2 * opposite_count - degrees
            return gains
        
        gains = compute_gains(labels)
        
        # Compute initial cut
        # Cut = sum of edges crossing the partition
        # = 0.5 * sum over all edges (adj[i,j]) * (1 if labels[i] != labels[j] else 0)
        # = 0.5 * (adj * (labels[None, :] != labels[:, None])).sum()
        def compute_cut(labels):
            crossing = (labels[None, :] != labels[:, None])
            return 0.5 * (adj * crossing).sum()
        
        current_cut = compute_cut(labels)
        
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Find the vertex with the maximum positive gain
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 0:
                break
            
            # Flip the vertex
            v = max_gain_idx
            old_label = labels[v]
            labels[v] = 1 - old_label
            
            # Update gains for v and its neighbors
            # New Gain[v] = -Old Gain[v]
            gains[v] = -gains[v]
            
            # For each neighbor u of v:
            # If labels[u] == old_label (same as v before flip): Gain[u] += 2
            # If labels[u] != old_label (opposite to v before flip): Gain[u] -= 2
            neighbors = np.where(adj[v] == 1)[0]
            for u in neighbors:
                if labels[u] == old_label:
                    gains[u] += 2
                else:
                    gains[u] -= 2
            
            current_cut += max_gain
            improved = True
        
        # Check if this is the best so far
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Early termination if we've hit a high-quality solution
        # For a 100-node graph, maximum cut is at most 0.5 * total_edges
        # Total edges = 0.5 * adj.sum()
        total_edges = 0.5 * adj.sum()
        if best_cut >= 0.75 * total_edges:
            break
    
    if best_labels is None:
        best_labels = labels
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
