"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Compute spectral initialization using the Fiedler vector (second eigenvector)
    # For max-cut, we want to maximize the cut, which relates to minimizing the
    # quadratic form x^T L x where L is the Laplacian. The Fiedler vector gives
    # a good initial partition.
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    # Use numpy's eigensolver for the Laplacian
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        # Fallback: no spectral init, use random restarts
        eigvecs = None

    best_labels = None
    best_cut = -1

    # Try multiple spectral initializations + local search
    num_restarts = 5
    if eigvecs is not None:
        # Use Fiedler vector and a few other eigenvectors as starting points
        # The Fiedler vector (index 1) is the best spectral init for max-cut
        candidate_indices = [1]
        # Add a few other low-frequency eigenvectors for diversity
        for k in range(2, min(6, n)):
            candidate_indices.append(k)
        
        for idx in candidate_indices:
            if time.time() > deadline - 5.0:
                break
            fiedler = eigvecs[:, idx]
            labels = (fiedler > 0).astype(np.int8)
            # Ensure both sides are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            
            labels, cut = _local_search(adj, labels, deadline)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
    else:
        # Fallback to random restarts
        for _ in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            labels = np.random.randint(0, 2, n).astype(np.int8)
            labels, cut = _local_search(adj, labels, deadline)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()

    # If no valid labels yet, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)

    return best_labels


def _local_search(adj, labels, deadline):
    """Run 1-flip and efficient 2-flip local search until convergence or deadline."""
    n = len(adj)
    
    # 1-flip local search
    for _ in range(20):
        if time.time() > deadline:
            break
        changed = False
        for v in range(n):
            if time.time() > deadline:
                break
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                changed = True
        if not changed:
            break

    # Efficient 2-flip (pairwise swap) local search
    # For pair (i, j) with different labels, gain = 
    #   (neighbors of i on j's side - neighbors of i on i's side) + 
    #   (neighbors of j on i's side - neighbors of j on j's side)
    # If adj[i,j] exists, it was already counted in both, so no double-count issue
    # Actually: gain = 2 * (adj[i,j] if they're on different sides... let me derive carefully)
    #
    # Let S_i = set of neighbors of i on same side as i
    # Let O_i = set of neighbors of i on opposite side from i
    # Currently, edges from i to O_i cross the cut.
    # After flipping i, edges from i to S_i cross the cut.
    # Net change for i alone: |S_i| - |O_i|
    # But if we flip both i and j (who are on different sides):
    #   i moves to j's side, j moves to i's side
    #   For i: edges to (old same side minus j) become crossing, edges to (old other side plus j) stop crossing
    #   This is complex; let's just compute the gain directly with precomputed sums
    #
    # Precompute for each vertex: count of neighbors on each side
    # For a pair (i, j) with labels[i] != labels[j]:
    #   Let a = labels[i], b = labels[j] (a != b)
    #   Currently crossing edges involving i: adj[i, labels != a].sum()
    #   Currently crossing edges involving j: adj[j, labels != b].sum()
    #   But edge (i,j) is counted in both if it exists
    #   After flip: i has label b, j has label a
    #   New crossing edges involving i: adj[i, labels != b].sum() (but j now has label a, so adj[i,j] contributes to same-side)
    #   New crossing edges involving j: adj[j, labels != a].sum()
    #
    # Gain = (new_crossing_i + new_crossing_j) - (old_crossing_i + old_crossing_j)
    # where we need to be careful about the (i,j) edge being counted correctly
    #
    # Simpler: just compute the cut size change directly but efficiently
    # For n=100, even O(n^2) per sweep with vectorized ops is fast enough
    
    for _ in range(10):
        if time.time() > deadline:
            break
        changed = False
        # Vectorized 2-flip: compute gain for all pairs at once
        # For each pair (i,j) with different labels, compute the gain
        # This is O(n^2) which is 10000 for n=100, very fast
        
        # Precompute: for each vertex v, count neighbors on each side
        # side0_count[v] = number of neighbors of v with label 0
        # side1_count[v] = number of neighbors of v with label 1
        side0_count = adj @ (labels == 0).astype(np.int8)
        side1_count = adj @ (labels == 1).astype(np.int8)
        
        # For vertex v with label a:
        #   neighbors on same side = sidea_count[v]
        #   neighbors on other side = side(1-a)_count[v]
        #   gain from flipping v alone = sidea_count[v] - side(1-a)_count[v]
        #
        # For pair (i, j) with labels[i] != labels[j]:
        #   Let a = labels[i], b = 1-a = labels[j]
        #   Old crossing from i: side1_count[i] if a=0 else side0_count[i]
        #   Old crossing from j: side0_count[j] if b=1 else side1_count[j]
        #   = side(1-a)_count[i] + side(1-b)_count[j] = side(1-a)_count[i] + side(a)_count[j]
        #   
        #   After flip: i has label b, j has label a
        #   New crossing from i: side(1-b)_count[i] adjusted for j's new label
        #   This is getting complicated. Let me just use the direct approach but vectorize the gain computation.
        
        # Alternative efficient approach: for each pair (i,j) with different labels,
        # compute gain as:
        #   gain = (adj[i, labels==labels[j]].sum() - adj[i, labels==labels[i]].sum()) +
        #          (adj[j, labels==labels[i]].sum() - adj[j, labels==labels[j]].sum())
        #   But we need to account for the edge between i and j
        #   If adj[i,j] == 1, it's in both "same side" counts before and "other side" counts after
        #   Actually: before, i and j are on different sides, so adj[i,j] is a crossing edge
        #   After, i and j are on different sides still (swapped), so adj[i,j] is still a crossing edge
        #   So the edge (i,j) doesn't change its crossing status
        #   
        #   gain_i = adj[i, labels==labels[j]].sum() - adj[i, labels==labels[i]].sum()
        #   gain_j = adj[j, labels==labels[i]].sum() - adj[j, labels==labels[j]].sum()
        #   total_gain = gain_i + gain_j
        
        # Let's compute this efficiently
        # For all i: gain_i_if_flipped = adj[i, labels==0].sum() - adj[i, labels==1].sum() if labels[i]==1
        #                                     adj[i, labels==1].sum() - adj[i, labels==0].sum() if labels[i]==0
        # = (side1_count[i] - side0_count[i]) if labels[i]==1 else (side0_count[i] - side1_count[i])
        # = (2*labels[i] - 1) * (side1_count[i] - side0_count[i])
        
        flip_gain = (2 * labels.astype(np.int8) - 1) * (side1_count - side0_count)
        
        # For pair (i, j) with different labels, total gain = flip_gain[i] + flip_gain[j]
        # We want to find pairs with positive total gain
        # Create a gain matrix where gain_matrix[i,j] = flip_gain[i] + flip_gain[j] if labels[i] != labels[j]
        # Only consider i < j for uniqueness
        
        # Vectorized: create gain matrix
        gain_matrix = flip_gain[:, None] + flip_gain[None, :]
        
        # Mask: only pairs with different labels and i < j
        diff_mask = (labels[:, None] != labels[None, :])
        upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        valid_mask = diff_mask & upper_mask
        
        # Find the best pair
        masked_gains = np.where(valid_mask, gain_matrix, -np.inf)
        max_gain = masked_gains.max()
        
        if max_gain > 0:
            # Find the pair
            idx = np.unravel_index(masked_gains.argmax(), (n, n))
            i, j = idx
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            changed = True
        else:
            break
    
    return labels, _cut_size(adj, labels)


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
