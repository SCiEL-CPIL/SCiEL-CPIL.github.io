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
    """Run interleaved 1-flip and 2-flip local search until convergence or deadline."""
    n = len(labels)
    
    while time.time() < deadline:
        changed = False
        
        # 1-flip pass
        # Compute gain for each vertex
        side0 = (labels == 0).astype(np.int8)
        side1 = (labels == 1).astype(np.int8)
        s0_counts = adj @ side0
        s1_counts = adj @ side1
        
        # Gain if flip v:
        # If label 0: gain = s0[v] - s1[v]
        # If label 1: gain = s1[v] - s0[v]
        flip_gain_1 = np.where(labels == 0, s0_counts - s1_counts, s1_counts - s0_counts)
        
        # Find the best 1-flip
        max_gain_1 = flip_gain_1.max()
        if max_gain_1 > 0:
            idx = np.argmax(flip_gain_1)
            labels[idx] = 1 - labels[idx]
            changed = True
        else:
            # No 1-flip improvement, try 2-flip
            # Compute gain for pairs
            # Gain(i, j) = flip_gain[i] + flip_gain[j] + 2 * adj[i, j]
            # Note: flip_gain here is based on CURRENT labels.
            # We need to ensure labels[i] != labels[j].
            
            # Construct gain matrix
            fg = flip_gain_1
            gain_mat = fg[:, None] + fg[None, :]
            # Add 2 * adj term
            gain_mat += 2 * adj
            
            # Mask: i < j and labels[i] != labels[j]
            diff_mask = (labels[:, None] != labels[None, :])
            upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            valid_mask = diff_mask & upper_mask
            
            masked_gains = np.where(valid_mask, gain_mat, -np.inf)
            max_gain_2 = masked_gains.max()
            
            if max_gain_2 > 0:
                idx = np.unravel_index(masked_gains.argmax(), (n, n))
                i, j = idx
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                changed = True
            else:
                # No 1-flip or 2-flip improvement
                break
    
    return labels, _cut_size(adj, labels)


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
