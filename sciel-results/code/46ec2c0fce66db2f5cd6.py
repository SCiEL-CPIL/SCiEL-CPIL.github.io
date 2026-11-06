"""Rewritten program for Maximum Cut using SDP-inspired spectral initialization + efficient local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 130.0  # safety margin before 150s kill

    # Compute spectral initialization using the Laplacian eigenvectors
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj

    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except Exception:
        eigvecs = None

    best_labels = None
    best_cut = -1

    # Collect initial partition candidates
    candidates = []

    if eigvecs is not None:
        # Direct eigenvector sign initializations (indices 1..6)
        for idx in range(1, min(7, n)):
            fiedler = eigvecs[:, idx]
            labels = (fiedler > 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            candidates.append(labels.copy())

        # SDP random hyperplane initialization: project random hyperplanes
        # onto the subspace spanned by the first k Laplacian eigenvectors
        k_sub = min(6, n - 1)
        V = eigvecs[:, 1:1 + k_sub]  # n x k_sub matrix

        for _ in range(8):
            if time.time() > deadline - 30.0:
                break
            # Random hyperplane in the eigenvector subspace
            r = np.random.randn(k_sub)
            proj = V @ r  # n-vector
            labels = (proj > 0).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels[n // 2] = 1 - labels[n // 2]
            candidates.append(labels)

    # Also add a couple of random initializations for diversity
    for _ in range(3):
        labels = np.random.randint(0, 2, n).astype(np.int8)
        if labels.sum() == 0 or labels.sum() == n:
            labels[n // 2] = 1 - labels[n // 2]
        candidates.append(labels)

    # Refine each candidate with local search (SA-based)
    for labels in candidates:
        if time.time() > deadline - 15.0:
            break
        labels, cut = _local_search(adj, labels, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()

    # Fallback
    if best_labels is None:
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)

    return best_labels


def _local_search(adj, labels, deadline):
    """Run batched 1-flip local search with simulated annealing to escape local minima."""
    n = len(labels)
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated annealing parameters
    T = 2.0  # initial temperature
    cooling = 0.98
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Compute flip gains
        side0 = (labels == 0).astype(np.int8)
        side1 = (labels == 1).astype(np.int8)
        s0_counts = adj @ side0
        s1_counts = adj @ side1
        flip_gain_1 = np.where(labels == 0, s0_counts - s1_counts, s1_counts - s0_counts)
        
        # Batch non-conflicting flips (greedy, sorted by gain)
        positive_gains = flip_gain_1 > 0
        if positive_gains.any():
            candidate_indices = np.where(positive_gains)[0]
            sorted_idx = candidate_indices[np.argsort(flip_gain_1[candidate_indices])[::-1]]
            
            selected = []
            used = np.zeros(n, dtype=bool)
            for idx in sorted_idx:
                if used[idx]:
                    continue
                neighbors = np.where(adj[idx] == 1)[0]
                if not np.any(used[neighbors]):
                    selected.append(idx)
                    used[idx] = True
                    used[neighbors] = True
            
            if selected:
                # Compute cut change for this batch
                old_cut = current_cut
                for idx in selected:
                    labels[idx] = 1 - labels[idx]
                new_cut = _cut_size(adj, labels)
                delta = new_cut - old_cut
                
                # Accept if improvement, or with probability exp(-delta/T) if worsening
                if delta >= 0 or np.random.random() < np.exp(-delta / T):
                    current_cut = new_cut
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                else:
                    # Reject: undo
                    for idx in selected:
                        labels[idx] = 1 - labels[idx]
        
        # Also try single 2-flip moves occasionally
        if np.random.random() < 0.3:
            # Try a random 2-flip
            i, j = np.random.choice(n, 2, replace=False)
            if labels[i] != labels[j]:  # only consider different sides
                old_cut = current_cut
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                new_cut = _cut_size(adj, labels)
                delta = new_cut - old_cut
                
                if delta >= 0 or np.random.random() < np.exp(-delta / T):
                    current_cut = new_cut
                    if current_cut > best_cut:
                        best_cut = current_cut
                        best_labels = labels.copy()
                else:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
        
        T *= cooling
    
    # Final greedy polish
    while time.time() < deadline - 2.0:
        side0 = (labels == 0).astype(np.int8)
        side1 = (labels == 1).astype(np.int8)
        s0_counts = adj @ side0
        s1_counts = adj @ side1
        flip_gain_1 = np.where(labels == 0, s0_counts - s1_counts, s1_counts - s0_counts)
        
        max_gain = flip_gain_1.max()
        if max_gain <= 0:
            break
        
        idx = np.argmax(flip_gain_1)
        labels[idx] = 1 - labels[idx]
        current_cut = _cut_size(adj, labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
    
    return best_labels, best_cut


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
