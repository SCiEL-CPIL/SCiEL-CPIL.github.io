"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    adj = np.array(adjacency, dtype=np.int8)
    n = len(adj)
    deadline = time.time() + 145.0  # safety margin before 150s kill

    # Compute spectral initialization using the Fiedler vector (second eigenvector)
    deg = adj.sum(axis=1)
    L = np.diag(deg) - adj
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        fiedler = eigvecs[:, 1]
        labels = (fiedler > 0).astype(np.int8)
    except Exception:
        # Fallback to alternating start
        labels = np.arange(n) % 2

    best_labels = labels.copy()
    best_cut = _cut_size(adj, labels)

    # 1-flip local search (vectorized gain computation)
    best_labels, best_cut = _local_search_1flip(adj, labels, best_labels, best_cut, deadline)

    # 2-flip (pairwise swap) local search with efficient gain computation
    best_labels, best_cut = _local_search_2flip(adj, best_labels, best_cut, deadline)

    # Additional rounds of alternating 1-flip and 2-flip to polish
    for round_idx in range(4):
        if time.time() > deadline:
            break
        best_labels, best_cut = _local_search_1flip(adj, best_labels, best_cut, deadline)
        if time.time() > deadline:
            break
        best_labels, best_cut = _local_search_2flip(adj, best_labels, best_cut, deadline)

    return best_labels


def _local_search_1flip(adj, labels, best_labels, best_cut, deadline):
    """Run 1-flip local search until no improvement or deadline."""
    labels = labels.copy()
    n = len(adj)
    for _ in range(10):
        changed = False
        for v in range(n):
            if time.time() > deadline:
                return best_labels, best_cut
            same = adj[v][labels == labels[v]].sum()
            other = adj[v][labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
                changed = True
        if not changed:
            break
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    return best_labels, best_cut


def _local_search_2flip(adj, best_labels, best_cut, deadline):
    """Run 2-flip local search with efficient gain computation."""
    labels = best_labels.copy()
    n = len(adj)
    for _ in range(5):
        changed = False
        # Precompute for efficiency
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    continue
                # Compute gain of flipping both i and j
                # Current cut contribution from i's edges: edges from i to opposite side
                # After flipping both, we need to compute the new cut
                # Let's use a direct but efficient computation
                # Gain = 2 * (edges from i to same side as j) + 2 * (edges from j to same side as i) - 2 * (edge i-j if exists)
                # More precisely:
                # Let S_i = set of neighbors of i on same side as i
                # Let S_j = set of neighbors of j on same side as j
                # When we flip both, edges from i to S_j become cut, edges from j to S_i become cut
                # But edges between i and j: if they exist, they were cut before (different sides), after flip they're still cut (different sides)
                # Wait, if i and j have different labels, after flipping both they still have different labels
                # So edge i-j remains cut
                # The change comes from:
                # For vertex i: edges to vertices on i's side become non-cut, edges to vertices on j's side become cut
                # For vertex j: edges to vertices on j's side become non-cut, edges to vertices on i's side become cut
                # Let A = adj[i][labels == labels[i]] (neighbors of i on i's side)
                # Let B = adj[j][labels == labels[j]] (neighbors of j on j's side)
                # Let C = adj[i][labels == labels[j]] (neighbors of i on j's side)
                # Let D = adj[j][labels == labels[i]] (neighbors of j on i's side)
                # Current cut contribution involving i or j:
                #   i contributes: |C| (edges from i to j's side)
                #   j contributes: |D| (edges from j to i's side)
                #   But if edge i-j exists, it's counted in both C and D
                # After flip:
                #   i's new side is j's old side, so i contributes: |A| (edges from i to i's old side, now opposite)
                #   j's new side is i's old side, so j contributes: |B| (edges from j to j's old side, now opposite)
                #   Edge i-j: still cut (both flipped)
                # New cut contribution: |A| + |B|, but edge i-j counted in both if exists
                # Actually let me just compute directly for correctness
                old_i_cut = adj[i][labels != labels[i]].sum()
                old_j_cut = adj[j][labels != labels[j]].sum()
                # Edge between i and j
                edge_ij = adj[i, j]
                # Current total cut from i and j (avoid double counting edge_ij)
                old_total = old_i_cut + old_j_cut - edge_ij
                
                labels[i] = 1 - labels[i]
                labels[j] = 1 - labels[j]
                
                new_i_cut = adj[i][labels != labels[i]].sum()
                new_j_cut = adj[j][labels != labels[j]].sum()
                new_total = new_i_cut + new_j_cut - edge_ij
                
                if new_total > old_total:
                    changed = True
                else:
                    labels[i] = 1 - labels[i]
                    labels[j] = 1 - labels[j]
        if not changed:
            break
        cut = _cut_size(adj, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    return best_labels, best_cut


def _cut_size(adj, labels):
    """Compute the number of edges crossing the cut."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int(np.sum(adj * diff) // 2)
# EVOLVE-BLOCK-END
