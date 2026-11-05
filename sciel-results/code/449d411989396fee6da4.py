"""Distance-based phylogenetic topology recovery: Jukes-Cantor + Neighbor-Joining."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected pairwise distance matrix (vectorized)."""
    n = len(labels)
    # Build a matrix of nucleotide positions for each taxon (n x L), encoded 0..3
    enc = np.full((n, 0), 0, dtype=np.int8)
    if len(sequences[labels[0]]) > 0:
        L = len(sequences[labels[0]])
        enc = np.empty((n, L), dtype=np.int8)
        for i, lab in enumerate(labels):
            s = sequences[lab]
            # map A,C,G,T -> 0,1,2,3
            arr = np.frombuffer(s.encode('ascii'), dtype=np.uint8)
            # A=65, C=67, G=71, T=84
            m = np.zeros(256, dtype=np.int8)
            m[65] = 0
            m[67] = 1
            m[71] = 2
            m[84] = 3
            enc[i] = m[arr]
        # pairwise hamming distance
        # diff matrix: n x n x L -> sum along L
        # do it in chunks to limit memory for large L
        diff_sum = np.zeros((n, n), dtype=np.int64)
        chunk = 1024
        for start in range(0, L, chunk):
            stop = min(start + chunk, L)
            block = enc[:, start:stop]  # n x c
            # for each pair (i,j) count differences
            # use broadcasting: n x n x c
            b = block[:, None, :]
            c = block[None, :, :]
            diff = (b != c).sum(axis=2)
            diff_sum += diff
        p = diff_sum / float(L)
        np.fill_diagonal(p, 0.0)
        # Jukes-Cantor correction: d = -3/4 * ln(1 - 4p/3)
        # guard against p >= 0.75 (log of non-positive)
        p_safe = np.clip(p, 0.0, 0.7499)
        D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_safe)
        # if p was >= 0.75, cap at a large finite value
        D[p >= 0.75] = 3.0
        np.fill_diagonal(D, 0.0)
        return D
    else:
        return np.zeros((n, n))


def _neighbor_joining(D):
    """Run Neighbor-Joining on distance matrix D (n x n). Returns Newick string."""
    n = D.shape[0]
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    if n == 3:
        # For 3 taxa, NJ collapses to a star; build a simple unrooted tree
        # Use the three pairwise distances to form a 3-leaf unrooted tree with branch lengths.
        # Standard 3-taxon NJ: internal node with three branches.
        a, b, c = 0, 1, 2
        d_ab = D[a, b]
        d_ac = D[a, c]
        d_bc = D[b, c]
        # branch lengths from the three-taxon solution
        # length from a to internal = (d_ab + d_ac - d_bc) / 2
        la = (d_ab + d_ac - d_bc) / 2.0
        lb = (d_ab + d_bc - d_ac) / 2.0
        lc = (d_ac + d_bc - d_ab) / 2.0
        # clamp negatives to 0 for validity
        la = max(la, 0.0)
        lb = max(lb, 0.0)
        lc = max(lc, 0.0)
        return f"(t0:{la:.6f},t1:{lb:.6f},t2:{lc:.6f});"

    # General NJ
    labels = list(range(n))
    # tree structure: each node stores its Newick subtree string
    tree = {i: f"t{i}" for i in range(n)}
    dist = D.copy()
    active = list(range(n))
    counter = n  # for new internal node ids

    while len(active) > 2:
        m = len(active)
        # row sums
        row_sums = np.array([dist[i, j] for i in active for j in active])
        row_sums = row_sums.reshape(m, m).sum(axis=1)
        # Q matrix: Q[i,j] = (m-2)*d[i,j] - row_sum[i] - row_sum[j]
        # find min
        best_val = float('inf')
        best_i = best_j = -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                q = (m - 2) * dist[i, j] - row_sums[ii] - row_sums[jj]
                if q < best_val:
                    best_val = q
                    best_i = ii
                    best_j = jj
        i = active[best_i]
        j = active[best_j]
        # branch lengths
        d_ij = dist[i, j]
        # omega = 1/(m-2)
        # len(i) = 0.5*d_ij + (row_sum[i] - row_sum[j])/(2*(m-2))
        len_i = 0.5 * d_ij + (row_sums[best_i] - row_sums[best_j]) / (2.0 * (m - 2))
        len_j = 0.5 * d_ij - (row_sums[best_i] - row_sums[best_j]) / (2.0 * (m - 2))
        len_i = max(len_i, 0.0)
        len_j = max(len_j, 0.0)
        # new node
        new_id = counter
        counter += 1
        new_tree = f"({tree[i]}:{len_i:.6f},{tree[j]}:{len_j:.6f})"
        # update distances to new node
        # d(new, k) = 0.5*(d[i,k] + d[j,k] - d[i,j])
        for k in active:
            if k == i or k == j:
                continue
            dist[new_id, k] = 0.5 * (dist[i, k] + dist[j, k] - dist[i, j])
            dist[k, new_id] = dist[new_id, k]
        dist[new_id, new_id] = 0.0
        # remove i and j
        active = [x for x in active if x != i and x != j]
        active.append(new_id)
        tree[new_id] = new_tree

    # final two nodes: connect with a branch; for unrooted topology, join them
    a, b = active
    d_ab = dist[a, b]
    # For the final two, we form a tree connecting their subtrees
    # branch length split
    half = d_ab / 2.0
    # We need to attach them with a connecting edge. Represent as (A:half,B:half)
    final_tree = f"({tree[a]}:{half:.6f},{tree[b]}:{half:.6f})"
    return final_tree + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 1:
        return "t0;"
    if n == 2:
        return "(t0,t1);"
    D = _jc_distance_matrix(sequences, labels)
    tree = _neighbor_joining(D)
    return tree
# EVOLVE-BLOCK-END
