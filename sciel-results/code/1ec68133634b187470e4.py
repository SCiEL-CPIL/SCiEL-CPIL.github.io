"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized Neighbor Joining on Jukes-Cantor corrected distances with deadline guard.'''
import math
import time
import numpy as np

_DEADLINE = 240.0  # wall-clock budget; leave margin for evaluator overhead


def _jc_distances(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using NumPy."""
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    # Build alignment matrix: (n, L) with encoded bases
    # Encode A=0, C=1, G=2, T=3
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        mat[i] = np.array([base_map[c] for c in s], dtype=np.int8)
    
    # Compute pairwise Hamming distances vectorized
    # diff[i,j] = number of positions where mat[i] != mat[j]
    # Use broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This can be memory-heavy for n=32, L=2000: 32*32*2000 = 2M ints = 8MB, fine
    diff = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.count_nonzero(mat[i] != mat[j])
            diff[i, j] = d
            diff[j, i] = d
    
    # Convert to Jukes-Cantor corrected distances
    p = diff / L
    # Clip to avoid log of non-positive
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _nj_vectorized(D, labels):
    """
    Vectorized Neighbor Joining.
    D: (n, n) distance matrix
    labels: list of n label strings
    Returns: Newick string
    """
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0] + ";"
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Work with indices; map back to labels at the end
    # active: list of currently active node indices (into original label space or internal)
    # We'll track nodes by their "tree string" and a distance matrix over active nodes
    
    # Initialize
    active = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    dist = D.copy()  # (n, n) full matrix, we'll maintain it over active subset
    
    while len(active) > 3:
        m = len(active)
        if time.time() > _DEADLINE - 5.0:
            # Return star tree as fallback
            return "(" + ",".join(node_trees[a] for a in active) + ");"
        
        # Get submatrix for active nodes
        Dsub = dist[np.ix_(active, active)]
        
        # Compute row sums (total)
        total = Dsub.sum(axis=1)
        
        # NJ criterion: (m-2)*D[i,j] - total[i] - total[j]
        # We want the minimum
        crit = (m - 2) * Dsub - total[:, None] - total[None, :]
        # Zero out diagonal and upper triangle to only consider i < j
        np.fill_diagonal(crit, np.inf)
        # Set upper triangle to inf (we'll take min over lower triangle)
        iu = np.triu_indices(m, k=1)
        crit[iu] = np.inf
        
        # Find min
        flat_idx = np.argmin(crit)
        i, j = np.unravel_index(flat_idx, (m, j))
        # Wait, that's wrong. Let me fix.
        flat_idx = np.argmin(crit)
        i, j = np.unravel_index(flat_idx, (m, m))
        
        ai = active[i]
        aj = active[j]
        
        # Create new node
        new_id = max(active) + 1
        joined = f"({node_trees[ai]},{node_trees[aj]})"
        
        # Compute new distances to all other active nodes
        rest = [x for x in active if x not in (ai, aj)]
        rest_indices = [active.index(x) for x in rest]
        
        # New distance: 0.5 * (D[ai,x] + D[aj,x] - D[ai,aj])
        new_dists = np.empty(len(rest))
        for k, r in enumerate(rest):
            new_dists[k] = 0.5 * (dist[ai, r] + dist[aj, r] - dist[ai, aj])
        
        # Update distance matrix: add new row/col
        new_dist = np.zeros((len(active) + 1, len(active) + 1), dtype=np.float64)
        # Copy existing
        for idx_a, a in enumerate(active):
            for idx_b, b in enumerate(active):
                new_dist[idx_a, idx_b] = dist[a, b]
        
        # Set new distances
        new_pos = len(active)
        for k, r in enumerate(rest):
            r_pos = active.index(r)
            new_dist[new_pos, r_pos] = new_dists[k]
            new_dist[r_pos, new_pos] = new_dists[k]
        
        dist = new_dist
        node_trees[new_id] = joined
        active = rest + [new_id]
    
    # Final 3 nodes
    if len(active) == 3:
        return f"({node_trees[active[0]]},{node_trees[active[1]]},{node_trees[active[2]]});"
    elif len(active) == 2:
        return f"({node_trees[active[0]]},{node_trees[active[1]]});"
    else:
        return node_trees[active[0]] + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    D = _jc_distances(sequences, labels)
    return _nj_vectorized(D, labels)
# EVOLVE-BLOCK-END
