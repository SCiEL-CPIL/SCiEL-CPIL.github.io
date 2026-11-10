"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Vectorized neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987).'''
import numpy as np


def _jc_distance_matrix(seq_array):
    """Compute pairwise JC-corrected distances from a 2D numpy array of character codes."""
    n = seq_array.shape[0]
    # Count mismatches: for each pair, count positions where values differ
    # seq_array is (n, L) with integer codes
    # Use broadcasting: seq_array[:, None, :] != seq_array[None, :, :] gives (n, n, L)
    # But this uses too much memory for large L. Instead, use a smarter approach.
    
    # For each nucleotide position, we can use one-hot encoding and matrix multiplication
    # Or simply loop over positions (L up to 2000, n up to 32)
    
    # More memory-efficient: accumulate mismatch counts
    mismatch_counts = np.zeros((n, n), dtype=np.int32)
    for pos in range(seq_array.shape[1]):
        col = seq_array[:, pos]  # shape (n,)
        # For each nucleotide, count how many taxa have it at this position
        # Then compute pairwise mismatches using: total pairs - sum over nucleotides of (count_k * count_k)
        # Actually simpler: for each pair (i,j), they mismatch if col[i] != col[j]
        # Use: mismatch = 1 - (col[i] == col[j])
        # Vectorized: (col[:, None] != col[None, :]) gives (n, n) boolean
        mismatch_counts += (col[:, None] != col[None, :]).astype(np.int32)
    
    # Convert to proportions
    L = seq_array.shape[1]
    p = mismatch_counts / L
    
    # JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 by capping
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    # Symmetrize (should already be symmetric, but just in case)
    d = (d + d.T) / 2.0
    
    return d


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    # Convert sequences to numpy array
    seq_array = np.array([[ord(c) - ord('A') if c in 'ACGT' else 0 for c in sequences[label]] for label in labels])
    
    # Compute initial distance matrix
    D = _jc_distance_matrix(seq_array)
    
    # Active nodes: indices into the original taxa
    active = list(range(n_taxa))
    trees = {i: labels[i] for i in range(n_taxa)}
    
    counter = 0
    while len(active) > 3:
        n = len(active)
        # Get current distance submatrix
        D_sub = D[np.ix_(active, active)]
        
        # Compute row sums (total distances)
        row_sums = D_sub.sum(axis=1)
        
        # Compute Q matrix: Q[i,j] = (n-2)*D[i,j] - row_sums[i] - row_sums[j]
        Q = (n - 2) * D_sub - row_sums[:, None] - row_sums[None, :]
        
        # Find the pair with minimum Q (excluding diagonal)
        np.fill_diagonal(Q, np.inf)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx
        
        # Ensure i < j for consistent naming
        if i > j:
            i, j = j, i
        
        a, b = active[i], active[j]
        
        # Create new node
        new_idx = -1 - counter
        counter += 1
        
        joined = f"({trees[a]},{trees[b]})"
        trees[new_idx] = joined
        
        # Compute distances from new node to all other active nodes
        rest_indices = [k for k in range(n) if k != i and k != j]
        for k in rest_indices:
            x = active[k]
            d_new_x = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[new_idx, x] = d_new_x
            D[x, new_idx] = d_new_x
        
        # Update active list
        rest = [active[k] for k in rest_indices]
        active = rest + [new_idx]
    
    # Build final Newick string
    return "(" + ",".join(trees[x] for x in active) + ");"
# EVOLVE-BLOCK-END
