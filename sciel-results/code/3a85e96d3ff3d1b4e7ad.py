"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Neighbour joining on Jukes-Cantor corrected distances (Saitou and Nei 1987) with NumPy vectorization.'''
import math
import time
import numpy as np


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distances using NumPy."""
    n = len(labels)
    seq_len = len(sequences[labels[0]])
    
    # Build a matrix of sequences: n x seq_len with encoded nucleotides
    # A=0, C=1, G=2, T=3
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.empty((n, seq_len), dtype=np.int8)
    for i, label in enumerate(labels):
        s = sequences[label]
        for j in range(seq_len):
            mat[i, j] = enc[s[j]]
    
    # Compute pairwise differences: |mat[i] - mat[j]| > 0
    # Use broadcasting: (n, 1, seq_len) vs (1, n, seq_len)
    # diff = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    # diff is (n, n, seq_len)
    # For each pair (i,j), count positions where diff != 0
    
    # To save memory for larger n, we can do it in chunks or use a more efficient approach
    # For n up to 32, n*n*seq_len = 32*32*2000 = 2,048,000 which is fine
    
    diff = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    # Count mismatches: where diff != 0
    mismatches = (diff != 0).sum(axis=2)  # shape (n, n)
    
    # Convert to proportions
    p = mismatches.astype(np.float64) / seq_len
    
    # Apply JC correction: d = -0.75 * ln(1 - 4p/3)
    # Handle p >= 0.75 (log of non-positive)
    p_clipped = np.clip(p, 0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    
    # Set diagonal to 0
    np.fill_diagonal(dist, 0.0)
    
    return dist


def _nj_newick(dist, labels):
    """Perform Neighbor-Joining on a distance matrix, return Newick string."""
    n = len(labels)
    # Work with a copy of the distance matrix
    D = dist.copy()
    # Track which original labels are still active
    active = list(range(n))
    # Track the Newick subtree for each node
    subtrees = [labels[i] for i in range(n)]
    
    while len(active) > 3:
        m = len(active)
        # Compute row sums for active nodes
        # D is n x n, but we only care about active indices
        r = np.zeros(m)
        for i, ai in enumerate(active):
            for j, aj in enumerate(active):
                if i != j:
                    r[i] += D[ai, aj]
        
        # Find the pair (a, b) that minimizes (m-2)*D[a,b] - r[a] - r[b]
        best_val = float('inf')
        best_i = best_j = -1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * D[active[i], active[j]] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i = i
                    best_j = j
        
        ai = active[best_i]
        aj = active[best_j]
        
        # Create new node
        new_node = n + (n - len(active))  # unique index for the new internal node
        # We need to extend D if new_node >= n
        if new_node >= D.shape[0]:
            # Extend D to size (new_node+1, new_node+1)
            new_size = new_node + 1
            new_D = np.zeros((new_size, new_size), dtype=np.float64)
            new_D[:D.shape[0], :D.shape[0]] = D
            D = new_D
        
        # Update distances from new node to all other active nodes
        for k, ak in enumerate(active):
            if k != best_i and k != best_j:
                D[new_node, ak] = D[ak, new_node] = 0.5 * (D[ai, ak] + D[aj, ak] - D[ai, aj])
        
        # Create subtree
        subtree = f"({subtrees[ai]},{subtrees[aj]})"
        subtrees.append(subtree)
        
        # Remove ai and aj from active, add new_node
        new_active = []
        for k, ak in enumerate(active):
            if k != best_i and k != best_j:
                new_active.append(ak)
        new_active.append(new_node)
        active = new_active
    
    # Join the remaining 3 nodes
    if len(active) == 3:
        a, b, c = active
        return f"({subtrees[a]},{subtrees[b]},{subtrees[c]})"
    elif len(active) == 2:
        return f"({subtrees[active[0]]},{subtrees[active[1]]})"
    else:
        return subtrees[active[0]]


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Compute distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    newick = _nj_newick(dist, labels)
    
    return newick + ";"
# EVOLVE-BLOCK-END
