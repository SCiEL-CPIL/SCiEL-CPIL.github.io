<<<<<<< SEARCH
def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    
    # Build alignment matrix: n x L with codes 0-3
    align = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        align[i] = [code_map[c] for c in s]
    
    # Compute pairwise differences efficiently
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(align[i] != align[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist
=======
def _jc_distance_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(sequences[labels[0]])
    
    # Build alignment matrix: n x L with codes 0-3
    # Use a lookup table for speed
    trans = np.zeros(256, dtype=np.int8)
    trans[ord('A')] = 0
    trans[ord('C')] = 1
    trans[ord('G')] = 2
    trans[ord('T')] = 3
    
    align = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        # Vectorized conversion
        align[i] = trans[np.frombuffer(s.encode('ascii'), dtype=np.uint8)]
    
    # Compute pairwise differences efficiently using broadcasting
    diff_matrix = (align[:, np.newaxis, :] != align[np.newaxis, :, :]).sum(axis=2)
    
    p = diff_matrix / L
    # Handle p >= 0.75 (log domain issue)
    mask = p < 0.75
    dist = np.zeros((n, n), dtype=np.float64)
    dist[mask] = -0.75 * math.log(1.0 - 4.0 * p[mask] / 3.0)
    dist[~mask] = 3.0 # Large distance for high divergence
    
    # Zero out diagonal
    np.fill_diagonal(dist, 0.0)
    return dist
>>>>>>> REPLACE

