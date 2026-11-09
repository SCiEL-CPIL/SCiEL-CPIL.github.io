import time
import numpy as np

DEADLINE_SECONDS = 270.0
SAFETY_MARGIN = 10.0

def _jc_distance_matrix_from_array(seq_array):
    """Compute pairwise JC distance matrix from a numeric sequence array."""
    n, L = seq_array.shape
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if L == 0:
        return np.zeros((n, n), dtype=np.float64)
        
    diff = seq_array[:, None, :] != seq_array[None, :, :]
    hamming = diff.sum(axis=2, dtype=np.int32)
    p = hamming.astype(np.float64) / L
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist

def _neighbor_joining_fast(dist_matrix):
    """
    Fast Neighbor-Joining. Returns a list of splits (frozensets of indices).
    """
    n = dist_matrix.shape[0]
    if n <= 2:
        return []
    
    active_ids = list(range(n))
    node_taxa = {i: frozenset([i]) for i in range(n)}
    D = dist_matrix.copy()
    next_id = n
    splits = []
    all_taxa = frozenset(range(n))
    
    m = n
    while m > 3:
        r = D.sum(axis=1)
        Q = (m - 2) * D - r[:, None] - r[None, :]
        np.fill_diagonal(Q, np.inf)
        
        i, j = np.unravel_index(np.argmin(Q), Q.shape)
        if i == j:
            break
        if i > j:
            i, j = j, i
            
        d_ij = D[i, j]
        
        clade_taxa = node_taxa[active_ids[i]] | node_taxa[active_ids[j]]
        
        if 1 < len(clade_taxa) < n - 1:
            splits.append(clade_taxa)
            
        new_id = next_id
        next_id += 1
        node_taxa[new_id] = clade_taxa
        
        other_indices = [k for k in range(m) if k != i and k != j]
        
        new_dists = np.zeros(len(other_indices), dtype=np.float64)
        for idx, k in enumerate(other_indices):
            new_dists[idx] = 0.5 * (D[i, k] + D[j, k] - d_ij)
            
        m_new = len(other_indices) + 1
        D_new = np.zeros((m_new, m_new), dtype=np.float64)
        
        for a_idx, a_old in enumerate(other_indices):
            for b_idx, b_old in enumerate(other_indices):
                D_new[a_idx, b_idx] = D[a_old, b_old]
                
        for a_idx, a_old in enumerate(other_indices):
            D_new[a_idx, m_new - 1] = new_dists[a_idx]
            D_new[m_new - 1, a_idx] = new_dists[a_idx]
            
        new_active = [active_ids[k] for k in other_indices] + [new_id]
        
        active_ids = new_active
        D = D_new
        m = m_new
        
    return splits

def _build_consensus_tree(splits, n, labels):
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    valid_splits = []
    for s in splits:
        if 1 < len(s) < n - 1:
            valid_splits.append(s)
            
    if not valid_splits:
        return "(" + ",".join(labels) + ");"
        
    def build(taxa_set):
        if len(taxa_set) == 1:
            idx = next(iter(taxa_set))
            return labels[idx]
            
        partition_found = None
        min_size = n + 1
        for s in valid_splits:
            A = s & taxa_set
            B = taxa_set - A
            if 0 < len(A) < len(taxa_set) and 0 < len(B) < len(taxa_set):
                if len(A) < min_size:
                    min_size = len(A)
                    partition_found = (A, B)
                    
        if partition_found:
            A, B = partition_found
            left = build(A)
            right = build(B)
            return "(" + left + "," + right + ")"
        else:
            return "(" + ",".join(labels[i] for i in sorted(taxa_set)) + ")"
            
    all_taxa = frozenset(range(n))
    return build(all_taxa)

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS - SAFETY_MARGIN
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
        
    L = len(sequences[labels[0]])
    
    lut = np.zeros(256, dtype=np.int8)
    lut[ord('A')] = 0
    lut[ord('C')] = 1
    lut[ord('G')] = 2
    lut[ord('T')] = 3
    
    seq_array = np.empty((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        b = sequences[lab].encode('ascii')
        arr = np.frombuffer(b, dtype=np.uint8)
        seq_array[i] = lut[arr]
        
    # Original Distance Matrix
    dist_matrix = _jc_distance_matrix_from_array(seq_array)
    orig_splits = _neighbor_joining_fast(dist_matrix)
    
    splits_counter = {}
    for s in orig_splits:
        splits_counter[s] = splits_counter.get(s, 0) + 1
        
    # Bootstrap
    # Determine number of bootstraps
    # n=32 is slower.
    if n <= 16:
        num_bootstraps = 100
    elif n <= 24:
        num_bootstraps = 60
    else:
        num_bootstraps = 30
        
    rng = np.random.default_rng(42)
    
    for b in range(num_bootstraps):
        if time.time() > deadline - 5.0:
            break
            
        indices = rng.integers(0, L, size=L)
        boot_seq = seq_array[:, indices]
        
        dist_b = _jc_distance_matrix_from_array(boot_seq)
        boot_splits = _neighbor_joining_fast(dist_b)
        
        for s in boot_splits:
            splits_counter[s] = splits_counter.get(s, 0) + 1
            
    threshold = (num_bootstraps + 1) / 2.0
    final_splits = [s for s, count in splits_counter.items() if count > threshold]
    
    if not final_splits:
        final_splits = orig_splits
        
    newick = _build_consensus_tree(final_splits, n, labels)
    
    return newick
