<<<<<<< SEARCH
def reconstruct(sequences):
    start_time = time.time()
    time_limit = 260.0  # 270s budget, leave 10s safety margin
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Convert sequences to integer matrix for fast comparison
    # A:0, C:1, G:2, T:3
    base_to_int = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    L = len(next(iter(sequences.values())))
    
    # Use vectorized conversion
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='U1')
    # Map characters to integers using vectorized approach
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for base, val in base_to_int.items():
        seq_mat[seq_chars == base] = val
    
    # Compute pairwise Hamming distances using vectorized operations
    diff = (seq_mat[:, None, :] != seq_mat[None, :, :]).astype(np.int8)
    hamming = diff.sum(axis=2)
    
    # Normalize to p
    p = hamming.astype(np.float64) / L
    
    # Jukes-Cantor Correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p >= 0.75 (which gives log of <= 0)
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    # Run Neighbor Joining
    newick = _nj_tree(dist_mat, labels)
    
    # Check time
    if time.time() - start_time > time_limit:
        return newick
    
    # For small n, try quartet-based refinement
    if n_labels <= 10:
        newick = _refine_with_quartets(dist_mat, labels, newick, time_limit, start_time)
    
    return newick
=======
def reconstruct(sequences):
    start_time = time.time()
    time_limit = 255.0  # 270s budget, leave 15s safety margin
    
    labels = sorted(sequences.keys(), key=lambda x: int(x[1:]))
    n_labels = len(labels)
    
    if n_labels < 2:
        return labels[0] + ";" if labels else ";"
    if n_labels == 2:
        return f"({labels[0]},{labels[1]})();"
    if n_labels == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}));"
    
    # Convert sequences to integer matrix for fast comparison
    # A:0, C:1, G:2, T:3
    L = len(next(iter(sequences.values())))
    
    # Use vectorized conversion with numpy
    seq_chars = np.array([list(sequences[label]) for label in labels], dtype='U1')
    # Map characters to integers using vectorized approach
    seq_mat = np.zeros((n_labels, L), dtype=np.int8)
    for base, val in [('A', 0), ('C', 1), ('G', 2), ('T', 3)]:
        seq_mat[seq_chars == base] = val
    
    # Compute pairwise Hamming distances efficiently
    # For each pair (i,j), count positions where seq_mat[i] != seq_mat[j]
    # Use broadcasting but in a memory-efficient way for large n
    if n_labels <= 50:
        # Direct computation for moderate n
        hamming = np.zeros((n_labels, n_labels), dtype=np.int32)
        for i in range(n_labels):
            for j in range(i + 1, n_labels):
                hamming[i, j] = np.sum(seq_mat[i] != seq_mat[j])
                hamming[j, i] = hamming[i, j]
    else:
        # For large n, use a more memory-efficient approach
        # Process in chunks to avoid creating huge intermediate arrays
        hamming = np.zeros((n_labels, n_labels), dtype=np.int32)
        chunk_size = max(1, 1000000 // n_labels)  # ~1M elements per chunk
        for i in range(n_labels):
            for j_start in range(i + 1, n_labels, chunk_size):
                j_end = min(j_start + chunk_size, n_labels)
                # Compare seq_mat[i] with seq_mat[j_start:j_end]
                diff = (seq_mat[i][None, :] != seq_mat[j_start:j_end]).astype(np.int32)
                hamming[i, j_start:j_end] = diff.sum(axis=1)
                hamming[j_start:j_end, i] = hamming[i, j_start:j_end]
    
    # Normalize to p
    p = hamming.astype(np.float64) / L
    
    # Jukes-Cantor Correction
    # d = -0.75 * ln(1 - 4/3 * p)
    # Handle p >= 0.75 (which gives log of <= 0)
    p_clamped = np.clip(p, 0.0, 0.7499999)
    dist_mat = -0.75 * np.log(1.0 - (4.0/3.0) * p_clamped)
    
    # Ensure symmetry and zero diagonal
    dist_mat = 0.5 * (dist_mat + dist_mat.T)
    np.fill_diagonal(dist_mat, 0.0)
    
    # Choose method based on n
    if n_labels <= 7:
        # Exhaustive search for small n
        newick = _exhaustive_quartet_search(dist_mat, labels)
    else:
        # NJ for larger n
        newick = _nj_tree(dist_mat, labels)
        
        # Check time before attempting refinement
        if time.time() - start_time < time_limit - 30:
            if n_labels <= 12:
                # Try NNI local search
                newick = _nni_local_search(dist_mat, labels, newick, max_iterations=20)
    
    return newick
>>>>>>> REPLACE
