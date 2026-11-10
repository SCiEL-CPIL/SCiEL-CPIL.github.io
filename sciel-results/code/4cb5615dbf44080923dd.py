"""
Reconstructs phylogenetic topology using Neighbor Joining (NJ) on Jukes-Cantor corrected distances.
This method is standard and highly effective for additive distance matrices generated under the JC model.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance(a, b):
    """Calculate Jukes-Cantor corrected distance."""
    n = len(a)
    if n == 0:
        return 0.0
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix_vectorized(sequences, labels):
    """Compute full JC distance matrix using NumPy for speed."""
    n = len(labels)
    # Convert sequences to numpy arrays of character codes
    # A=0, C=1, G=2, T=3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build matrix of shape (n, L)
    L = len(sequences[labels[0]])
    seq_matrix = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            seq_matrix[i, j] = char_map[c]
    
    # Compute pairwise differences
    # For each pair (i, j), count positions where seq_matrix[i] != seq_matrix[j]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Compute initial distance matrix using vectorized NumPy approach
    dist_np = _compute_distance_matrix_vectorized(sequences, labels)
    dist = dist_np.tolist()

    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Newick representation for each active node
    newick = [labels[i] for i in range(n)]
    
    # We will maintain a distance matrix for active nodes.
    # Since we merge nodes, we need to update distances.
    # To keep it simple and fast for n<=32, we can just keep a dictionary
    # or a 2D array that we resize. 
    # However, standard NJ implementation often just keeps the full matrix
    # and ignores inactive nodes, or updates a smaller matrix.
    # Given n is small, let's use a dictionary for clarity and dynamic updates,
    # or better, just maintain the list of active indices and a full matrix,
    # updating the new node's distances.
    
    # Let's use the dictionary approach for the dynamic matrix to avoid resizing complexity,
    # but pre-fill it for initial nodes.
    
    d_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix[(i, j)] = dist[i][j]

    node_newick = {i: labels[i] for i in range(n)}
    next_id = n
    
    start_time = time.time()
    timeout = 270.0
    safety_margin = 5.0

    while len(active) > 3:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        n_active = len(active)
        
        # Calculate sum of distances for each active node
        # r[i] = sum_{j != i} d(i, j)
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += d_matrix[(i, j)]
        
        # Find the pair (i, j) that minimizes the NJ criterion
        # Q(i, j) = (n_active - 2) * d(i, j) - r[i] - r[j]
        best_val = float('inf')
        best_i = None
        best_j = None
        
        for u_idx in range(n_active):
            i = active[u_idx]
            for v_idx in range(u_idx + 1, n_active):
                j = active[v_idx]
                d_ij = d_matrix[(i, j)]
                criterion = (n_active - 2) * d_ij - r[i] - r[j]
                if criterion < best_val:
                    best_val = criterion
                    best_i = i
                    best_j = j
        
        if best_i is None:
            # Fallback if something went wrong
            break
            
        i, j = best_i, best_j
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        node_newick[new_id] = f"({node_newick[i]},{node_newick[j]})"
        
        # Calculate distances from new node to all other active nodes
        # d(new, x) = 0.5 * (d(i, x) + d(j, x) - d(i, j))
        d_ij = d_matrix[(i, j)]
        
        for x in active:
            if x == i or x == j:
                continue
            d_ix = d_matrix[(i, x)]
            d_jx = d_matrix[(j, x)]
            d_nx = 0.5 * (d_ix + d_jx - d_ij)
            # Clamp to non-negative just in case of numerical issues, though NJ usually handles it
            if d_nx < 0:
                d_nx = 0.0
            d_matrix[(new_id, x)] = d_nx
            d_matrix[(x, new_id)] = d_nx
            
        # Remove i and j from active
        active.remove(i)
        active.remove(j)
        active.append(new_id)

    # Assemble the final tree
    if len(active) == 3:
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]});"
    else:
        # Fallback: complete the joining using simple distance min if we ran out of time
        while len(active) > 3:
             # Find closest pair
             best_dist = float('inf')
             best_i = active[0]
             best_j = active[1]
             for u_idx in range(len(active)):
                 i = active[u_idx]
                 for v_idx in range(u_idx + 1, len(active)):
                     j = active[v_idx]
                     d = d_matrix[(i, j)]
                     if d < best_dist:
                         best_dist = d
                         best_i = i
                         best_j = j
             
             new_id = next_id
             next_id += 1
             node_newick[new_id] = f"({node_newick[best_i]},{node_newick[best_j]})"
             
             d_ij = best_dist
             for x in active:
                 if x == best_i or x == best_j:
                     continue
                 d_nx = 0.5 * (d_matrix[(best_i, x)] + d_matrix[(best_j, x)] - d_ij)
                 if d_nx < 0:
                     d_nx = 0.0
                 d_matrix[(new_id, x)] = d_nx
                 d_matrix[(x, new_id)] = d_nx
                 
             active.remove(best_i)
             active.remove(best_j)
             active.append(new_id)
             
        a, b, c = active
        return f"({node_newick[a]},{node_newick[b]},{node_newick[c]});"

# EVOLVE-BLOCK-END
