"""
Reconstructs phylogenetic topology using Neighbor Joining (NJ) on Jukes-Cantor corrected distances.
Optimized with NumPy for vectorized distance calculation and efficient matrix operations.
"""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START

def _jc_distance_vectorized(hamming_matrix, n_sites):
    """
    Calculate Jukes-Cantor corrected distance matrix from Hamming distances.
    """
    # p = hamming / n_sites
    # d = -0.75 * log(1 - 4/3 * p)
    # Handle p=0 and p>=0.75
    p = hamming_matrix / n_sites
    # Clip p to avoid log of negative or zero
    # 1 - 4/3 * p must be > 0 => p < 0.75
    # If p >= 0.75, distance is effectively infinite/saturated. We cap it.
    p_safe = np.clip(p, 0.0, 0.749)
    d = -0.75 * np.log(1.0 - (4.0/3.0) * p_safe)
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    return d

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    start_time = time.time()
    timeout = 270.0
    safety_margin = 10.0 

    # 1. Pre-process sequences for fast vectorized distance calculation
    # Map characters to integers
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Build sequence matrix (N, L)
    # Assuming all sequences are same length
    seq_len = len(sequences[labels[0]])
    seq_matrix = np.zeros((n, seq_len), dtype=np.int8)
    
    for i, label in enumerate(labels):
        s = sequences[label]
        # Convert string to array of mapped integers
        # Using a loop is okay for N<=32, L<=2000. 
        # Or use vectorized mapping if possible, but simple loop is fine here.
        for j, c in enumerate(s):
            seq_matrix[i, j] = char_map.get(c, 0) # Default 0 if unexpected char
            
    # 2. Calculate Hamming distances vectorized
    # Broadcasting: (N, 1, L) vs (1, N, L) -> (N, N, L)
    # This creates a large intermediate array. N=32, L=2000 -> 32*32*2000 = 2M bytes. Very small.
    hamming = np.sum(seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :], axis=2)
    
    # 3. Calculate JC distances
    dist = _jc_distance_vectorized(hamming, seq_len)
    
    # Active nodes: initially 0..n-1
    active = list(range(n))
    # Newick representation for each active node
    newick = {i: labels[i] for i in range(n)}
    
    # Distance matrix for active nodes
    # We will maintain a dictionary for dynamic updates as in the original, 
    # but initialize from the vectorized result.
    d_matrix = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                d_matrix[(i, j)] = dist[i][j]

    next_id = n
    
    while len(active) > 3:
        if time.time() - start_time > timeout - safety_margin:
            break
            
        n_active = len(active)
        
        # Calculate sum of distances for each active node
        r = {i: 0.0 for i in active}
        for i in active:
            for j in active:
                if i != j:
                    r[i] += d_matrix[(i, j)]
        
        # Find the pair (i, j) that minimizes the NJ criterion
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
            break
            
        i, j = best_i, best_j
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        newick[new_id] = f"({newick[i]},{newick[j]})"
        
        # Calculate distances from new node to all other active nodes
        d_ij = d_matrix[(i, j)]
        
        for x in active:
            if x == i or x == j:
                continue
            d_ix = d_matrix[(i, x)]
            d_jx = d_matrix[(j, x)]
            d_nx = 0.5 * (d_ix + d_jx - d_ij)
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
        return f"({newick[a]},{newick[b]},{newick[c]});"
    else:
        # Fallback: complete the joining using simple distance min if we ran out of time
        while len(active) > 3:
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
             newick[new_id] = f"({newick[best_i]},{newick[best_j]})"
             
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
        return f"({newick[a]},{newick[b]},{newick[c]});"

# EVOLVE-BLOCK-END
