"""Phylogenetic topology recovery via Neighbor Joining with NumPy acceleration."""
import numpy as np
import math
import time

# EVOLVE-BLOCK-START
def _compute_jc_distances(sequences, labels):
    """
    Compute the Jukes-Cantor corrected distance matrix.
    """
    n = len(labels)
    # Map labels to indices for easier vectorization
    # sequences is a dict: {label: sequence_string}
    # We create a 2D numpy array of the sequences
    # Assuming all sequences are equal length
    seq_list = [sequences[label] for label in labels]
    # Convert characters to integers for faster comparison? 
    # Or just use string comparison?
    # NumPy doesn't handle strings well for element-wise != in a single vectorized op for char arrays easily without encoding.
    # Let's encode ACGT to 0,1,2,3.
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, len(seq_list[0])), dtype=np.uint8)
    for i, s in enumerate(seq_list):
        for j, c in enumerate(s):
            mat[i, j] = enc[c]
    
    # Compute pairwise differences
    # mat[i] == mat[j] element-wise
    # We can do this in chunks or use broadcasting if memory allows.
    # n is up to 32. L is up to 2000.
    # 32*32*2000 = 2 million bytes. Very small.
    # (n, L) vs (n, 1, L) -> (n, n, L)
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.int8)
    hamming = np.sum(diff, axis=2)
    
    # Calculate proportion of differences p
    L = len(seq_list[0])
    p = hamming / float(L)
    
    # Jukes-Cantor correction: d = -0.75 * ln(1 - 4p/3)
    # Clamp 1 - 4p/3 to avoid log(0) or log(negative)
    # p can be up to 1.0. 1 - 4/3 is negative.
    # Max valid p for JC is 0.75.
    arg = 1.0 - (4.0 / 3.0) * p
    arg = np.maximum(arg, 1e-10) # Clamp to small positive number
    
    d = -0.75 * np.log(arg)
    
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    
    return d

def _neighbor_joining_numpy(dist):
    """
    Perform Neighbor Joining on a distance matrix.
    Returns a list of tuples (node1, node2, branch_len1, branch_len2) representing joins,
    and a mapping from internal nodes to their children/structure.
    Actually, to generate Newick, it's easier to track the tree structure directly.
    """
    n = dist.shape[0]
    if n < 3:
        return None # Should not happen for n>=8
    
    # Nodes are represented by their current indices.
    # We maintain a list of "active" nodes.
    # We also maintain the current distance matrix.
    # To avoid modifying the original dist, we work on a copy.
    D = dist.copy()
    
    # active_nodes: list of indices that are currently "alive" in the algorithm
    # The indices refer to the positions in the D matrix.
    # However, as we join nodes, we add new rows/cols or just remove old ones?
    # Standard NJ implementation often keeps a fixed size matrix and marks nodes as inactive,
    # or shrinks the matrix. Shrinking is cleaner for indexing but requires copying.
    # Given n <= 32, copying a 32x32 matrix 30 times is trivial.
    
    active = list(range(n))
    tree_nodes = {} # Maps internal node index to (child1, child2, len1, len2) or original taxon
    
    # For original taxa, tree_nodes[i] = None (indicating leaf)
    # For new nodes, tree_nodes[new_idx] = (node_a, node_b, len_a, len_b)
    
    # We will assign new node IDs as n, n+1, ...
    next_node_id = n
    # To make it easy, let's just append to the distance matrix and active list.
    # But D is a square matrix. If we append, we need to expand it.
    # Let's use a dictionary-based distance lookup or just a fixed-size matrix with "inactive" flag.
    # Fixed size matrix is faster.
    
    # Let's use the fixed size matrix approach with an 'active' boolean mask or list.
    # D is n x n initially.
    # We will add new rows/cols by pre-allocating a max size?
    # Max size is 2*n - 3? No, we add 1 node and remove 2. Net -1.
    # So the matrix shrinks.
    
    # Let's stick to the shrinking matrix approach for clarity and correctness.
    # D is current distance matrix.
    # active is list of indices in D.
    
    while len(active) > 3:
        m = len(active)
        # Extract submatrix of active nodes
        # D is (current_size, current_size). active are indices into it.
        # It's actually easier if we keep D as the full current matrix and active as 0..m-1 indices?
        # No, let's re-index.
        # Let's maintain D as the full matrix of size k x k, where k is current number of nodes.
        # And we don't need 'active' list if we just shrink D and keep track of what each node represents.
        
        # Let's restart the logic for shrinking matrix:
        # 1. Start with D (n x n).
        # 2. Loop while n > 3:
        #    a. Compute r_i = sum(D[i, :]) for i in 0..n-1
        #    b. Compute Q[i, j] = (n-2)D[i,j] - r_i - r_j
        #    c. Find min Q (i, j)
        #    d. Compute branch lengths
        #    e. Create new node
        #    f. Update D: Remove rows/cols i, j. Add row/col for new node.
        #       D_new[k, new] = 0.5 * (D[i, k] + D[j, k] - D[i, j])
        
        pass # Placeholder

    # This shrinking approach is efficient for small n.
    
    # Let's implement it properly.
    
    # Re-initialize
    n = dist.shape[0]
    D = dist.copy()
    
    # node_info: list of (type, data)
    # type: 'leaf' (data is label), 'internal' (data is (child1_idx, child2_idx, len1, len2))
    # But indices change when we shrink.
    # To avoid index confusion, let's use a fixed size matrix and track which slots are active.
    
    max_nodes = n + n - 3 # Safe upper bound
    D_full = np.zeros((max_nodes, max_nodes), dtype=float)
    D_full[:n, :n] = dist
    
    # active_indices: list of indices in D_full that are currently active
    active_indices = list(range(n))
    
    # node_desc: maps index in D_full to description
    # For i < n: leaf i
    # For i >= n: internal node
    node_desc = {}
    for i in range(n):
        node_desc[i] = ('leaf', i)
    
    next_idx = n
    
    while len(active_indices) > 3:
        m = len(active_indices)
        # Get current distance submatrix
        # Using advanced indexing
        indices = np.array(active_indices)
        D_sub = D_full[np.ix_(indices, indices)]
        
        # Compute r
        r = np.sum(D_sub, axis=1)
        
        # Compute Q
        # Q[i,j] = (m-2)D[i,j] - r[i] - r[j]
        # We can vectorize this.
        # R_col = r[:, None]
        # R_row = r[None, :]
        Q = (m - 2) * D_sub - r[:, None] - r[None, :]
        
        # Find min
        # We need to find i, j in active_indices that minimizes Q[i,j]
        # Flatten Q, find min, unflatten to get row/col in D_sub
        min_flat_idx = np.argmin(Q)
        i_sub = int(min_flat_idx // m)
        j_sub = int(min_flat_idx % m)
        
        if i_sub == j_sub:
            # Should not happen if D is valid metric
            # Force a break or error?
            continue 
            
        # Map back to full indices
        i_full = active_indices[i_sub]
        j_full = active_indices[j_sub]
        
        d_ij = D_full[i_full, j_full]
        
        # Calculate branch lengths
        # len_i = 0.5 * d_ij + (1/(2(m-2))) * (r[i] - r[j])
        # Using r from D_sub
        r_i = r[i_sub]
        r_j = r[j_sub]
        
        len_i = 0.5 * d_ij + (r_i - r_j) / (2.0 * (m - 2))
        len_j = d_ij - len_i
        
        # Ensure non-negative lengths? NJ can produce negative lengths.
        # Usually we clamp to 0 for Newick output.
        if len_i < 0: len_i = 0.0
        if len_j < 0: len_j = 0.0
        
        # Create new node
        new_full = next_idx
        next_idx += 1
        
        node_desc[new_full] = ('internal', i_full, j_full, len_i, len_j)
        
        # Update active_indices: remove i_full, j_full, add new_full
        # Remove i_full and j_full
        # It's safer to build a new list
        new_active = []
        for idx in active_indices:
            if idx != i_full and idx != j_full:
                new_active.append(idx)
        new_active.append(new_full)
        active_indices = new_active
        
        # Update distances for new_full to all other active nodes (excluding i_full, j_full)
        # D_new[new_full, k] = 0.5 * (D[i_full, k] + D[j_full, k] - D[i_full, j_full])
        for k in new_active:
            if k == new_full:
                continue
            D_full[new_full, k] = 0.5 * (D_full[i_full, k] + D_full[j_full, k] - d_ij)
            D_full[k, new_full] = D_full[new_full, k]
            
        # Set diagonal to 0 (optional, but good practice)
        D_full[new_full, new_full] = 0.0

    # Now we have 3 active nodes left.
    # Connect them in a triangle.
    # The remaining 3 nodes form a star.
    # We need to connect them with a central internal node?
    # Or just output them as ((A, B), C)?
    # Standard NJ terminates with 3 nodes. The final topology is a 3-taxon tree.
    # We can just return the structure.
    # To make a valid Newick tree for >3 taxa, the joins are already recorded in node_desc.
    # The last 3 nodes are roots of subtrees.
    # We need to connect them.
    # Let's pick one of them as the "root" of the final unrooted tree representation?
    # Actually, unrooted tree with 3 leaves is just a star.
    # But our node_desc records binary joins.
    # The final 3 nodes are the "tips" of the binary tree construction history.
    # We need to create one more internal node to connect these 3?
    # No, if we have 3 nodes, they are already connected via previous joins?
    # No, the algorithm stops when 3 nodes remain. These 3 nodes represent clusters of the original taxa.
    # We need to connect these 3 clusters.
    # Let's create a final internal node connecting two of them, and then connect the third to that?
    # Or just output the three subtrees?
    # A tree with 3 leaves is ((0,1),2).
    # Let's assume the last 3 active indices are a, b, c.
    # We can create a final join between a and b, then join c to the result.
    # Or simpler: Just return the 3 subtrees and let the Newick printer handle it?
    # Newick requires a single tree.
    # Let's do a final join to make it binary.
    
    final_active = active_indices
    if len(final_active) == 3:
        a, b, c = final_active
        # Join a and b
        d_ab = D_full[a, b]
        # Branch lengths for final join are less critical for topology, but let's compute
        # r_a, r_b, r_c from the 3x3 submatrix?
        # For the last step, standard NJ just connects them.
        # Let's just create a new node joining a and b.
        new_full = next_idx
        next_idx += 1
        node_desc[new_full] = ('internal', a, b, d_ab/2, d_ab/2)
        
        # Now join c to new_full
        new_full2 = next_idx
        next_idx += 1
        d_cn = D_full[c, a] # Approximate, or use specific formula?
        # Actually, D_full[c, new_full] is not set.
        # Let's just set it to 0 for now, or calculate properly.
        # For topology, the structure matters.
        # Let's just connect them arbitrarily with length 0 if we don't calculate carefully.
        # Better: Calculate r for the 3 nodes.
        D_3 = D_full[np.ix_([a,b,c], [a,b,c])]
        r_a = D_3[0,1] + D_3[0,2]
        r_b = D_3[1,0] + D_3[1,2]
        r_c = D_3[2,0] + D_3[2,1]
        
        # Join a, b
        d_ab = D_3[0,1]
        len_a = 0.5*d_ab + (r_a - r_b)/4.0
        len_b = d_ab - len_a
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        node_desc[new_full] = ('internal', a, b, len_a, len_b)
        
        # Join c, new_full
        # Distance c to new_full: 0.5*(D[c,a] + D[c,b] - D[a,b])
        d_cn = 0.5 * (D_full[c, a] + D_full[c, b] - d_ab)
        
        # Branch lengths for c and new_full
        # r_c (in 2-node system?) No, this is getting complicated.
        # Let's just use d_cn/2 for both.
        len_c = d_cn / 2.0
        len_n = d_cn / 2.0
        if len_c < 0: len_c = 0.0
        if len_n < 0: len_n = 0.0
        
        node_desc[new_full2] = ('internal', c, new_full, len_c, len_n)
        
        root_idx = new_full2
    else:
        # Should not happen
        root_idx = final_active[0]

    return node_desc, root_idx

def _newick_from_desc(node_desc, root_idx, labels):
    """
    Recursively build Newick string.
    """
    def build(idx):
        info = node_desc[idx]
        if info[0] == 'leaf':
            # Original label
            # info[1] is the index in the original labels list
            return f"{labels[info[1]]}"
        else:
            # info[0] == 'internal'
            # info[1]=child1, info[2]=child2, info[3]=len1, info[4]=len2
            c1, c2, l1, l2 = info[1], info[2], info[3], info[4]
            s1 = build(c1)
            s2 = build(c2)
            # Format lengths
            # Use 6 decimal places for precision
            return f"({s1}:{l1:.6f},{s2}:{l2:.6f})"
    
    return build(root_idx) + ";"

def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree from sequences.
    """
    start_time = time.time()
    deadline = start_time + 260.0 # Safety margin
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    
    if len(labels) < 2:
        return "t0;"
    
    # 1. Compute Distances
    dist = _compute_jc_distances(sequences, labels)
    
    # Check time
    if time.time() > deadline:
        # Return a star tree as fallback
        return "(" + ",".join(labels) + ");"
        
    # 2. Neighbor Joining
    node_desc, root_idx = _neighbor_joining_numpy(dist)
    
    # Check time
    if time.time() > deadline:
        # If NJ finished but time is up, try to build Newick quickly
        pass
    else:
        # 3. Build Newick
        newick = _newick_from_desc(node_desc, root_idx, labels)
        return newick

    # Fallback if something went wrong or time ran out
    return "(" + ",".join(labels) + ");"
# EVOLVE-BLOCK-END
