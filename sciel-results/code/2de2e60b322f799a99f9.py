"""Phylogenetic topology recovery via Neighbor-Joining."""
import time
import numpy as np

_DEADLINE_FRACTION = 0.92
_start_time = None


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using numpy vectorization."""
    n = len(labels)
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[enc[c] for c in s] for s in (sequences[lab] for lab in labels)], dtype=np.uint8)
    n_sites = seqs.shape[1]
    
    # Pairwise differences
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    pmat = diff.sum(axis=2).astype(np.float64) / n_sites
    
    pmat = np.clip(pmat, 0, 0.7499)
    dmat = -0.75 * np.log(1.0 - 4.0 * pmat / 3.0)
    np.fill_diagonal(dmat, 0.0)
    return dmat


def _neighbor_joining(dmat, labels):
    """
    Perform Neighbor-Joining to reconstruct a tree.
    Returns a Newick string.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # active_nodes: list of indices currently in the distance matrix
    # We maintain a list of "node names" which are either original labels or new node IDs
    # dmat is a 2D list of lists for easy modification
    
    # Initialize
    active_nodes = list(range(n))
    current_dmat = dmat.tolist()
    node_names = list(labels)
    
    # We will build the tree by recording the splits
    # tree_structure: dict mapping node_id -> (child1, child2, len1, len2)
    # For leaves, node_id is the label.
    # For internal nodes, we assign new unique IDs.
    
    tree_nodes = {}
    next_node_id = n
    
    # Main loop
    while len(active_nodes) > 2:
        m = len(active_nodes)
        if m < 3:
            break
            
        # Compute row sums r_i
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += current_dmat[active_nodes[i]][active_nodes[j]]
                    
        # Compute Q matrix
        min_q = float('inf')
        i_idx = -1
        j_idx = -1
        
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                q = (m - 2) * current_dmat[active_nodes[i]][active_nodes[j]] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_idx = i
                    j_idx = j
                    
        if i_idx == -1:
            # Should not happen
            break
            
        i_node = active_nodes[i_idx]
        j_node = active_nodes[j_idx]
        
        # Calculate branch lengths
        d_ij = current_dmat[i_node][j_node]
        li = 0.5 * d_ij + (r[i_idx] - r[j_idx]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[i_idx] - r[j_idx]) / (2 * (m - 2))
        
        # Ensure non-negative lengths for valid tree representation (though NJ can produce negative)
        # For topology, sign doesn't matter, but for Newick, negative lengths are weird.
        # We'll keep them as is, or clamp to 0 if they are very small negative?
        # Standard NJ allows negative internal branches.
        
        # Create new node
        w_id = next_node_id
        next_node_id += 1
        w_name = f"w{w_id}"
        
        # Record tree structure
        # We store the children and their branch lengths
        # Note: li is length from i to w, lj from j to w
        tree_nodes[w_id] = (i_node, j_node, li, lj)
        
        # Update distance matrix
        # D(w, k) = 0.5 * (D(i, k) + D(j, k) - D(i, j))
        new_row = [0.0] * len(active_nodes)
        for k_idx, k_node in enumerate(active_nodes):
            if k_node == i_node or k_node == j_node:
                new_row[k_idx] = 0.0
            else:
                new_row[k_idx] = 0.5 * (current_dmat[i_node][k_node] + current_dmat[j_node][k_node] - d_ij)
        
        # We need to insert this new row/col into current_dmat?
        # It's easier to just build a new list of active nodes and a new dmat?
        # Or update in place?
        # Let's just create a new active_nodes list and a new dmat structure for the next iteration.
        # Since n is small (<=32), O(n^3) with list copies is fine.
        
        new_active_nodes = []
        for node in active_nodes:
            if node == i_node or node == j_node:
                continue
            new_active_nodes.append(node)
        new_active_nodes.append(w_id)
        
        # Construct new dmat for next iteration
        # We need distances between all pairs in new_active_nodes
        # We can extract from current_dmat and add the new row
        # But current_dmat is indexed by original node IDs (0..n-1 and w_ids)
        # We need a mapping from node_id to index in current_dmat?
        # Actually, current_dmat is a full n x n matrix (plus added nodes if we were doing it in place).
        # Since we are not adding to the global matrix, but rather shrinking the "active" set,
        # we can just keep the full matrix and use the active_nodes list to index it.
        # However, we added w_id which is not in the original dmat.
        # So we must expand the dmat or store distances differently.
        
        # Let's use a dictionary for distances to handle dynamic node addition easily.
        # dist[(u, v)]
        
        # Re-initializing distance storage as dict for simplicity and correctness with new nodes
        pass # We will refactor to use dict for distances

    # Refactoring to use Dict for distances to handle new nodes easily
    # Let's restart the NJ logic with Dicts
    
    # This function is being replaced entirely below
    return _nj_dict(dmat, labels)


def _nj_dict(dmat, labels):
    """Neighbor-Joining using dictionaries for flexibility."""
    n = len(labels)
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    # dist[(u, v)]
    dist = {}
    for i in range(n):
        for j in range(n):
            dist[(labels[i], labels[j])] = dmat[i, j]
            
    active_nodes = list(labels)
    tree_nodes = {}
    next_node_id = 0
    
    while len(active_nodes) > 2:
        m = len(active_nodes)
        
        # Compute r_i
        r = {}
        for i in active_nodes:
            r[i] = sum(dist[(i, k)] for k in active_nodes if k != i)
            
        # Compute Q and find min
        min_q = float('inf')
        i_node = None
        j_node = None
        
        for i in active_nodes:
            for j in active_nodes:
                if i == j:
                    continue
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_node = i
                    j_node = j
                    
        if i_node is None:
            break
            
        d_ij = dist[(i_node, j_node)]
        li = 0.5 * d_ij + (r[i_node] - r[j_node]) / (2 * (m - 2))
        lj = 0.5 * d_ij - (r[i_node] - r[j_node]) / (2 * (m - 2))
        
        w_name = f"w{next_node_id}"
        next_node_id += 1
        
        tree_nodes[w_name] = (i_node, j_node, li, lj)
        
        # Update distances for w
        for k in active_nodes:
            if k == i_node or k == j_node:
                continue
            d_wk = 0.5 * (dist[(i_node, k)] + dist[(j_node, k)] - d_ij)
            dist[(w_name, k)] = d_wk
            dist[(k, w_name)] = d_wk
            
        # Remove i, j and add w
        active_nodes.remove(i_node)
        active_nodes.remove(j_node)
        active_nodes.append(w_name)
        
    # Connect the last two
    if len(active_nodes) == 2:
        u, v = active_nodes
        d_uv = dist.get((u, v), 0.0)
        # Create a final root? No, just connect them.
        # We need to represent this in the tree structure.
        # We can add a dummy parent or just handle it in the builder.
        # Let's add a final node 'root'
        w_name = f"w{next_node_id}"
        next_node_id += 1
        tree_nodes[w_name] = (u, v, d_uv / 2.0, d_uv / 2.0)
        active_nodes = [w_name]
        
    # Build Newick string
    def build_newick(node):
        if node in labels:
            return node
        if node not in tree_nodes:
            return node # Fallback
            
        child1, child2, len1, len2 = tree_nodes[node]
        newick1 = build_newick(child1)
        newick2 = build_newick(child2)
        return f"({newick1}:{len1:.4f},{newick2}:{len2:.4f})"
        
    # The root is the last node created
    root = active_nodes[0]
    return build_newick(root)


def reconstruct(sequences):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Time check
    elapsed = time.time() - _start_time
    remaining = _DEADLINE_FRACTION * 270.0 - elapsed
    if remaining < 5.0:
        # Fallback: Star tree
        return "(" + ",".join(labels) + ")"
        
    # 1. Compute Distance Matrix
    dmat = _jc_distance_matrix(sequences, labels)
    
    # 2. Neighbor Joining
    tree = _nj_dict(dmat, labels)
    
    return tree
