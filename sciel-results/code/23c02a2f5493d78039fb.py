import math
import time
import numpy as np
import re

_GLOBAL_START = None
_DEADLINE = 200.0  # Total budget for all instances

def _elapsed():
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
    return time.time() - _GLOBAL_START

def _time_exceeded():
    return _elapsed() > _DEADLINE

def _is_valid_newick(newick, labels):
    if not newick:
        return False
    found_labels = re.findall(r't\d+', newick)
    return set(found_labels) == set(labels)

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.zeros((n, L), dtype=np.int8)
    for i, l in enumerate(labels):
        s = sequences[l]
        for j, c in enumerate(s):
            seqs[i, j] = enc[c]
            
    # Pairwise differences
    diff = (seqs[:, None, :] != seqs[None, :, :]).sum(axis=2)
    p = diff.astype(float) / L
    p_safe = np.minimum(p, 0.7499)
    d = -0.75 * np.log(1 - 4 * p_safe / 3)
    np.fill_diagonal(d, 0)
    return d

def _nj_tree(labels, dist):
    """
    Neighbor-Joining algorithm.
    Returns a Newick string.
    """
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
        
    # dist is a numpy array
    # We will maintain a list of active nodes and a distance matrix
    # Active nodes: 0..n-1 are leaves, n.. are internal
    
    # Initialize
    active = list(range(n))
    # d is the current distance matrix for active nodes
    # We need to map active node indices to their positions in the 'active' list
    # Actually, it's easier to keep a dictionary or just use the indices in the active list.
    
    # Let's use a dictionary for the tree structure
    # tree[i] = Newick string for node i
    tree = {i: labels[i] for i in range(n)}
    
    # Distance matrix D where D[i, j] is distance between active nodes i and j
    # We'll use a 2D array and update it, or just use the active list indices.
    # To make it efficient, let's keep the full dist matrix and a list of active indices.
    
    D = dist.copy()
    active_indices = list(range(n))
    
    next_id = n
    
    while len(active_indices) > 3:
        m = len(active_indices)
        
        # Calculate r vector
        # r[i] = sum_j D[i, j] for all active j
        # D is n x n, but we only care about active indices.
        # Let's extract the submatrix for active indices.
        idx = np.array(active_indices)
        D_sub = D[np.ix_(idx, idx)]
        
        r = D_sub.sum(axis=1)
        
        # Calculate Q matrix
        # Q[i, j] = (m - 2) * D[i, j] - r[i] - r[j]
        # We want to find the pair (i, j) that minimizes Q[i, j]
        
        # Efficient calculation of min Q
        # Q[i, j] = (m-2)*D[i,j] - r[i] - r[j]
        
        min_q = float('inf')
        best_i = -1
        best_j = -1
        
        for i in range(m):
            for j in range(i + 1, m):
                q_val = (m - 2) * D_sub[i, j] - r[i] - r[j]
                if q_val < min_q:
                    min_q = q_val
                    best_i = i
                    best_j = j
        
        # best_i and best_j are indices in the active_indices list
        node_a = active_indices[best_i]
        node_b = active_indices[best_j]
        
        # Calculate branch lengths
        # d_a = 0.5 * D[a, b] + 0.5 * (r[a] - r[b]) / (m - 2)
        # d_b = 0.5 * D[a, b] - 0.5 * (r[a] - r[b]) / (m - 2)
        # Note: r here corresponds to the active list indices.
        # r_a = r[best_i], r_b = r[best_j]
        # D_ab = D_sub[best_i, best_j]
        
        D_ab = D_sub[best_i, best_j]
        r_a = r[best_i]
        r_b = r[best_j]
        
        d_a = 0.5 * D_ab + 0.5 * (r_a - r_b) / (m - 2)
        d_b = 0.5 * D_ab - 0.5 * (r_a - r_b) / (m - 2)
        
        # Ensure non-negative (though NJ usually yields positive for valid trees)
        if d_a < 0: d_a = 0
        if d_b < 0: d_b = 0
        
        # Create new node
        new_node = next_id
        next_id += 1
        
        # Update tree structure
        # The new node connects to node_a and node_b
        # We need to store the branch lengths to build the Newick string later?
        # The task says branch lengths are optional. We can just build topology.
        # However, storing lengths might help if we wanted to do refinement, but NJ topology is the goal.
        # Let's just build the Newick string directly.
        
        # Newick for new node: (node_a, node_b)
        # We need to know the Newick string of node_a and node_b.
        # tree[node_a] and tree[node_b] hold the Newick strings.
        
        # We need to be careful: if node_a or node_b are internal nodes created previously,
        # their tree string is already built.
        
        tree[new_node] = f"({tree[node_a]},{tree[node_b]})"
        
        # Update distance matrix
        # For all other active nodes k:
        # D[new, k] = 0.5 * (D[a, k] + D[b, k] - D[a, b])
        
        # We need to update the global D matrix.
        # First, calculate new distances for new_node to all other active nodes (excluding a and b).
        # Then, replace a and b with new_node in active_indices.
        
        # Indices of a and b in active_indices are best_i and best_j.
        # Let's create a new active list.
        new_active = []
        for k, node in enumerate(active_indices):
            if node == node_a or node == node_b:
                continue
            new_active.append(node)
        new_active.append(new_node)
        
        # Update D matrix entries for new_node
        for k, node in enumerate(new_active):
            if node == new_node:
                continue
            # node is one of the remaining active nodes
            # D[new, node] = 0.5 * (D[a, node] + D[b, node] - D[a, b])
            D[new_node, node] = 0.5 * (D[node_a, node] + D[node_b, node] - D_ab)
            D[node, new_node] = D[new_node, node]
            
        # We don't strictly need to clear old entries, as we only look at active nodes.
        # But to be safe/clean, we could zero them out.
        
        active_indices = new_active
        
        if _time_exceeded():
            # If we run out of time, we should return the current best tree.
            # But we are in the middle of NJ. We can just break and build the tree from remaining nodes.
            break
    
    # Build final Newick string from remaining active nodes
    # There should be 3 nodes left.
    # The topology is a star among these 3.
    # (node1, node2, node3)
    
    # However, the tree dictionary contains the full Newick for each node.
    # The final tree is a Newick string that groups the remaining 3 nodes.
    # Since the remaining 3 nodes are connected in a star, and each node represents a clade,
    # the final Newick is (tree[n1], tree[n2], tree[n3])
    
    # Wait, if we broke out of the loop early, we might have more than 3 nodes?
    # No, the loop runs while len > 3. So we exit when len == 3.
    
    # Get the 3 remaining nodes
    remaining = active_indices
    
    # The final Newick string
    final_newick = "(" + ",".join([tree[node] for node in remaining]) + ");"
    
    return final_newick

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return "(" + ",".join(labels) + ");"
        
    # Calculate JC distance matrix
    dist = _jc_distance_matrix(sequences, labels)
    
    # Run Neighbor-Joining
    nj_newick = _nj_tree(labels, dist)
    
    # Validate NJ tree
    if not _is_valid_newick(nj_newick, labels):
        # Fallback to star tree
        return "(" + ",".join(labels) + ");"
        
    return nj_newick
