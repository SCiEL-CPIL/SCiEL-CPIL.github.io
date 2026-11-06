import numpy as np
import time
import math

# EVOLVE-BLOCK-START

_DEADLINE = time.time() + 265.0

def _time_left():
    return _DEADLINE - time.time()

def _jc_distance(p):
    p = min(max(p, 0.0), 0.7499)
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_distance_matrix(sequences, labels):
    n = len(labels)
    seqs = [sequences[lb] for lb in labels]
    length = len(seqs[0])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arrs = []
    for s in seqs:
        arr = np.array([char_map[c] for c in s], dtype=np.int8)
        arrs.append(arr)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(arrs[i] != arrs[j])
            p = diff / length
            d = _jc_distance(p)
            mat[i, j] = d
            mat[j, i] = d
    return mat

def _nj_tree_struct(mat, labels):
    n = len(labels)
    if n <= 1:
        return {0: {'left': None, 'right': None, 'parent': None, 'label': labels[0], 'branch_len': 0.0}}, 0

    active = list(range(n))
    D = {}
    for i in range(n):
        for j in range(n):
            D[(i, j)] = mat[i, j]
            
    next_id = n
    nodes = {}
    for i in range(n):
        nodes[i] = {'left': None, 'right': None, 'parent': None, 'label': labels[i], 'branch_len': 0.0}
        
    while len(active) > 3:
        m = len(active)
        r = {}
        for i in active:
            r[i] = sum(D[(i, j)] for j in active if j != i)
            
        min_val = float('inf')
        min_i = min_j = None
        for i in active:
            for j in active:
                if i == j: continue
                q = (m - 2) * D[(i, j)] - r[i] - r[j]
                if q < min_val:
                    min_val = q
                    min_i = i
                    min_j = j
                    
        i, j = min_i, min_j
        a = i
        b = j
        
        d_ab = D[(a, b)]
        delta = (r[a] - r[b]) / (2 * (m - 2))
        la = 0.5 * d_ab + 0.5 * delta
        lb = 0.5 * d_ab - 0.5 * delta
        la = max(0.0, la)
        lb = max(0.0, lb)
        
        new_id = next_id
        next_id += 1
        
        nodes[new_id] = {'left': a, 'right': b, 'parent': None, 'label': None, 'branch_len': 0.0}
        nodes[a]['parent'] = new_id
        nodes[b]['parent'] = new_id
        nodes[a]['branch_len'] = la
        nodes[b]['branch_len'] = lb
        
        new_active = [x for x in active if x != a and x != b]
        new_active.append(new_id)
        
        for k in new_active:
            if k == new_id: continue
            D[(new_id, k)] = 0.5 * (D[(a, k)] + D[(b, k)] - d_ab)
            D[(k, new_id)] = D[(new_id, k)]
            
        active = new_active
        
    if len(active) == 3:
        a, b, c = active
        d_ab = D[(a, b)]
        d_bc = D[(b, c)]
        d_ac = D[(a, c)]
        
        la = (d_ab + d_ac - d_bc) / 2.0
        lb = (d_ab + d_bc - d_ac) / 2.0
        lc = (d_ac + d_bc - d_ab) / 2.0
        
        la = max(0.0, la)
        lb = max(0.0, lb)
        lc = max(0.0, lc)
        
        new_id = next_id
        next_id += 1
        
        nodes[new_id] = {'left': a, 'right': b, 'parent': None, 'label': None, 'branch_len': 0.0}
        nodes[a]['parent'] = new_id
        nodes[b]['parent'] = new_id
        nodes[a]['branch_len'] = la
        nodes[b]['branch_len'] = lb
        
        new_id2 = next_id
        next_id += 1
        
        nodes[new_id2] = {'left': new_id, 'right': c, 'parent': None, 'label': None, 'branch_len': 0.0}
        nodes[new_id]['parent'] = new_id2
        nodes[c]['parent'] = new_id2
        nodes[c]['branch_len'] = lc
        nodes[new_id]['branch_len'] = 0.0
        
        return nodes, new_id2
    elif len(active) == 2:
        a, b = active
        d = D[(a, b)]
        la = d / 2.0
        lb = d / 2.0
        
        new_id = next_id
        next_id += 1
        
        nodes[new_id] = {'left': a, 'right': b, 'parent': None, 'label': None, 'branch_len': 0.0}
        nodes[a]['parent'] = new_id
        nodes[b]['parent'] = new_id
        nodes[a]['branch_len'] = la
        nodes[b]['branch_len'] = lb
        
        return nodes, new_id
    else:
        return nodes, active[0]

def _transition_matrix(length):
    exp_term = math.exp(-4.0 / 3.0 * length)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term
    
    return np.array([
        [p_same, p_diff, p_diff, p_diff],
        [p_diff, p_same, p_diff, p_diff],
        [p_diff, p_diff, p_same, p_diff],
        [p_diff, p_diff, p_diff, p_same]
    ])

def _calc_likelihood(nodes, root_id, data, label_to_idx):
    """
    data: (m, n) array of int8
    label_to_idx: dict label -> index in data
    Returns log-likelihood.
    """
    m = data.shape[0]
    L = {}
    
    # Post-order traversal
    def dfs(node_id):
        if node_id is None:
            return
        left = nodes[node_id]['left']
        right = nodes[node_id]['right']
        
        if left is not None:
            dfs(left)
            dfs(right)
            
            L_left = L[left]
            L_right = L[right]
            
            len_left = nodes[left]['branch_len']
            len_right = nodes[right]['branch_len']
            
            T_left = _transition_matrix(len_left)
            T_right = _transition_matrix(len_right)
            
            V_left = L_left @ T_left.T
            V_right = L_right @ T_right.T
            
            L[node_id] = V_left * V_right
        else:
            # Leaf
            label = nodes[node_id]['label']
            idx = label_to_idx[label]
            # L is (m, 4)
            L[node_id] = np.zeros((m, 4), dtype=np.float64)
            # Set the observed state to 1.0
            L[node_id][np.arange(m), data[:, idx]] = 1.0

    dfs(root_id)
    
    # Root likelihood
    L_root = L[root_id]
    # Sum over states for each site, then log and sum
    probs = L_root.sum(axis=1)
    # Avoid log(0)
    probs = np.maximum(probs, 1e-300)
    log_likelihood = np.sum(np.log(probs))
    
    return log_likelihood

def _nni_refine(nodes, root_id, data, label_to_idx, time_limit=10.0):
    """
    Refine the tree using NNI moves.
    """
    start_time = time.time()
    
    # Get all internal edges
    # An internal edge is defined by a node and its parent.
    # We iterate over all nodes that have a parent and are not the root.
    # For each such node (child), we consider the edge (parent, child).
    
    # We need to be careful with the tree structure.
    # NNI move on edge (u, v) where v is child of u.
    # Let w be the other child of u.
    # Let x, y be the children of v.
    
    improved = True
    while improved and (time.time() - start_time) < time_limit and _time_left() > 5.0:
        improved = False
        
        # Get current likelihood
        current_ll = _calc_likelihood(nodes, root_id, data, label_to_idx)
        
        # Iterate over all internal edges
        for v in nodes:
            if v == root_id:
                continue
            if nodes[v]['parent'] is None:
                continue
            if nodes[v]['left'] is None: # Must be internal
                continue
                
            u = nodes[v]['parent']
            # Check if u is internal
            if nodes[u]['left'] is None:
                continue
                
            # Find the other child of u
            if nodes[u]['left'] == v:
                w = nodes[u]['right']
            else:
                w = nodes[u]['left']
                
            # Find children of v
            x = nodes[v]['left']
            y = nodes[v]['right']
            
            # Try NNI move 1: ((x, w), y)
            # Original: ((x, y), w)
            # New: v's children become x, w? No.
            # The edge is (u, v).
            # u has children v and w.
            # v has children x and y.
            
            # Move 1: Swap y and w.
            # New structure:
            # u has children v' and w'?
            # No, NNI changes the topology locally.
            # The four leaves involved are x, y, w, and the rest of the tree (attached to u's parent).
            # The change is in the connections of x, y, w.
            
            # Original: u -> (v, w), v -> (x, y)
            # Move 1: u -> (v_new, w), v_new -> (x, y_new)?
            # No, the standard NNI swaps the two subtrees on the right side of the edge.
            
            # Let's define the move clearly.
            # Edge (u, v).
            # Subtrees: A (at u's parent), B (at v), C (at w), D (at x), E (at y).
            # Wait, x and y are children of v.
            # So the four "tips" of the NNI move are the subtrees rooted at x, y, w, and the rest of the tree (A).
            # The edge is between u and v.
            
            # Original: ((x, y), w)
            # Move 1: ((x, w), y)
            # Move 2: ((y, w), x)
            
            # To implement Move 1:
            # 1. Temporarily change pointers.
            # 2. v's children become x and w? No.
            #    u's children become v and w.
            #    v's children become x and y.
            
            #    New:
            #    u's children: v_new, w
            #    v_new's children: x, y_new?
            
            #    Actually, the node v remains, but its children change?
            #    No, the topology changes.
            #    In Move 1:
            #    The edge (u, v) is replaced by (u, w) and (v, x)?
            
            #    Let's use the standard definition:
            #    Original: ( (x, y), w )
            #    Move 1: ( (x, w), y )
            #    Move 2: ( (y, w), x )
            
            #    This means:
            #    In Move 1:
            #    The node that was v (parent of x, y) now becomes parent of x and w?
            #    And the node that was u (parent of v, w) now becomes parent of v and y?
            
            #    This is getting confusing. Let's just swap the pointers and check validity.
            
            #    Original:
            #    u.left = v, u.right = w
            #    v.left = x, v.right = y
            
            #    Move 1:
            #    u.left = v, u.right = y  (w moves to v?)
            #    v.left = x, v.right = w
            
            #    Let's try this.
            
            #    Save old pointers
            u_l, u_r = nodes[u]['left'], nodes[u]['right']
            v_l, v_r = nodes[v]['left'], nodes[v]['right']
            
            #    Apply Move 1
            nodes[u]['left'] = v
            nodes[u]['right'] = y
            nodes[v]['left'] = x
            nodes[v]['right'] = w
            
            #    Update parents
            nodes[y]['parent'] = u
            nodes[w]['parent'] = v
            
            #    Calculate new likelihood
            new_ll = _calc_likelihood(nodes, root_id, data, label_to_idx)
            
            if new_ll > current_ll + 1e-9:
                current_ll = new_ll
                improved = True
            else:
                # Revert
                nodes[u]['left'] = u_l
                nodes[u]['right'] = u_r
                nodes[v]['left'] = v_l
                nodes[v]['right'] = v_r
                nodes[w]['parent'] = u
                nodes[y]['parent'] = v
                
            #    Apply Move 2
            #    u.left = v, u.right = x
            #    v.left = y, v.right = w
            
            nodes[u]['left'] = v
            nodes[u]['right'] = x
            nodes[v]['left'] = y
            nodes[v]['right'] = w
            
            nodes[x]['parent'] = u
            nodes[w]['parent'] = v
            
            new_ll = _calc_likelihood(nodes, root_id, data, label_to_idx)
            
            if new_ll > current_ll + 1e-9:
                current_ll = new_ll
                improved = True
            else:
                # Revert
                nodes[u]['left'] = u_l
                nodes[u]['right'] = u_r
                nodes[v]['left'] = v_l
                nodes[v]['right'] = v_r
                nodes[w]['parent'] = u
                nodes[x]['parent'] = v

def _tree_to_newick(nodes, root_id):
    """Convert tree structure to Newick string."""
    def dfs(node_id):
        if node_id is None:
            return ""
        left = nodes[node_id]['left']
        right = nodes[node_id]['right']
        label = nodes[node_id]['label']
        branch_len = nodes[node_id]['branch_len']
        
        if left is not None:
            left_str = dfs(left)
            right_str = dfs(right)
            return f"({left_str},{right_str})"
        else:
            return label
        
    # We need to include branch lengths?
    # The task says branch lengths are optional.
    # So we can just return the topology.
    
    def dfs_topo(node_id):
        if node_id is None:
            return ""
        left = nodes[node_id]['left']
        right = nodes[node_id]['right']
        label = nodes[node_id]['label']
        
        if left is not None:
            left_str = dfs_topo(left)
            right_str = dfs_topo(right)
            return f"({left_str},{right_str})"
        else:
            return label
            
    return dfs_topo(root_id) + ";"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    # Build data array
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = np.zeros((len(sequences[labels[0]]), n), dtype=np.int8)
    for j, lb in enumerate(labels):
        seq = sequences[lb]
        for i, c in enumerate(seq):
            data[i, j] = char_map[c]
            
    label_to_idx = {lb: j for j, lb in enumerate(labels)}
    
    # Build distance matrix
    mat = _build_distance_matrix(sequences, labels)
    
    # Build NJ tree
    nodes, root_id = _nj_tree_struct(mat, labels)
    
    # Refine with NNI
    _nni_refine(nodes, root_id, data, label_to_idx)
    
    # Convert to Newick
    newick = _tree_to_newick(nodes, root_id)
    
    return newick

# EVOLVE-BLOCK-END
