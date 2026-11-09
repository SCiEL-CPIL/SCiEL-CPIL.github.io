<<<<<<< SEARCH
def _compute_log_likelihood(tree, seq_matrix):
    """
    Compute log-likelihood of the tree given the sequence matrix under JC model.
    tree: dict {node_id: (parent, child1, child2, branch_len)}
    seq_matrix: NxL array of ints (0-3)
    """
    n = seq_matrix.shape[0]
    L = seq_matrix.shape[1]
    
    # Find the root (node with parent -1)
    root = -1
    for node, (p, l, r, bl) in tree.items():
        if p == -1:
            root = node
            break
            
    if root == -1:
        # If no root found, pick 0
        root = 0

    # Post-order traversal
    # We need to compute L_u(s) for each node u.
    # L_u is an array of shape (4, L)
    
    # To avoid recursion depth issues and for speed, we can do iterative post-order.
    # Or just use recursion since n is small.
    
    import sys
    sys.setrecursionlimit(10000)
    
    def dfs(node):
        p, l, r, bl = tree[node]
        
        if l == -1 and r == -1:
            # Leaf
            # L_s = 1 if seq == s else 0
            # We represent this as a vector of 4s.
            # Instead of creating a (4, L) array for each leaf, we can just use the seq_matrix row.
            # But for uniformity, let's create the array.
            # Optimization: For leaves, we don't need to store the full (4, L) array if we handle them specially.
            # But let's stick to the general case for simplicity.
            
            # Create a (4, L) array
            # L[0, :] = 1 where seq == 0, else 0
            # This is slow if done naively.
            # We can use:
            # L_leaf = np.eye(4, dtype=np.float64)[:, None] * (seq_matrix[node] == np.arange(4)[:, None])
            # This creates a (4, L) array.
            
            # Actually, for JC, the likelihood calculation for a leaf is just the indicator.
            # We can return the seq_matrix row and a flag?
            # Let's just compute the (4, L) array.
            
            # Efficient way:
            # L_leaf[0, :] = (seq_matrix[node] == 0)
            # L_leaf[1, :] = (seq_matrix[node] == 1)
            # ...
            
            # Using boolean indexing:
            # L_leaf = np.zeros((4, L), dtype=np.float64)
            # L_leaf[seq_matrix[node], np.arange(L)] = 1.0
            # This is fast.
            
            L_node = np.zeros((4, L), dtype=np.float64)
            L_node[seq_matrix[node], np.arange(L)] = 1.0
            return L_node
            
        # Internal node
        L_left = dfs(l)
        L_right = dfs(r)
        
        # Calculate S_left and S_right
        S_left = np.sum(L_left, axis=0) # Shape (L,)
        S_right = np.sum(L_right, axis=0)
        
        # Branch lengths
        # bl is the branch length from this node to its parent.
        # The children have their own branch lengths.
        # We need the branch lengths from this node to its children.
        # In our tree representation, tree[child][3] is the branch length from child to parent.
        # So, branch length from node to l is tree[l][3].
        
        len_l = tree[l][3]
        len_r = tree[r][3]
        
        exp_l = math.exp(-4.0 * len_l)
        exp_r = math.exp(-4.0 * len_r)
        
        # Contribution from left child to parent node for each state s
        # C_l(s) = 0.25 * (S_left + 3 * exp_l * L_left(s))
        C_left = 0.25 * (S_left[None, :] + 3.0 * exp_l * L_left)
        
        # Contribution from right child
        C_right = 0.25 * (S_right[None, :] + 3.0 * exp_r * L_right)
        
        # L_node(s) = C_left(s) * C_right(s)
        L_node = C_left * C_right
        
        return L_node
        
    L_root = dfs(root)
    
    # Total likelihood is sum over root states (assuming uniform prior 0.25)
    # L_total = 0.25 * sum(L_root)
    log_likelihood = np.sum(np.log(np.sum(L_root, axis=0) + 1e-300))
    
    return log_likelihood
=======
def _compute_log_likelihood(tree, seq_matrix):
    """
    Compute log-likelihood of the tree given the sequence matrix under JC model.
    Uses iterative post-order traversal to avoid recursion errors.
    tree: dict {node_id: (parent, child1, child2, branch_len)}
    seq_matrix: NxL array of ints (0-3)
    """
    n = seq_matrix.shape[0]
    L = seq_matrix.shape[1]
    
    # Find the root (node with parent -1)
    root = -1
    for node, (p, l, r, bl) in tree.items():
        if p == -1:
            root = node
            break
            
    if root == -1:
        root = 0

    # Iterative post-order traversal
    # We store the likelihood vector for each node in a dictionary
    L_cache = {}
    
    # Stack for DFS: (node, state)
    # state 0: first visit, state 1: second visit (children processed)
    stack = [(root, 0)]
    
    while stack:
        node, state = stack.pop()
        p, l, r, bl = tree[node]
        
        if l == -1 and r == -1:
            # Leaf
            L_node = np.zeros((4, L), dtype=np.float64)
            L_node[seq_matrix[node], np.arange(L)] = 1.0
            L_cache[node] = L_node
        else:
            if state == 0:
                # Push back for second visit
                stack.append((node, 1))
                # Push children
                stack.append((r, 0))
                stack.append((l, 0))
            else:
                # Children are processed
                L_left = L_cache[l]
                L_right = L_cache[r]
                
                S_left = np.sum(L_left, axis=0)
                S_right = np.sum(L_right, axis=0)
                
                len_l = tree[l][3]
                len_r = tree[r][3]
                
                exp_l = math.exp(-4.0 * len_l)
                exp_r = math.exp(-4.0 * len_r)
                
                C_left = 0.25 * (S_left[None, :] + 3.0 * exp_l * L_left)
                C_right = 0.25 * (S_right[None, :] + 3.0 * exp_r * L_right)
                
                L_node = C_left * C_right
                L_cache[node] = L_node
                
    L_root = L_cache[root]
    log_likelihood = np.sum(np.log(np.sum(L_root, axis=0) + 1e-300))
    
    return log_likelihood
>>>>>>> REPLACE

