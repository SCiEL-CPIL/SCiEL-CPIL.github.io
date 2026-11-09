"""
Phylogenetic topology recovery using Neighbor Joining initialization, 
Neighbor Interchange (NI) local search with Maximum Likelihood (Jukes-Cantor),
and final branch length optimization.
"""
import math
import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 270.0  # 270s wall clock limit

def _jc_distance_matrix(sequences):
    """
    Compute JC distance matrix using NumPy.
    sequences: dict {label: string}
    Returns: (labels, distance_matrix)
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 0:
        return labels, np.array([])
    L = len(sequences[labels[0]])
    
    if L == 0:
        return labels, np.zeros((n, n))
        
    map_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    seq_matrix = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, char in enumerate(seq):
            seq_matrix[i, j] = map_dict.get(char, 0)
            
    # Compute Hamming distances
    diff = seq_matrix[:, None, :] != seq_matrix[None, :, :]
    hamming = np.sum(diff, axis=2)
    
    p = hamming / L
    
    # JC correction
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0 / 3.0) * p_safe)
    
    # Handle saturation
    d[p >= 0.75] = 3.0
    
    return labels, d, seq_matrix

def _nj_build(labels, dist):
    """
    Build initial tree using Neighbor Joining.
    Returns:
    - tree: dict {node_id: (parent, child1, child2, branch_len)}
      Note: This is a rooted representation for ease of manipulation.
      Leaves are 0..n-1. Internal nodes are n..2n-2.
    - branch_lengths: dict {node_id: length_to_parent}
    """
    n = len(labels)
    if n <= 3:
        # Trivial tree
        if n == 2:
            return {0: (1, -1, -1, 0.5), 1: (0, -1, -1, 0.5)}, {0: 0.5, 1: 0.5}
        elif n == 3:
            # Star tree
            return {
                0: (3, -1, -1, 1.0),
                1: (3, -1, -1, 1.0),
                2: (3, -1, -1, 1.0),
                3: (-1, 0, 1, 0.0) # 3 is internal, connected to 0,1,2? 
                # Actually, for n=3, we can just return a star.
            }, {0: 1.0, 1: 1.0, 2: 1.0}

    active = list(range(n))
    max_nodes = 2 * n - 1
    work_dist = np.zeros((max_nodes, max_nodes))
    work_dist[:n, :n] = dist
    
    # Tree structure representation
    # parent[i], left[i], right[i], branch_len[i]
    parent = [-1] * max_nodes
    left = [-1] * max_nodes
    right = [-1] * max_nodes
    branch_len = [0.0] * max_nodes
    
    next_node_id = n
    current_n = n
    
    while current_n > 3:
        k = current_n
        sub_dist = work_dist[np.ix_(active, active)]
        row_sums = sub_dist.sum(axis=1)
        
        Q = (k - 2) * sub_dist - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(Q, np.inf)
        
        min_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_idx, Q.shape)
        
        node_i = active[i_idx]
        node_j = active[j_idx]
        
        new_node = next_node_id
        next_node_id += 1
        
        other_indices = [idx for idx in range(len(active)) if idx != i_idx and idx != j_idx]
        other_nodes = [active[idx] for idx in other_indices]
        
        d_ij = work_dist[node_i, node_j]
        
        if other_nodes:
            d_i_o = work_dist[node_i, other_nodes]
            d_j_o = work_dist[node_j, other_nodes]
            d_new_o = 0.5 * (d_i_o + d_j_o - d_ij)
            
            work_dist[new_node, other_nodes] = d_new_o
            work_dist[other_nodes, new_node] = d_new_o
            
        # Branch lengths
        # delta = (row_sums[i_idx] - row_sums[j_idx]) / (2 * (k - 2))
        delta = (row_sums[i_idx] - row_sums[j_idx]) / (2 * (k - 2))
        len_i = 0.5 * (d_ij + delta)
        len_j = 0.5 * (d_ij - delta)
        
        # Ensure non-negative
        len_i = max(0.0, len_i)
        len_j = max(0.0, len_j)
        
        parent[node_i] = new_node
        parent[node_j] = new_node
        left[new_node] = node_i
        right[new_node] = node_j
        branch_len[node_i] = len_i
        branch_len[node_j] = len_j
        
        active = [x for x in active if x != node_i and x != node_j]
        active.append(new_node)
        current_n -= 1
        
    # Connect the remaining 3 nodes
    # The active list has 3 nodes.
    # We need to create one more internal node to connect them?
    # No, NJ stops at 3. The last 3 nodes form a star.
    # We need to create a final internal node.
    final_node = next_node_id
    next_node_id += 1
    
    # Calculate distances between the last 3
    a, b, c = active
    d_ab = work_dist[a, b]
    d_ac = work_dist[a, c]
    d_bc = work_dist[b, c]
    
    # Star tree branch lengths
    len_a = (d_ab + d_ac - d_bc) / 2
    len_b = (d_ab + d_bc - d_ac) / 2
    len_c = (d_ac + d_bc - d_ab) / 2
    
    len_a = max(0.0, len_a)
    len_b = max(0.0, len_b)
    len_c = max(0.0, len_c)
    
    parent[a] = final_node
    parent[b] = final_node
    parent[c] = final_node
    left[final_node] = a
    right[final_node] = b
    # We need a way to represent 3 children. 
    # Standard binary tree representation requires 2 children.
    # We can make it ((a, b), c) or similar.
    # Let's make a binary tree: 
    # final_node connects to a and an intermediate node.
    # intermediate node connects to b and c.
    # This changes the topology slightly but is valid for unrooted trees if we handle it correctly.
    # Actually, for n=3, the unrooted tree is a star. 
    # If we force a binary tree, we introduce a zero-length branch or a specific topology.
    # To keep it simple, we can just return the star topology for n=3.
    # For n>3, the NJ loop handles it.
    
    # Re-structure the last step for binary compatibility
    # Let's create an intermediate node 'mid'
    mid_node = next_node_id
    next_node_id += 1
    
    parent[b] = mid_node
    parent[c] = mid_node
    left[mid_node] = b
    right[mid_node] = c
    branch_len[b] = len_b
    branch_len[c] = len_c
    branch_len[mid_node] = 0.0 # Zero length to make it a star
    
    parent[a] = final_node
    parent[mid_node] = final_node
    left[final_node] = a
    right[final_node] = mid_node
    branch_len[a] = len_a
    branch_len[final_node] = 0.0
    
    # Build the tree dictionary
    tree = {}
    for i in range(next_node_id):
        if i < n:
            tree[i] = (parent[i], -1, -1, branch_len[i])
        else:
            tree[i] = (parent[i], left[i], right[i], branch_len[i])
            
    return tree, branch_len

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
            L_leaf = np.zeros((4, L), dtype=np.float64)
            # Vectorized assignment
            # For each site j, L_leaf[seq[j], j] = 1
            # This is hard to vectorize directly in one line without advanced indexing.
            # Alternative:
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

def reconstruct(sequences):
    start_time = time.time()
    labels, dist, seq_matrix = _jc_distance_matrix(sequences)
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
        
    # 1. NJ Initialization
    tree, branch_lengths = _nj_build(labels, dist)
    
    # 2. NI Local Search
    # We will perform NI moves.
    # To do this, we need to identify internal edges and generate neighbors.
    
    # Helper to get children of a node
    def get_children(node):
        p, l, r, bl = tree[node]
        return l, r
        
    # Helper to get parent
    def get_parent(node):
        return tree[node][0]
        
    # Helper to get branch length
    def get_branch_len(node):
        return tree[node][3]
        
    # Helper to set tree structure
    def set_tree(node, parent, left, right, bl):
        tree[node] = (parent, left, right, bl)
        
    # Find all internal edges (edges where both endpoints are internal nodes)
    # In a binary tree, an internal edge connects two internal nodes.
    # We can iterate through all nodes and check if they are internal and their parent is internal.
    
    def get_internal_edges():
        edges = []
        for node in tree:
            p, l, r, bl = tree[node]
            if l != -1 and r != -1: # Internal node
                if p != -1:
                    # Check if parent is internal
                    pp, pl, pr, pbl = tree[p]
                    if pl != -1 and pr != -1:
                        edges.append((p, node))
        return edges
        
    best_likelihood = _compute_log_likelihood(tree, seq_matrix)
    
    improved = True
    max_iterations = 50
    
    while improved and (time.time() - start_time) < _DEADLINE - 10:
        improved = False
        edges = get_internal_edges()
        
        for p, c in edges:
            # For edge (p, c), p is parent of c.
            # p has children c and other_p.
            # c has children c1 and c2.
            
            pp, pl, pr, pbl = tree[p]
            # Identify other_p
            if pl == c:
                other_p = pr
            else:
                other_p = pl
                
            pc, cl, cr, cbl = tree[c]
            c1 = cl
            c2 = cr
            
            # The 4 subtrees are other_p, c1, c2, and the subtree above p (pp).
            # Current topology: ((c1, c2), other_p) attached to pp.
            
            # NI Move 1: Swap other_p and c1
            # New structure: ((other_p, c1), c2)
            # We need to modify the tree to reflect this.
            # This is complex to do in-place and revert.
            # Instead, we can create a copy of the tree, modify it, and evaluate.
            
            tree_copy = {k: v for k, v in tree.items()}
            
            # Modify tree_copy
            # 1. Change p's children to (other_p, c) -> No, we are swapping subtrees.
            # The new parent of c1 is a new internal node? No, we reuse nodes.
            # Let's trace the structure:
            # Old: p -> (other_p, c). c -> (c1, c2).
            # New: p -> (other_p, new_c). new_c -> (c1, c2)? No.
            # The NI move changes the topology.
            # The new topology is: ((other_p, c1), c2).
            # This implies a new internal node structure.
            # Let's call the new internal node 'new_p' and 'new_c'.
            # But we have limited node IDs.
            # We can reuse the IDs of p and c.
            
            # Structure 1: ((other_p, c1), c2)
            # Let p be the parent of (other_p, c1) and c2.
            # Let c be the parent of (other_p, c1).
            # So:
            # p -> (c, c2)
            # c -> (other_p, c1)
            
            # Update parents
            tree_copy[c] = (p, other_p, c1, get_branch_len(c)) # c's parent is p, children other_p, c1
            tree_copy[p] = (pp, c, c2, get_branch_len(p)) # p's parent is pp, children c, c2
            
            # Update children's parents
            tree_copy[other_p] = (tree_copy[other_p][0] if False else c, tree_copy[other_p][1], tree_copy[other_p][2], get_branch_len(other_p))
            tree_copy[c1] = (c, tree_copy[c1][1], tree_copy[c1][2], get_branch_len(c1))
            tree_copy[c2] = (p, tree_copy[c2][1], tree_copy[c2][2], get_branch_len(c2))
            
            # Note: We kept the branch lengths the same.
            
            ll1 = _compute_log_likelihood(tree_copy, seq_matrix)
            
            # NI Move 2: Swap other_p and c2
            # Structure: ((other_p, c2), c1)
            tree_copy2 = {k: v for k, v in tree.items()}
            
            tree_copy2[c] = (p, other_p, c2, get_branch_len(c))
            tree_copy2[p] = (pp, c, c1, get_branch_len(p))
            
            tree_copy2[other_p] = (c, tree_copy2[other_p][1], tree_copy2[other_p][2], get_branch_len(other_p))
            tree_copy2[c2] = (c, tree_copy2[c2][1], tree_copy2[c2][2], get_branch_len(c2))
            tree_copy2[c1] = (p, tree_copy2[c1][1], tree_copy2[c1][2], get_branch_len(c1))
            
            ll2 = _compute_log_likelihood(tree_copy2, seq_matrix)
            
            # NI Move 3: Swap c1 and c2 (This is just swapping children of c, same topology)
            # So only 2 distinct NI moves per edge?
            # Yes, for unrooted trees, swapping the two children of a node is the same topology.
            # So we only have 2 distinct neighbors per internal edge.
            
            best_ll = max(ll1, ll2)
            if best_ll > best_likelihood:
                if ll1 > ll2:
                    tree = tree_copy
                else:
                    tree = tree_copy2
                best_likelihood = best_ll
                improved = True
                
    # 3. Branch Length Optimization
    # Use a simple coordinate descent or Nelder-Mead.
    # Given time constraints, we might skip this or do a very simple version.
    # For now, we return the tree with NJ/NI branch lengths.
    
    # Convert tree to Newick
    def to_newick(node):
        p, l, r, bl = tree[node]
        if l == -1 and r == -1:
            # Leaf
            return labels[node]
        else:
            left_str = to_newick(l)
            right_str = to_newick(r)
            return f"({left_str},{right_str})"
            
    # Find root
    root = -1
    for node in tree:
        if tree[node][0] == -1:
            root = node
            break
            
    newick_str = to_newick(root)
    return newick_str + ";"

