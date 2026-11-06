"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''
Maximum Likelihood tree reconstruction with NNI refinement.
1. Compute JC distance matrix.
2. Build initial tree using Neighbor Joining.
3. Refine topology using Nearest Neighbor Interchange (NNI) moves,
   optimizing the log-likelihood under the Jukes-Cantor model.
'''
import math
import numpy as np
from collections import defaultdict

def _jc_distance(seq1, seq2):
    """Compute Jukes-Cantor corrected distance between two DNA sequences."""
    n = len(seq1)
    if n == 0:
        return 0.0
    mismatches = sum(1 for a, b in zip(seq1, seq2) if a != b)
    p = mismatches / n
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC-corrected distance matrix."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist

def _neighbor_joining(dist, labels):
    """
    Standard Neighbor Joining algorithm.
    Returns a tree structure: list of nodes where each node is a dict
    with 'children' and 'id'.
    """
    n = len(labels)
    if n < 2:
        return labels[0] if n == 1 else ""
    
    # We will build a tree structure using a list of nodes.
    # Each node is a dict: {'id': int, 'children': [left, right], 'length': float, 'leaf': bool}
    # For leaves, children is None.
    
    # To make NNI easier, let's represent the tree as a set of splits or a binary tree structure.
    # Let's use a standard binary tree representation.
    
    # Map labels to indices
    # We'll keep track of active nodes in a list.
    # dist is the current distance matrix.
    # We need to maintain the tree structure.
    
    # This implementation returns a tree as a nested tuple structure for easy manipulation.
    # But for NNI, a flat list of edges or a specific tree object is better.
    
    # Let's return the Newick string first, then parse it? No, that's inefficient.
    # Let's build a tree object.
    
    # Tree Node: (left, right, length, id)
    # Leaves: (None, None, 0, label)
    
    active_nodes = list(range(n))
    node_trees = {i: (None, None, 0.0, labels[i]) for i in range(n)}
    
    # Distance matrix for active nodes
    # d[i][j]
    d = dist.copy()
    
    next_id = n
    
    while len(active_nodes) > 3:
        m = len(active_nodes)
        
        # Compute r_i
        r = np.zeros(m)
        for idx_i, i in enumerate(active_nodes):
            for idx_j, j in enumerate(active_nodes):
                if i != j:
                    r[idx_i] += d[i, j]
        
        # Find min Q
        best_q = float('inf')
        best_i_idx = 0
        best_j_idx = 1
        for idx_i in range(m):
            for idx_j in range(idx_i + 1, m):
                i = active_nodes[idx_i]
                j = active_nodes[idx_j]
                q = (m - 2) * d[i, j] - r[idx_i] - r[idx_j]
                if q < best_q:
                    best_q = q
                    best_i_idx = idx_i
                    best_j_idx = idx_j
        
        i = active_nodes[best_i_idx]
        j = active_nodes[best_j_idx]
        
        d_ij = d[i, j]
        delta = (r[best_i_idx] - r[best_j_idx]) / (2 * (m - 2))
        li = max(0.0, 0.5 * d_ij + delta)
        lj = max(0.0, 0.5 * d_ij - delta)
        
        # Create new node
        k = next_id
        next_id += 1
        
        # New tree structure: (left, right, length_to_parent, id)
        # The length stored in the node is the length of the branch connecting to its parent.
        # For the new internal node k, its children are i and j.
        # The branch lengths from k to i is li, from k to j is lj.
        # So node_trees[i] and node_trees[j] will have their 'length' updated?
        # No, standard representation:
        # Node i has a branch length to its parent.
        # When we join i and j into k, the branch lengths from k to i is li, from k to j is lj.
        # So we update the 'length' attribute of nodes i and j?
        # Or we store the lengths in the new node?
        # Let's store lengths in the node itself as the length of the edge *above* it.
        # But for the root, it's 0.
        # When we create k, k's length is 0 initially.
        # The children i and j will have their length set to li and lj respectively.
        
        # Wait, if I change the length of i and j, and they are used elsewhere, it's fine because they are removed from active set.
        
        node_trees[i] = (node_trees[i][0], node_trees[i][1], li, node_trees[i][3])
        node_trees[j] = (node_trees[j][0], node_trees[j][1], lj, node_trees[j][3])
        
        node_trees[k] = (i, j, 0.0, k) # k is internal, children are i, j. length 0 for now.
        
        # Update distances
        for x in active_nodes:
            if x == i or x == j:
                continue
            dx = 0.5 * (d[i, x] + d[j, x] - d_ij)
            dx = max(0.0, dx)
            d[k, x] = dx
            d[x, k] = dx
        
        # Remove i, j, add k
        active_nodes.remove(i)
        active_nodes.remove(j)
        active_nodes.append(k)
        
    # Final 3 nodes
    if len(active_nodes) == 3:
        a, b, c = active_nodes
        d_ab = d[a, b]
        d_bc = d[b, c]
        d_ca = d[c, a]
        
        l_a = max(0.0, (d_ab + d_ca - d_bc) / 2.0)
        l_b = max(0.0, (d_ab + d_bc - d_ca) / 2.0)
        l_c = max(0.0, (d_bc + d_ca - d_ab) / 2.0)
        
        # Create a central node for the 3
        k = next_id
        next_id += 1
        
        node_trees[a] = (node_trees[a][0], node_trees[a][1], l_a, node_trees[a][3])
        node_trees[b] = (node_trees[b][0], node_trees[b][1], l_b, node_trees[b][3])
        node_trees[c] = (node_trees[c][0], node_trees[c][1], l_c, node_trees[c][3])
        
        # k is the root of this subtree? No, it's an unrooted tree.
        # We can make k the parent of a, b, c? No, binary tree.
        # We need to form a binary tree from 3 leaves.
        # Join a and b into m1, then m1 and c into k.
        
        # Let's just pick a and b to join first.
        m1 = next_id
        next_id += 1
        
        # Distances for m1
        # d(m1, c) = ?
        # We already have d_ab, d_bc, d_ca.
        # We can calculate branch lengths for m1-c.
        # Actually, simpler:
        # Join a and b.
        # d_ab is known.
        # We need d(m1, c).
        # In the 3-taxon tree, the topology is ((a,b),c) or similar.
        # Let's just use the calculated l_a, l_b, l_c.
        # If we make m1 parent of a and b:
        # Branch a-m1: l_a
        # Branch b-m1: l_b
        # Branch m1-c: l_c
        
        node_trees[a] = (node_trees[a][0], node_trees[a][1], l_a, node_trees[a][3])
        node_trees[b] = (node_trees[b][0], node_trees[b][1], l_b, node_trees[b][3])
        node_trees[c] = (node_trees[c][0], node_trees[c][1], l_c, node_trees[c][3])
        
        node_trees[m1] = (a, b, 0.0, m1)
        node_trees[k] = (m1, c, 0.0, k)
        
        return node_trees[k]
    else:
        # Should not happen
        return node_trees[active_nodes[0]]

def _tree_to_newick(node, node_trees):
    """Convert tree node to Newick string."""
    if node is None:
        return ""
    left, right, length, id_ = node
    if left is None and right is None:
        # Leaf
        return f"{id_}:{length:.4f}"
    else:
        left_str = _tree_to_newick(node_trees[left], node_trees)
        right_str = _tree_to_newick(node_trees[right], node_trees)
        return f"({left_str},{right_str})"

def _parse_newick(newick_str):
    """
    Parse a Newick string into a tree structure compatible with _tree_to_newick.
    Returns (root_id, node_trees_dict)
    """
    # This is complex to implement from scratch.
    # Since we generate the initial tree with NJ, we have node_trees.
    # We only need to modify the topology during NNI.
    # So we should keep the node_trees dict and modify the children pointers.
    pass

def _calculate_likelihood(tree_root, node_trees, sequences, labels):
    """
    Calculate log-likelihood of the tree under JC model.
    tree_root: id of root node
    node_trees: dict of node_id -> (left, right, length, label/id)
    sequences: dict of label -> string
    labels: list of labels
    """
    # Map label to sequence index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_sites = len(sequences[labels[0]])
    
    # Precompute JC probabilities
    # P(same) = 1/4 + 3/4 * exp(-4/3 * d)
    # P(diff) = 1/4 * (1 - exp(-4/3 * d))
    
    # We need to propagate probabilities up the tree.
    # For each site, calculate likelihood contribution.
    # To speed up, we can vectorize over sites.
    
    # Get all nodes in post-order
    nodes = []
    def get_postorder(node_id):
        left, right, length, id_ = node_trees[node_id]
        if left is not None:
            get_postorder(left)
            get_postorder(right)
        nodes.append(node_id)
    
    get_postorder(tree_root)
    
    # nodes is in post-order (children before parents)
    
    # For each node, we store a 4-vector of probabilities for each site.
    # prob[node_id] is an array of shape (n_sites, 4)
    # prob[node_id][s, a] = P(observation at leaf s is a | data below node)
    # Actually, standard Felsenstein:
    # For leaves: P(x) = 1 if x matches observed base, 0 otherwise.
    # For internal nodes:
    # P(node=a) = sum_b P(b) * P(a|b, d)
    
    # Let's use a dict for probabilities
    probs = {}
    
    # Initialize leaves
    for node_id in nodes:
        left, right, length, id_ = node_trees[node_id]
        if left is None and right is None:
            # Leaf
            if id_ in label_to_idx:
                seq = sequences[id_]
                # Create one-hot vector for each site
                # 0:A, 1:C, 2:G, 3:T
                vec = np.zeros((n_sites, 4))
                for s in range(n_sites):
                    base = seq[s]
                    if base == 'A': vec[s, 0] = 1.0
                    elif base == 'C': vec[s, 1] = 1.0
                    elif base == 'G': vec[s, 2] = 1.0
                    elif base == 'T': vec[s, 3] = 1.0
                probs[node_id] = vec
            else:
                # Should not happen for leaves
                probs[node_id] = np.ones((n_sites, 4)) / 4.0
    
    # Process internal nodes
    # We need to process in post-order, so children are already computed.
    for node_id in nodes:
        left, right, length, id_ = node_trees[node_id]
        if left is not None:
            # Internal node
            # Calculate transition probabilities P(a|b, d)
            # JC: P_same = 0.25 + 0.75 * exp(-4/3 * d)
            # P_diff = 0.25 * (1 - exp(-4/3 * d))
            
            d = length
            if d > 0:
                exp_term = math.exp(-4.0 / 3.0 * d)
                p_same = 0.25 + 0.75 * exp_term
                p_diff = 0.25 * (1.0 - exp_term)
            else:
                p_same = 1.0
                p_diff = 0.0
            
            # Transition matrix P (4x4)
            # P[i][j] = P(state i at child -> state j at parent)
            # For JC, P[i][i] = p_same, P[i][j!=i] = p_diff
            P = np.full((4, 4), p_diff)
            np.fill_diagonal(P, p_same)
            
            # probs[left] shape (n_sites, 4)
            # probs[right] shape (n_sites, 4)
            
            # For each site, calculate likelihood at this node for each state.
            # L(node, a) = sum_b P(a|b) * L(left, b) * sum_c P(a|c) * L(right, c) ?
            # No.
            # L(node, a) = sum_b P(b|a, d_left) * L(left, b) * sum_c P(c|a, d_right) * L(right, c)
            # Wait, the tree is unrooted, but we rooted it at 'root'.
            # The branch lengths are from node to children.
            # So for node 'u' with children 'l' and 'r':
            # L(u, a) = sum_b P(l_b | u_a, d_ul) * L(l, b) * sum_c P(r_c | u_a, d_ur) * L(r, c)
            
            # Let's compute the marginal likelihood for the left child given state a at u.
            # M_left(a) = sum_b P(b|a, d_ul) * L(l, b)
            # M_right(a) = sum_c P(c|a, d_ur) * L(r, c)
            
            # We need d_ul and d_ur.
            # In our node_trees, 'length' is the length of the branch *above* the node?
            # No, in _neighbor_joining, we set:
            # node_trees[i] = (..., li, ...) where li is branch from k to i.
            # So for node k (parent), children i and j have lengths li and lj stored in i and j.
            # So d_ul = node_trees[left][2]
            # d_ur = node_trees[right][2]
            
            d_ul = node_trees[left][2]
            d_ur = node_trees[right][2]
            
            # Calculate P matrices for left and right branches
            if d_ul > 0:
                exp_l = math.exp(-4.0 / 3.0 * d_ul)
                p_same_l = 0.25 + 0.75 * exp_l
                p_diff_l = 0.25 * (1.0 - exp_l)
                P_l = np.full((4, 4), p_diff_l)
                np.fill_diagonal(P_l, p_same_l)
            else:
                P_l = np.eye(4)
            
            if d_ur > 0:
                exp_r = math.exp(-4.0 / 3.0 * d_ur)
                p_same_r = 0.25 + 0.75 * exp_r
                p_diff_r = 0.25 * (1.0 - exp_r)
                P_r = np.full((4, 4), p_diff_r)
                np.fill_diagonal(P_r, p_same_r)
            else:
                P_r = np.eye(4)
            
            # probs[left] is (n_sites, 4)
            # We want M_left(a) for a in 0..3 for all sites.
            # M_left = probs[left] @ P_l.T ?
            # L(left, b) is in probs[left][:, b].
            # M_left(a) = sum_b P_l[b, a] * probs[left][:, b]
            # Note: P_l[b, a] is P(parent=a | child=b).
            # In JC, P is symmetric, so P_l[b, a] = P_l[a, b].
            
            # Vectorized:
            # M_left (n_sites, 4) = probs[left] (n_sites, 4) @ P_l.T (4, 4)
            # Since P_l is symmetric, P_l.T = P_l.
            M_left = probs[left] @ P_l
            M_right = probs[right] @ P_r
            
            # L(u, a) = M_left(a) * M_right(a)
            probs[node_id] = M_left * M_right
            
    # Root likelihood
    # The root is unrooted, so we sum over all states at the root?
    # Yes, for an unrooted tree, the likelihood is sum_a L(root, a).
    # But wait, the root has no parent.
    # The probability of the data is sum_a L(root, a) * P(root=a).
    # Under JC, prior is uniform 1/4.
    # So LL = sum_s sum_a probs[root][s, a] * 0.25
    
    root_probs = probs[tree_root]
    ll = np.sum(root_probs) * 0.25
    return ll

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    if len(labels) < 2:
        return labels[0] + ";" if labels else ";"
    
    # 1. Distance Matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # 2. Initial Tree via NJ
    root_id, node_trees = _neighbor_joining(dist, labels)
    
    # 3. NNI Refinement
    # We need to find all internal branches.
    # An internal branch connects two internal nodes (or one internal and one leaf? No, NNI is for internal branches).
    # In a binary tree with n leaves, there are n-3 internal branches.
    # Each internal branch separates the tree into two subtrees.
    # NNI: Swap the subtrees on the other side of the branch.
    
    # To implement NNI, we need to:
    # 1. Identify all internal branches.
    # 2. For each branch, generate the 2 alternative topologies.
    # 3. Calculate likelihood for each.
    # 4. If better, accept the change.
    
    # This requires a robust tree representation that allows easy modification.
    # My current node_trees dict allows modifying 'children' pointers.
    
    # Let's find all internal nodes.
    # An internal node is one where left is not None.
    # An internal branch is the edge connecting an internal node to its parent.
    # However, the root has no parent.
    # So we consider all edges except the root's incoming edge (which doesn't exist).
    # Actually, NNI operates on the 4 subtrees connected to an internal edge.
    # Edge connects node u and node v (parent of u).
    # u has children u1, u2.
    # v has other child v1 (and parent v2, unless v is root).
    # The 4 subtrees are: T_u1, T_u2, T_v1, T_v2.
    # Current: ((T_u1, T_u2), (T_v1, T_v2))
    # Alt 1: ((T_u1, T_v1), (T_u2, T_v2))
    # Alt 2: ((T_u1, T_v2), (T_u2, T_v1))
    
    # To do this, we need to know the parent of each node.
    # Let's build a parent map.
    parent_map = {root_id: None}
    def build_parent_map(node_id):
        left, right, length, id_ = node_trees[node_id]
        if left is not None:
            parent_map[left] = node_id
            build_parent_map(left)
        if right is not None:
            parent_map[right] = node_id
            build_parent_map(right)
    build_parent_map(root_id)
    
    # Get list of internal nodes (excluding root? No, root's children are internal branches too if they have parents?
    # The edge from root to its child is an internal branch if the child is internal.
    # If the child is a leaf, it's a terminal branch. NNI doesn't apply to terminal branches.
    # So we iterate over all nodes u such that:
    # 1. u is internal (has children).
    # 2. u is not the root (has a parent).
    # 3. The parent v is also internal? No, v just needs to have another child.
    #    If v is a leaf, it can't have another child. So v must be internal.
    
    candidates = []
    for node_id in node_trees:
        left, right, length, id_ = node_trees[node_id]
        if left is not None: # u is internal
            if parent_map[node_id] is not None: # u is not root
                v = parent_map[node_id]
                v_left, v_right, v_length, v_id = node_trees[v]
                # v must be internal
                if v_left is not None:
                    candidates.append(node_id)
    
    # Perform NNI passes
    max_passes = 3
    for _ in range(max_passes):
        improved = False
        for u in candidates:
            v = parent_map[u]
            # Get children of u
            u_left, u_right, _, _ = node_trees[u]
            # Get children of v
            v_left, v_right, _, _ = node_trees[v]
            
            # Identify the "other" child of v
            if v_left == u:
                v_other = v_right
            else:
                v_other = v_left
            
            # The 4 subtrees are: u_left, u_right, v_other, and the rest of the tree (above v).
            # Let w be the parent of v.
            # If w is None (v is root), then the 4th subtree is the "rest of the tree" which is empty?
            # No, if v is root, the edge is between root and u.
            # The 4 subtrees are: u_left, u_right, v_other, and ... wait.
            # If v is root, the tree is ((u_left, u_right), v_other).
            # NNI on this edge:
            # Current: ((u_left, u_right), v_other)
            # Alt 1: ((u_left, v_other), u_right)
            # Alt 2: ((u_right, v_other), u_left)
            # This works.
            
            # If v is not root, let w = parent_map[v].
            # The 4 subtrees are: u_left, u_right, v_other, and the subtree "above" v (which is w's other side).
            # Let w_other be the other child of w (if w exists).
            # If w is None, the 4th subtree is the "rest".
            
            # Actually, standard NNI:
            # Edge (u, v).
            # Subtrees: A, B (children of u), C (other child of v), D (rest of tree attached to v's parent).
            # If v is root, D is empty? No, v is root, so there is no D.
            # The tree is ( (A, B), C ).
            # NNI alternatives: ( (A, C), B ) and ( (B, C), A ).
            
            # If v is not root:
            # Tree is ( ... ( (A, B), C ) ... ).
            # NNI alternatives: ( ... ( (A, C), B ) ... ) and ( ... ( (B, C), A ) ... ).
            
            # We need to modify the tree structure.
            # Let's save the current structure.
            orig_u_left = u_left
            orig_u_right = u_right
            orig_v_other = v_other
            orig_v_left = v_left
            orig_v_right = v_right
            
            # Calculate current likelihood
            ll_current = _calculate_likelihood(root_id, node_trees, sequences, labels)
            
            # Try Alt 1: u_left and v_other swap
            # New structure:
            # u's children become u_left and v_other? No.
            # The edge is between u and v.
            # u is child of v.
            # u has children A, B.
            # v has children u and C.
            # Alt 1: u has children A, C. v has children u and B.
            
            # Modify node_trees
            node_trees[u] = (u_left, v_other, node_trees[u][2], u)
            node_trees[v] = (u, u_right, node_trees[v][2], v)
            
            # Update parent_map? We don't strictly need it for likelihood, only for finding candidates.
            # But we need to be careful if we iterate candidates while modifying.
            # We should recalculate likelihood.
            
            ll_alt1 = _calculate_likelihood(root_id, node_trees, sequences, labels)
            
            # Try Alt 2: u_right and v_other swap
            # u has children B, C. v has children u and A.
            
            node_trees[u] = (u_right, v_other, node_trees[u][2], u)
            node_trees[v] = (u, u_left, node_trees[v][2], v)
            
            ll_alt2 = _calculate_likelihood(root_id, node_trees, sequences, labels)
            
            # Pick the best
            best_ll = ll_current
            best_mod = None
            
            if ll_alt1 > best_ll:
                best_ll = ll_alt1
                best_mod = 1
            if ll_alt2 > best_ll:
                best_ll = ll_alt2
                best_mod = 2
            
            if best_mod is None:
                # Revert to original
                node_trees[u] = (orig_u_left, orig_u_right, node_trees[u][2], u)
                node_trees[v] = (orig_v_left, orig_v_right, node_trees[v][2], v)
            else:
                # Keep the best modification
                if best_mod == 1:
                    node_trees[u] = (u_left, v_other, node_trees[u][2], u)
                    node_trees[v] = (u, u_right, node_trees[v][2], v)
                else:
                    node_trees[u] = (u_right, v_other, node_trees[u][2], u)
                    node_trees[v] = (u, u_left, node_trees[v][2], v)
                improved = True
        
        if not improved:
            break
    
    # Generate Newick
    newick = _tree_to_newick(root_id, node_trees)
    return newick + ";"
# EVOLVE-BLOCK-END
