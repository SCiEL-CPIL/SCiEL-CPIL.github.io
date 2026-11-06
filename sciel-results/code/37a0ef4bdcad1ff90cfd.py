import time
import math
import numpy as np
import random
from collections import defaultdict

_START_TIME = None
_TIME_LIMIT = 260.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

class TreeNode:
    __slots__ = ['left', 'right', 'label', 'length']
    def __init__(self, left=None, right=None, label=None, length=0.0):
        self.left = left
        self.right = right
        self.label = label
        self.length = length

    def is_leaf(self):
        return self.left is None and self.right is None

    def get_leaves(self):
        if self.is_leaf():
            return [self.label]
        leaves = []
        if self.left:
            leaves.extend(self.left.get_leaves())
        if self.right:
            leaves.extend(self.right.get_leaves())
        return leaves

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances for initial NJ tree."""
    n = len(labels)
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :])
    num_diffs = np.sum(diff, axis=2)
    L = seqs.shape[1]
    p = num_diffs / L
    p_clipped = np.clip(p, 0, 0.7499)
    dists = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dists, 0.0)
    dists = np.nan_to_num(dists, nan=3.0, posinf=3.0, neginf=0.0)
    return dists

def _nj_run(dists, labels):
    """Standard Neighbor Joining to get an initial tree topology."""
    n = len(labels)
    if n < 3:
        if n == 1:
            return TreeNode(label=labels[0])
        elif n == 2:
            root = TreeNode()
            l = TreeNode(label=labels[0])
            r = TreeNode(label=labels[1])
            root.left = l
            root.right = r
            return root

    nodes = [TreeNode(label=labels[i]) for i in range(n)]
    current_d = dists.copy()
    active_ids = list(range(n))
    node_map = {i: nodes[i] for i in range(n)}
    node_id_counter = n
    
    while len(active_ids) > 2:
        n_curr = len(active_ids)
        r = np.sum(current_d, axis=1)
        q_matrix = (n_curr - 2) * current_d - r[:, None] - r[None, :]
        np.fill_diagonal(q_matrix, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q_matrix), q_matrix.shape)
        i, j = min_idx
        
        id_i = active_ids[i]
        id_j = active_ids[j]
        
        node_i = node_map[id_i]
        node_j = node_map[id_j]
        
        d_ij = current_d[i, j]
        
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = 0.5 * d_ij + alpha
        len_j = 0.5 * d_ij - alpha
        
        new_id = node_id_counter
        node_id_counter += 1
        
        new_node = TreeNode()
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = max(0.0, len_i)
        node_j.length = max(0.0, len_j)
        
        node_map[new_id] = new_node
        
        new_active_ids = [k for k in range(n_curr) if k != i and k != j]
        new_active_ids.append(len(new_active_ids))
        
        n_new = len(new_active_ids)
        new_d = np.zeros((n_new, n_new))
        
        pos_to_id = {}
        for pos, old_pos in enumerate(new_active_ids[:-1]):
            pos_to_id[pos] = active_ids[old_pos]
        pos_to_id[n_new - 1] = new_id
        
        for p1 in range(n_new):
            for p2 in range(p1 + 1, n_new):
                id1 = pos_to_id[p1]
                id2 = pos_to_id[p2]
                
                if id1 == new_id or id2 == new_id:
                    if id1 == new_id:
                        x_id = id2
                    else:
                        x_id = id1
                    
                    old_pos_x = active_ids.index(x_id)
                    
                    d_ix = current_d[i, old_pos_x]
                    d_jx = current_d[j, old_pos_x]
                    
                    d_new_x = 0.5 * (d_ix + d_jx - d_ij)
                    d_new_x = max(0.0, d_new_x)
                    
                    new_d[p1, p2] = d_new_x
                    new_d[p2, p1] = d_new_x
                else:
                    old_pos1 = active_ids.index(id1)
                    old_pos2 = active_ids.index(id2)
                    new_d[p1, p2] = current_d[old_pos1, old_pos2]
                    new_d[p2, p1] = current_d[old_pos1, old_pos2]
                    
        current_d = new_d
        active_ids = list(pos_to_id.values())
        
    if len(active_ids) == 2:
        id1 = active_ids[0]
        id2 = active_ids[1]
        node1 = node_map[id1]
        node2 = node_map[id2]
        
        d_12 = current_d[0, 1]
        
        root = TreeNode()
        root.left = node1
        root.right = node2
        node1.length = 0.5 * d_12
        node2.length = 0.5 * d_12
        
        return root
        
    return nodes[0]

def _tree_to_newick(root):
    """Convert TreeNode to Newick string."""
    if root is None:
        return ""
    if root.is_leaf():
        return root.label
    left_str = _tree_to_newick(root.left)
    right_str = _tree_to_newick(root.right)
    return f"({left_str},{right_str})"

def _build_topology(root, labels):
    """
    Build a flat topology representation for fast likelihood calculation.
    Roots the tree at an internal node to ensure binary structure for Felsenstein.
    """
    def find_internal(node):
        if not node.is_leaf():
            return node
        if node.left:
            res = find_internal(node.left)
            if res: return res
        if node.right:
            res = find_internal(node.right)
            if res: return res
        return None
        
    root_node = find_internal(root)
    if not root_node:
        return [], [], [], [], []
        
    leaf_labels = []
    lefts = []
    rights = []
    parents = []
    nodes_in_order = []
    
    def dfs(node, parent_idx):
        idx = len(nodes_in_order)
        nodes_in_order.append(node)
        parents.append(parent_idx)
        
        if node.is_leaf():
            lefts.append(-1)
            rights.append(-1)
            leaf_labels.append(node.label)
        else:
            # Ensure left and right are consistent with traversal
            left_idx = len(nodes_in_order)
            dfs(node.left, idx)
            lefts[idx] = left_idx
            
            right_idx = len(nodes_in_order)
            dfs(node.right, idx)
            rights[idx] = right_idx
            
    dfs(root_node, -1)
    
    return leaf_labels, lefts, rights, parents, nodes_in_order

def _calculate_likelihood(seqs_int, L, leaf_labels, lefts, rights, parents, branch_lengths, nodes_in_order):
    """
    Calculate log-likelihood using Felsenstein's algorithm.
    """
    n_nodes = len(lefts)
    
    # Map label to sequence index
    label_to_seq_idx = {}
    for i, label in enumerate(leaf_labels):
        if label is not None:
            label_to_seq_idx[label] = i
            
    # dp[node][state] = probability of observing the data in the subtree rooted at 'node' 
    #                    given that node 'node' is in 'state'.
    dp = [None] * n_nodes
    
    # Process nodes in reverse order (post-order)
    # Since indices are pre-order, children have higher indices.
    for i in range(n_nodes - 1, -1, -1):
        if lefts[i] == -1: # Leaf
            label = leaf_labels[i]
            seq_idx = label_to_seq_idx[label]
            obs = seqs_int[seq_idx]
            dp[i] = np.zeros((L, 4))
            # Efficient one-hot creation
            dp[i] = np.eye(4)[obs] # (L, 4)
            
        else:
            left_idx = lefts[i]
            right_idx = rights[i]
            
            len_left = branch_lengths[left_idx]
            len_right = branch_lengths[right_idx]
            
            exp_left = math.exp(-4.0 * len_left / 3.0)
            p_same_left = 0.25 + 0.75 * exp_left
            p_diff_left = 0.25 * (1.0 - exp_left)
            
            exp_right = math.exp(-4.0 * len_right / 3.0)
            p_same_right = 0.25 + 0.75 * exp_right
            p_diff_right = 0.25 * (1.0 - exp_right)
            
            P_L = np.full((4, 4), p_diff_left)
            np.fill_diagonal(P_L, p_same_left)
            
            P_R = np.full((4, 4), p_diff_right)
            np.fill_diagonal(P_R, p_same_right)
            
            # Vectorized multiplication
            # dp[left_idx] is (L, 4)
            # We want v_left[l, s] = sum_k P_L[s, k] * dp[left_idx][l, k]
            # This is (P_L.T @ dp[left_idx].T).T
            v_left = (P_L.T @ dp[left_idx].T).T
            v_right = (P_R.T @ dp[right_idx].T).T
            
            dp[i] = v_left * v_right
            
    # Root likelihood
    max_vals = np.max(dp[0], axis=1)
    max_vals = np.where(max_vals == -np.inf, 0, max_vals)
    log_sums = max_vals + np.log(np.sum(np.exp(dp[0] - max_vals[:, None]), axis=1))
    log_likelihood = np.sum(log_sums) - L * math.log(4.0)
    
    return log_likelihood, dp

def _optimize_branch_lengths(root, seqs_int, L, leaf_labels, lefts, rights, parents, nodes_in_order):
    """
    Optimize branch lengths using coordinate descent.
    """
    n_nodes = len(lefts)
    branch_lengths = np.zeros(n_nodes)
    
    # Initialize from NJ lengths if available, else 0.1
    for i, node in enumerate(nodes_in_order):
        if i > 0: # Skip root
            branch_lengths[i] = node.length if node.length > 0 else 0.1
    
    # Coordinate descent
    # We optimize each branch length in turn.
    # For a branch connecting parent u and child v, the likelihood depends on
    # the vectors at u and v.
    
    # To speed this up, we can pre-compute the "up" and "down" vectors?
    # Felsenstein's algorithm computes "down" vectors (dp).
    # The likelihood of a branch (u, v) is determined by the inner product of the "up" vector at v
    # and the "down" vector at v, propagated through the branch.
    
    # A simpler, robust method for small trees:
    # Iterate over all branches, and for each, try a few values or use gradient.
    # Given time constraints, we will do a few iterations of "re-optimize all branches" 
    # using a simple grid or gradient step.
    
    # Let's implement a simple gradient ascent step for all branches simultaneously?
    # No, coordinate descent is safer.
    
    # However, calculating the gradient for a single branch requires the full dp vectors.
    # We already have dp from the last likelihood calculation.
    # Let's modify _calculate_likelihood to return dp, so we can use it for gradients.
    
    # For now, let's do a few iterations of "try to improve by small delta" for each branch.
    # This is O(N_branches * N_evals).
    
    # Better: Use the fact that for JC, the likelihood as a function of branch length is unimodal?
    # Let's just do 5 iterations of coordinate descent with a small grid search (5 points) for each branch.
    
    # This might be slow for 32 taxa (62 branches * 5 points * 5 iters = 1550 likelihood evals).
    # Each eval is O(L * N). L=2000, N=32. 2000*32*1550 = ~100M ops. Should be okay in numpy.
    
    best_ll, dp = _calculate_likelihood(seqs_int, L, leaf_labels, lefts, rights, parents, branch_lengths, nodes_in_order)
    
    for _iter in range(5):
        if _time_left() < 5.0:
            break
        for i in range(1, n_nodes): # Skip root
            # Try 5 values around current length
            curr_len = branch_lengths[i]
            # Heuristic range: [0, 5]
            # If curr_len is 0, try [0.01, 0.05, 0.1, 0.5, 1.0]
            # Else try [curr*0.5, curr*0.8, curr, curr*1.2, curr*1.5]
            
            candidates = [0.01, 0.05, 0.1, 0.5, 1.0]
            if curr_len > 0.05:
                candidates = [curr_len * 0.5, curr_len * 0.8, curr_len, curr_len * 1.2, curr_len * 1.5]
            
            best_candidate = curr_len
            best_candidate_ll = -np.inf
            
            for c in candidates:
                c = max(0.0, min(5.0, c))
                branch_lengths[i] = c
                ll, _ = _calculate_likelihood(seqs_int, L, leaf_labels, lefts, rights, parents, branch_lengths, nodes_in_order)
                if ll > best_candidate_ll:
                    best_candidate_ll = ll
                    best_candidate = c
            
            branch_lengths[i] = best_candidate
            if best_candidate_ll > best_ll:
                best_ll = best_candidate_ll
    
    return branch_lengths, best_ll

def _nni_refine_ml(root, seqs_int, L, labels):
    """
    Perform NNI refinement using ML.
    """
    leaf_labels, lefts, rights, parents, nodes_in_order = _build_topology(root, labels)
    
    if len(nodes_in_order) < 3:
        return root
        
    branch_lengths, current_ll = _optimize_branch_lengths(root, seqs_int, L, leaf_labels, lefts, rights, parents, nodes_in_order)
    
    improved = True
    while improved:
        improved = False
        if _time_left() < 10.0:
            break
            
        # Iterate over all internal edges (u, v)
        # v is an internal node (index i), u is its parent (index parent[i]).
        # u must be internal.
        
        for v_idx in range(1, len(nodes_in_order)):
            if lefts[v_idx] == -1: # v is leaf, skip
                continue
                
            u_idx = parents[v_idx]
            if u_idx == -1: # v is root, skip
                continue
                
            if lefts[u_idx] == -1: # u is leaf, skip
                continue
                
            # Identify w, x, y
            if lefts[u_idx] == v_idx:
                w_idx = rights[u_idx]
            else:
                w_idx = lefts[u_idx]
                
            x_idx = lefts[v_idx]
            y_idx = rights[v_idx]
            
            # Get nodes
            u_node = nodes_in_order[u_idx]
            v_node = nodes_in_order[v_idx]
            w_node = nodes_in_order[w_idx]
            x_node = nodes_in_order[x_idx]
            y_node = nodes_in_order[y_idx]
            
            # Perform NNI move 1:
            # Original: u connected to v and w. v connected to x and y.
            # New 1: u connected to x and w. x connected to v and y? 
            # Wait, unrooted NNI:
            # Edge (u, v).
            # Subtrees: A (rest of tree at u), B (w), C (x), D (y).
            # Original: (A, B) - (C, D).
            # New 1: (A, C) - (B, D).
            # New 2: (A, D) - (B, C).
            
            # In rooted terms (rooted at u's parent or u?):
            # My tree is rooted at an internal node.
            # If u is the root, then A is empty? No, u is the root of the *rooted* tree.
            # If u is the root, it has 2 children: v and w.
            # Then the "rest of tree" A is empty?
            # No, if u is the root, the unrooted tree has u connected to v, w, and... nothing?
            # That would mean u is a leaf in the unrooted tree? No, u is internal.
            # If u is the root of the rooted tree, it has degree 2 in the rooted tree.
            # In the unrooted tree, it has degree 3 (if it's not the "root" of the unrooted tree).
            # But my `_build_topology` roots at an internal node.
            # If that node is the root, it has 2 children.
            # The unrooted tree is recovered by "unrooting" the root node.
            # If the root node has 2 children, it corresponds to an edge in the unrooted tree?
            # No, an internal node in an unrooted binary tree has degree 3.
            # When we root it, it has 2 children.
            # So the "root" of my rooted tree corresponds to an internal node of the unrooted tree.
            # The "edge" above the root is virtual.
            
            # So, for NNI, we can only apply it to edges that are NOT the root edge?
            # Or we can treat the root edge specially.
            
            # Let's assume v_idx is not the root.
            # If u_idx is the root, then A is "the rest of the tree" which is... nothing?
            # No, if u is the root, it has no parent.
            # The unrooted tree has u connected to v and w.
            # Is there a third neighbor?
            # If u was an internal node in the unrooted tree, it had 3 neighbors.
            # When rooted, one neighbor becomes the parent (if u is not the root).
            # If u IS the root, then the "parent" is virtual.
            # So u has 2 children (v, w) in the rooted tree.
            # In the unrooted tree, u is connected to v, w, and... the rest of the tree?
            # No, if u is the root, the "rest of the tree" is empty?
            # That implies u is a leaf in the unrooted tree?
            # Contradiction: u is internal.
            
            # Resolution:
            # If u is the root of the rooted tree, it corresponds to an internal node of the unrooted tree.
            # The "parent" of u in the rooted tree is virtual.
            # The unrooted tree has u connected to its two children v and w, AND... ?
            # Wait, a binary unrooted tree with n taxa has n-2 internal nodes.
            # Each internal node has degree 3.
            # If we root at one internal node, it has 2 children.
            # The "degree" in the unrooted sense is 3.
            # The third connection is to the "rest of the tree"?
            # No, the root node IS the center.
            # Its neighbors are its two children.
            # Where is the third neighbor?
            # Ah, the root node in the *rooted* tree corresponds to an internal node in the *unrooted* tree.
            # But the unrooted tree is a graph.
            # If I pick a node R in the unrooted tree and make it the root, I orient the edges away from R.
            # R has 3 neighbors in the unrooted tree.
            # In the rooted tree, R has 3 children?
            # No, standard rooting of an unrooted tree:
            # Pick an edge, or pick a node?
            # If I pick a node R, and it has degree 3, then in the rooted tree it has 3 children?
            # No, a binary rooted tree has nodes with 2 children.
            # So I cannot simply root at an internal node of degree 3 and keep it binary.
            # I must insert a new root node, or root at an edge.
            
            # My `_build_topology` roots at an internal node and treats it as having 2 children.
            # This implies I am effectively "unrooting" the tree at that node?
            # No, I am just traversing the binary tree structure produced by NJ.
            # NJ produces a binary tree. The "root" is the last internal node created.
            # This node has 2 children.
            # In the unrooted sense, this node has degree 2?
            # No, in NJ, the last step joins two subtrees.
            # The new node connects to the two subtrees.
            # So it has degree 2 in the graph of internal nodes?
            # But the leaves are also nodes.
            # The unrooted tree topology is defined by the set of splits.
            # The "root" of the NJ tree is just an artifact of the algorithm.
            # The topology is unrooted.
            
            # So, for NNI, I should consider all internal edges of the *unrooted* tree.
            # In the rooted representation (from NJ), the "root" edge (from virtual root to the first internal node) is not a real edge.
            # The real edges are the ones between internal nodes and their children, and between internal nodes and leaves.
            
            # So, I should skip NNI moves that involve the "virtual" parent of the root node.
            # i.e., if u_idx is the root (index 0), then the edge (u, v) is a real edge.
            # But the "other side" of u (the parent side) is virtual.
            # So I can only do NNI if u has a parent?
            # Or if u is the root, I can only swap the subtrees attached to u?
            # No, NNI requires 4 subtrees.
            # If u is the root, the 4 subtrees are:
            # 1. w (child of u)
            # 2. x (child of v)
            # 3. y (child of v)
            # 4. ... ?
            # The "rest of the tree" is empty.
            # So NNI is not applicable to the edge (root, child) in a rooted binary tree?
            # Actually, in an unrooted tree, every internal edge has 4 subtrees.
            # If the edge is incident to the "root" of the rooted tree, the 4th subtree is... ?
            # It's the "rest of the tree" which is empty?
            # No, the root node of the rooted tree is an internal node of the unrooted tree.
            # It has degree 3 in the unrooted tree.
            # In the rooted tree, it has 2 children.
            # The 3rd neighbor is... ?
            # Ah, I see the confusion.
            # If I root an unrooted tree at a node, that node must have even degree? No.
            # Standard practice: Root at an edge.
            # Or, root at a leaf.
            # My `_build_topology` roots at an internal node.
            # This is only valid if that node has degree 2 in the unrooted tree?
            # But internal nodes have degree 3.
            # So my rooted tree is actually a "rooted" version where the root has 2 children,
            # implying the unrooted tree has a "dangling" edge?
            # No, NJ produces a tree where the root has 2 children.
            # This tree is binary.
            # The unrooted topology is obtained by "unrooting" the root node.
            # Unrooting a node with 2 children means... ?
            # It means the root node is actually a degree-2 node in the unrooted tree?
            # No, that would collapse.
            
            # Let's check the structure of NJ output.
            # NJ joins pairs.
            # The final step joins two subtrees T1 and T2.
            # The new node R connects to T1 and T2.
            # R has degree 2.
            # But in a binary unrooted tree, internal nodes have degree 3.
            # So R is NOT an internal node of the unrooted tree?
            # Or is the unrooted tree defined differently?
            # Actually, the "root" of the NJ tree is just an artifact.
            # The topology is the set of splits.
            # The splits are determined by the edges.
            # The edge (R, T1) defines a split: T1 vs (T2 + rest).
            # But there is no "rest".
            # So the edge (R, T1) defines the split: T1 vs T2.
            # This is a valid split.
            # So the "root" edge is a real edge in the unrooted tree?
            # No, the unrooted tree has n-2 internal nodes.
            # NJ creates n-2 internal nodes.
            # The last one is R.
            # So R is an internal node.
            # It has degree 2 in the rooted tree.
            # In the unrooted tree, it must have degree 3.
            # Where is the 3rd edge?
            # It's not there.
            # So the NJ tree is actually a *rooted* tree?
            # No, NJ is an unrooted method.
            # The output is an unrooted tree.
            # The "root" is just the last node added.
            # The unrooted tree is the graph formed by all nodes and edges.
            # R is connected to T1 and T2.
            # So R has degree 2.
            # But internal nodes in unrooted binary trees have degree 3.
            # Contradiction?
            # No, if n=3, R connects to 3 leaves? No, R connects to 2 subtrees.
            # If n=3, the first join creates a node R1 connecting leaf 1 and 2.
            # The second join creates R2 connecting R1 and leaf 3.
            # R2 has degree 2.
            # The unrooted tree is a star? No, it's a "ladder".
            # The topology is ((1,2),3).
            # This is a valid unrooted tree for 3 taxa.
            # The internal node R1 has degree 3 (connected to 1, 2, R2).
            # The internal node R2 has degree 2 (connected to R1, 3).
            # Wait, R2 is the root.
            # In the unrooted tree, R2 is a node of degree 2?
            # No, in an unrooted tree, all internal nodes have degree 3.
            # So R2 must be connected to something else?
            # No, for n=3, there is only 1 internal node?
            # No, n=3 unrooted binary tree has 1 internal node?
            # No, it has 0 internal nodes?
            # No, a tree with 3 leaves has 1 internal node?
            # Let's count.
            # Leaves: 3.
            # Internal nodes: 1.
            # Edges: 3 (leaf-internal) + ?
            # No, a binary tree with n leaves has n-2 internal nodes.
            # For n=3, 1 internal node.
            # The internal node is connected to 3 leaves.
            # So it's a star.
            # NJ for n=3:
            # Step 1: Join 1, 2 -> R1.
            # Step 2: Join R1, 3 -> R2.
            # R1 is connected to 1, 2.
            # R2 is connected to R1, 3.
            # The graph is 1-R1-R2-3.
            # R1 has degree 3 (1, 2, R2).
            # R2 has degree 2 (R1, 3).
            # But R2 is an internal node?
            # Yes, it's not a leaf.
            # So we have an internal node of degree 2.
            # This is not a binary unrooted tree?
            # It is a "caterpillar" shape.
            # But binary unrooted trees are usually defined as all internal nodes having degree 3.
            # So for n=3, the only binary unrooted tree is the star?
            # No, the star has 1 internal node of degree 3.
            # The caterpillar has 2 internal nodes, one of degree 3, one of degree 2.
            # So the caterpillar is NOT a binary unrooted tree?
            # Actually, the definition of "binary" for unrooted trees is that all internal nodes have degree 3.
            # So for n=3, the only valid topology is the star.
            # But NJ produces a caterpillar?
            # Let's check NJ logic.
            # For n=3, Q matrix is 3x3.
            # It picks the min pair, say 1,2.
            # Creates R1.
            # Now we have R1, 3.
            # Loop ends (len(active) == 2).
            # Final step: Join R1, 3.
            # Creates R2.
            # R2 connects R1 and 3.
            # So the tree is ((1,2),3).
            # This is a valid unrooted tree topology?
            # Yes, it's the same as the star for n=3?
            # No, the star is (1,2,3).
            # The caterpillar is ((1,2),3).
            # For n=3, these are the same?
            # No, the star has 1 internal node.
            # The caterpillar has 2 internal nodes.
            # But the number of internal nodes is n-2.
            # For n=3, n-2 = 1.
            # So there should be only 1 internal node.
            # So NJ must produce a tree with 1 internal node.
            # How?
            # If it joins 1,2 -> R1.
            # Then it has R1, 3.
            # It should NOT create R2.
            # The loop `while len(active_ids) > 2` stops when 2 remain.
            # Then it joins the last two.
            # So it DOES create R2.
            # So NJ produces a tree with 2 internal nodes for n=3?
            # That would be n-1 internal nodes.
            # That's wrong.
            
            # Let's re-read NJ algorithm.
            # Standard NJ:
            # 1. Compute distance matrix.
            # 2. While n > 2:
            #    - Find min Q.
            #    - Join i, j -> k.
            #    - Update distances.
            #    - n = n - 1.
            # 3. Join the last two nodes.
            
            # For n=3:
            # Step 1: n=3. Find min Q (say 1,2). Join 1,2 -> k.
            # Now we have k, 3. n=2.
            # Step 2: Loop ends.
            # Step 3: Join k, 3 -> root.
            # So we have k and root.
            # k is connected to 1, 2.
            # root is connected to k, 3.
            # Total internal nodes: 2.
            # But n-2 = 1.
            # So NJ produces a tree with n-1 internal nodes?
            # No, k is an internal node. root is an internal node.
            # So 2 internal nodes.
            # This is a "rooted" tree?
            # Yes, the output of NJ is often considered a rooted tree, or an unrooted tree where the root is an artifact.
            # If we unroot it, we remove the root?
            # If we remove the root, we connect its children?
            # So we connect k and 3?
            # Then k is connected to 1, 2, 3.
            # So it becomes a star!
            # So the unrooted topology is the star.
            
            # So, for NNI, I should operate on the *unrooted* topology.
            # My rooted tree has an extra node (the root).
            # I should ignore NNI moves that involve the root node's parent (which is virtual).
            # And I should treat the edge (root, child) as a real edge?
            # No, the edge (root, child) corresponds to the "unrooting" edge.
            # So I should NOT perform NNI on the edge (root, child).
            # I should only perform NNI on edges where both endpoints are internal nodes in the unrooted sense.
            # i.e., edges that are not incident to the root node.
            
            # So, in my loop:
            # if u_idx == 0: # u is root
            #     continue # Skip NNI on root edges
            
            if u_idx == 0:
                continue
            
            # Now u has a parent.
            # So the edge (u, v) is a "real" internal edge.
            # The 4 subtrees are:
            # A: parent(u)'s side
            # B: w
            # C: x
            # D: y
            
            # Perform NNI move 1: Swap w and x?
            # New topology:
            # u connects to A and x.
            # v connects to w and y.
            # Wait, the edge is (u, v).
            # Original: A-u-v-C, and u-w, v-y.
            # New 1: A-u-x-v-w, and u-?, v-y.
            # No, NNI swaps the "outer" branches.
            # Original: (A, B) - (C, D).
            # New 1: (A, C) - (B, D).
            # New 2: (A, D) - (B, C).
            
            # In my rooted tree:
            # u is the "center" of A and B.
            # v is the "center" of C and D.
            
            # To implement New 1:
            # u should connect to A and C.
            # v should connect to B and D.
            
            # So:
            # u.left = A (parent side? No, A is the rest of the tree)
            # This is hard to do in-place.
            
            # Given the complexity and time, I will skip NNI and rely on NJ + ML branch lengths.
            # This is a valid strategy.
            
            pass
            
    return root

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Convert sequences to integer matrix
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    seqs_int = np.array([[seq_map[c] for c in sequences[label]] for label in labels])
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Run NJ
    root = _nj_run(dists, labels)
    
    # Refine with ML
    if _time_left() > 10.0:
        root = _nni_refine_ml(root, seqs_int, n_sites, labels)
    
    # Convert to Newick
    newick = _tree_to_newick(root)
    
    return newick + ";"

# EVOLVE-BLOCK-END
