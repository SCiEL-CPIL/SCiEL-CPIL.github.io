# EVOLVE-BLOCK-START
import time
import math
import numpy as np

_START = None
_DEADLINE_FRACTION = 0.85

def _now():
    return time.time()

def _init_time():
    global _START
    if _START is None:
        _START = time.time()

def _time_left():
    _init_time()
    return max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START))

class TreeNode:
    def __init__(self, name=None, length=0.0):
        self.name = name
        self.length = length
        self.children = []
        self.parent = None
        self.id = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)
        return child

    def is_leaf(self):
        return len(self.children) == 0

    def to_newick(self):
        if self.is_leaf():
            s = self.name
        else:
            s = "(" + ",".join(c.to_newick() for c in self.children) + ")"
        if self.length > 0:
            s += f":{self.length:.4f}"
        return s

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.int8)
    return labels, arr

def _jc_matrix(arr):
    n, L = arr.shape
    diff = (arr[:, None, :] != arr[None, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    p_safe = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1 - (4.0/3.0) * p_safe)
    np.fill_diagonal(d, 0)
    return d

def _nj(dist, labels):
    n = len(labels)
    if n == 1:
        return TreeNode(labels[0])
    if n == 2:
        root = TreeNode()
        root.add_child(TreeNode(labels[0]))
        root.add_child(TreeNode(labels[1]))
        return root

    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: TreeNode(labels[i]) for i in range(n)}
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    while len(nodes) > 3:
        n_curr = len(nodes)
        total = {a: sum(d[a][b] for b in nodes if b != a) for a in nodes}
        
        best_criterion = float('inf')
        best_a, best_b = -1, -1
        
        for i, a in enumerate(nodes):
            for b in nodes[i+1:]:
                criterion = (n_curr - 2) * d[a][b] - total[a] - total[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a, best_b = a, b
        
        a, b = best_a, best_b
        
        # Create new node
        new_id = -1 - len(nodes)
        new_node = TreeNode()
        
        # Calculate branch lengths
        # l_ab = 0.5 * d_ab
        # l_a_new = 0.5 * (d_ab + (1/(n-2)) * (sum_a - sum_b))
        # l_b_new = 0.5 * (d_ab - (1/(n-2)) * (sum_a - sum_b))
        
        d_ab = d[a][b]
        sum_a = total[a]
        sum_b = total[b]
        
        l_a = 0.5 * (d_ab + (sum_a - sum_b) / (n_curr - 2))
        l_b = 0.5 * (d_ab - (sum_a - sum_b) / (n_curr - 2))
        
        # Ensure positive lengths (NJ can produce negative ones)
        if l_a < 0: l_a = 0.0
        if l_b < 0: l_b = 0.0
        
        trees[a].length = l_a
        trees[b].length = l_b
        new_node.add_child(trees[a])
        new_node.add_child(trees[b])
        trees[new_id] = new_node
        
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    # Final 3 nodes
    # Connect them
    # Find the 3 nodes
    n1, n2, n3 = nodes
    # Calculate lengths to a central node? Or just make a 3-star?
    # NJ usually joins the last 3 into a central node.
    # d12, d13, d23
    d12 = d[n1][n2]
    d13 = d[n1][n3]
    d23 = d[n2][n3]
    
    l1 = 0.5 * (d12 + d13 - d23)
    l2 = 0.5 * (d12 + d23 - d13)
    l3 = 0.5 * (d13 + d23 - d12)
    
    if l1 < 0: l1 = 0.0
    if l2 < 0: l2 = 0.0
    if l3 < 0: l3 = 0.0
    
    center = TreeNode()
    trees[n1].length = l1
    trees[n2].length = l2
    trees[n3].length = l3
    center.add_child(trees[n1])
    center.add_child(trees[n2])
    center.add_child(trees[n3])
    
    return center

def _root_tree(root):
    """Root the tree at a leaf to make it binary for Felsenstein."""
    # Find a leaf
    def find_leaf(node):
        if node.is_leaf():
            return node
        return find_leaf(node.children[0])
    
    leaf = find_leaf(root)
    
    # Re-root at this leaf
    # This is tricky with the current structure.
    # Alternative: Just traverse from the root we have.
    # Felsenstein's algorithm works on a rooted tree.
    # If we have an unrooted tree represented as a rooted tree (with a 3-way split at the root),
    # we can treat the root as the root for likelihood calculation.
    # The likelihood of an unrooted tree is the same as the likelihood of any rooting of it,
    # provided we sum over the state at the root (or fix it and average, but summing is better).
    
    # So, we just use the `root` we have.
    # We need to calculate P(Data | Tree).
    # This involves summing over the states at the root.
    pass

def _calc_likelihood(root, data):
    """
    Calculate log-likelihood of the tree given the data under JC model.
    data: NumPy array of shape (n, L) with values 0,1,2,3.
    root: TreeNode representing the tree.
    """
    n = data.shape[0]
    L = data.shape[1]
    
    # Map leaf names to indices
    leaf_names = []
    def collect_leaves(node):
        if node.is_leaf():
            leaf_names.append(node.name)
        else:
            for c in node.children:
                collect_leaves(c)
    collect_leaves(root)
    
    name_to_idx = {name: i for i, name in enumerate(leaf_names)}
    
    # Precompute JC transition probabilities
    # P(t) matrix 4x4
    # We need P for each branch length.
    
    def jc_probs(length):
        if length <= 0:
            return np.eye(4)
        u = math.exp(-4.0 * length / 3.0)
        p_off = 0.25 * (1.0 - u)
        p_on = 0.25 * (1.0 + 3.0 * u)
        # Matrix:
        # [p_on, p_off, p_off, p_off]
        # [p_off, p_on, p_off, p_off]
        # ...
        m = np.full((4, 4), p_off)
        np.fill_diagonal(m, p_on)
        return m

    # Post-order traversal to compute conditional probabilities
    # alpha(node, site) = vector of 4 probs
    # For leaf: alpha = one-hot vector of the observed base
    # For internal: alpha = sum_children (alpha_child @ P_child_length)
    
    # We process site by site to save memory? Or vectorize?
    # n is small, L is up to 2000.
    # Vectorizing over sites is better.
    
    # Let's store for each node a matrix of shape (4, L)
    # alpha[node] = (4, L)
    
    alpha = {}
    
    def traverse(node):
        if node.is_leaf():
            idx = name_to_idx[node.name]
            # One-hot vectors for all sites
            # data[idx] is (L,)
            vec = np.zeros((4, L), dtype=np.float64)
            vec[data[idx], np.arange(L)] = 1.0
            alpha[node] = vec
        else:
            # Process children
            for c in node.children:
                traverse(c)
            
            # Sum over children
            # alpha_node = sum ( alpha_child @ P_child )
            # alpha_child is (4, L), P_child is (4, 4)
            # (4, L) @ (4, 4) is not directly compatible.
            # We need (alpha_child.T @ P_child).T ?
            # alpha_child[i, s] = P(x_i=s | ...)? No.
            # Let's define alpha[node, i, s] = P(Data_subtree | x_node = i, site s)
            # For leaf: alpha[i, s] = 1 if i == data[idx, s] else 0.
            # For internal: alpha[i, s] = sum_j ( P(j -> i | length) * alpha_child[j, s] )
            # P(j -> i | length) is element (j, i) of the transition matrix?
            # Standard convention: P_ij = P(x_t = j | x_0 = i).
            # So P(j -> i) is P_ij.
            # My `jc_probs` returns P_ij where row is from, col is to.
            # So we need alpha_child[j, s] * P[j, i].
            # This is a matrix multiplication: alpha_child (4, L) and P (4, 4).
            # We want result[i, s] = sum_j alpha_child[j, s] * P[j, i].
            # This is (alpha_child.T @ P.T).T ?
            # alpha_child.T is (L, 4). P.T is (4, 4).
            # (L, 4) @ (4, 4) = (L, 4).
            # So (alpha_child.T @ P.T).T is (4, L).
            # Let's check:
            # Res[i, s] = sum_k (alpha_child.T[s, k] * P.T[k, i]) = sum_k alpha_child[k, s] * P[i, k].
            # This gives sum_k alpha_child[k, s] * P[i, k].
            # We wanted sum_j alpha_child[j, s] * P[j, i].
            # So we need P[j, i], which is P.T[i, j].
            # So yes, (alpha_child.T @ P.T).T is correct.
            
            sum_alpha = np.zeros((4, L), dtype=np.float64)
            for c in node.children:
                P = jc_probs(c.length)
                # sum_alpha += (alpha[c].T @ P.T).T
                sum_alpha += (alpha[c].T @ P.T).T
            
            alpha[node] = sum_alpha
        
    traverse(root)
    
    # Likelihood at root
    # L(Data) = sum_i alpha[root, i] (summed over sites)
    # alpha[root] is (4, L)
    # Log-likelihood = sum_s log( sum_i alpha[root, i, s] )
    
    root_alpha = alpha[root]
    # Sum over states (axis 0)
    probs = np.sum(root_alpha, axis=0) # (L,)
    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-300, 1.0)
    log_lik = np.sum(np.log(probs))
    
    return log_lik

def _nni_refine(root, data, max_rounds=5):
    """
    Refine tree using NNI moves.
    """
    def get_internal_edges(node):
        # Returns list of (node, child) pairs where both are internal
        # Or rather, edges that can be NNI'd.
        # An edge is NNI-able if it connects two internal nodes.
        edges = []
        def traverse(n):
            if n.is_leaf(): return
            for c in n.children:
                if not c.is_leaf():
                    edges.append((n, c))
                traverse(c)
        traverse(node)
        return edges

    def perform_nni(node, child):
        """
        Perform NNI on edge (node, child).
        Returns the new tree if it's better, else None.
        """
        # Get the other child of node
        other_children = [c for c in node.children if c != child]
        if len(other_children) != 1:
            return None
        other = other_children[0]
        
        # Get the two children of child
        if len(child.children) != 2:
            return None
        c1, c2 = child.children
        
        # Current topology: ((c1, c2), other) under node
        # NNI alternatives:
        # 1. ((c1, other), c2)
        # 2. ((c2, other), c1)
        
        # We need to rewire.
        # This is complex with the current tree structure.
        # Simplification: Since n is small, we can reconstruct the tree from Newick?
        # No, let's just calculate likelihood of the 3 topologies.
        # To do that, we need to temporarily modify the tree.
        
        # Save original structure
        # node.children = [child, other]
        # child.children = [c1, c2]
        
        # Topology 1: Original
        # (Already have it)
        
        # Topology 2: ((c1, other), c2)
        # We need to create a new tree structure or modify in place and revert.
        
        # In-place modification is risky but fast.
        # Let's try modifying in place.
        
        # Save lengths
        len_node_child = child.length
        len_child_c1 = c1.length
        len_child_c2 = c2.length
        len_node_other = other.length
        
        # Modify for Topology 2
        # New edge: node - new_internal - c1
        # New edge: new_internal - other
        # New edge: node - c2 (Wait, no)
        
        # Structure:
        #       node
        #      /    \
        #    child  other
        #    /   \
        #   c1    c2
        
        # NNI 1:
        #       node
        #      /    \
        #    new    c2
        #    /  \
        #  c1   other
        
        # Branch lengths:
        # In NNI, we usually keep the same set of branch lengths but reassign them?
        # Or optimize them?
        # For simplicity, let's keep the lengths of the "outer" branches (node-external) same,
        # and the "inner" branches (leaf-internal) same.
        # The lengths of the edges involved in the swap might need adjustment.
        # A simple heuristic: Keep all lengths fixed.
        
        # Let's implement the swap.
        
        # 1. Detach c1, c2, other
        # node.children.remove(child)
        # node.children.remove(other)
        # child.children.remove(c1)
        # child.children.remove(c2)
        
        # 2. Re-attach
        # new_node = child (reuse)
        # new_node.children = [c1, other]
        # node.children = [new_node, c2]
        
        # Update lengths?
        # For now, just keep old lengths.
        
        # Calculate likelihood
        # We need to re-run _calc_likelihood.
        
        # This is getting complex to implement robustly in a short time.
        # Alternative: Just return the NJ tree.
        # The NJ tree is usually good enough for this score.
        # The bootstrap consensus in the original code was the "refinement".
        # Let's stick to the bootstrap consensus but improve the selection metric.
        # Or, just use the full-data NJ tree.
        
        # Given the time, I will stick to the original strategy of generating candidates
        # and picking the best, but I will switch the metric from "Min Mean RF" to "Max Likelihood".
        # This requires computing likelihood for each candidate.
        
        return None

    return root

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    
    d_full = _jc_matrix(arr)
    
    # Generate candidates
    candidates = []
    
    # 1. Full data NJ
    tree_full = _nj(d_full, labels)
    candidates.append(tree_full)
    
    # 2. Bootstrap NJs
    L = arr.shape[1]
    num_bootstraps = 0
    max_bootstraps = 20 # Reduced to allow time for likelihood calc
    
    while num_bootstraps < max_bootstraps and _time_left() > 10.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot = _nj(d_boot, labels)
        candidates.append(tree_boot)
        num_bootstraps += 1
        
    # Select best by Likelihood
    best_tree = candidates[0]
    best_lik = -float('inf')
    
    for tree in candidates:
        try:
            lik = _calc_likelihood(tree, arr)
            if lik > best_lik:
                best_lik = lik
                best_tree = tree
        except Exception:
            continue
            
    return best_tree.to_newick() + ";"

# EVOLVE-BLOCK-END
