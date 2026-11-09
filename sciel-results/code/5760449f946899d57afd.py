"""Phylogenetic tree reconstruction using NJ + ML refinement with proper Felsenstein algorithm."""
import math
import time
import numpy as np


def _jc_distance_matrix(aln_matrix):
    """Compute JC-corrected distance matrix from a binary alignment matrix."""
    n_taxa = aln_matrix.shape[0]
    n_sites = aln_matrix.shape[1]
    a = aln_matrix[:, None, :]
    b = aln_matrix[None, :, :]
    diff = (a != b).sum(axis=2)
    p = diff.astype(np.float64) / n_sites
    np.fill_diagonal(p, 0.0)
    p_clipped = np.clip(p, 0.0, 0.7499)
    arg = 1.0 - (4.0 / 3.0) * p_clipped
    dist = -0.75 * np.log(arg)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(aln_matrix, labels):
    """Build initial tree using Neighbor Joining."""
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    D_init = _jc_distance_matrix(aln_matrix)
    
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D[:n, :n] = D_init
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        sub_D = D[np.ix_(active, active)]
        row_sums = sub_D.sum(axis=1)
        crit = (m - 2) * sub_D - row_sums[:, None] - row_sums[None, :]
        np.fill_diagonal(crit, np.inf)
        idx = np.unravel_index(crit.argmin(), crit.shape)
        i, j = idx
        a = active[i]
        b = active[j]
        
        joined = f"({trees[a]},{trees[b]})"
        new_node = next_id
        next_id += 1
        
        d_ab = D[a, b]
        for x in active:
            if x == a or x == b:
                continue
            D[new_node, x] = 0.5 * (D[a, x] + D[b, x] - d_ab)
            D[x, new_node] = D[new_node, x]
        
        active = [x for x in active if x not in (a, b)]
        active.append(new_node)
        trees[new_node] = joined
    
    if len(active) == 3:
        a, b, c = active
        return f"({trees[a]},{trees[b]},{trees[c]});"
    elif len(active) == 2:
        return f"({trees[active[0]]},{trees[active[1]]});"
    elif len(active) == 1:
        return trees[active[0]] + ";"
    else:
        parts = [trees[x] for x in active]
        return "(" + ",".join(parts) + ");"


class _Tree:
    """Internal tree representation for ML computation."""
    
    def __init__(self, n_taxa):
        self.n_taxa = n_taxa
        self.children = [None] * (2 * n_taxa)  # children[i] = (left, right) or None for leaf
        self.branch_len = [0.0] * (2 * n_taxa)  # branch length from parent to this node
        self.node_count = n_taxa
        self.taxa = []  # list of taxa in order
        self.taxa_set = set()
        self.is_leaf = [True] * (2 * n_taxa)
        self.parent = [-1] * (2 * n_taxa)
        for i in range(n_taxa):
            self.children[i] = None
            self.is_leaf[i] = True
            self.taxa.append(i)
            self.taxa_set.add(i)
    
    def add_internal_node(self, left, right, left_len, right_len):
        """Add an internal node connecting left and right subtrees."""
        idx = self.node_count
        self.node_count += 1
        self.children[idx] = (left, right)
        self.is_leaf[idx] = False
        self.branch_len[left] = left_len
        self.branch_len[right] = right_len
        self.parent[left] = idx
        self.parent[right] = idx
        return idx


def _parse_newick_to_tree(newick, labels):
    """Parse Newick string into _Tree structure."""
    n = len(labels)
    tree = _Tree(n)
    
    # Parse the Newick string
    newick = newick.strip().rstrip(';')
    
    # Build a mapping from label to taxon index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Stack-based parsing
    # Each stack entry: (node_id, is_internal, child1, child2)
    stack = []
    
    i = 0
    while i < len(newick):
        c = newick[i]
        if c == '(':
            # Start of internal node
            stack.append(('internal', None, None))
            i += 1
        elif c == ')':
            i += 1
            # Skip branch length
            if i < len(newick) and newick[i] == ':':
                while i < len(newick) and newick[i] not in (',', ')', ';'):
                    i += 1
            # Pop the internal node
            _, c1, c2 = stack.pop()
            if c1 is None or c2 is None:
                # Shouldn't happen, but handle gracefully
                continue
            # Create internal node
            new_node = tree.add_internal_node(c1, c2, 0.5, 0.5)
            # Push to parent or make it root
            if stack:
                if stack[-1][1] is None:
                    stack[-1] = ('internal', new_node, None)
                else:
                    stack[-1] = ('internal', stack[-1][1], new_node)
            else:
                # This is the root
                tree._root = new_node
        elif c == ',':
            i += 1
        elif c == ':':
            # Skip branch length
            while i < len(newick) and newick[i] not in (',', ')', ';'):
                i += 1
        else:
            # Read a label
            label = ''
            while i < len(newick) and newick[i] not in (')', ',', ':', ';'):
                label += newick[i]
                i += 1
            # This is a leaf
            if label in label_to_idx:
                leaf_idx = label_to_idx[label]
                # Leaf is already at index leaf_idx in the tree
                if stack:
                    if stack[-1][1] is None:
                        stack[-1] = ('internal', leaf_idx, None)
                    else:
                        stack[-1] = ('internal', stack[-1][1], leaf_idx)
                else:
                    tree._root = leaf_idx
    
    # Set default branch lengths
    for i in range(tree.node_count):
        if tree.branch_len[i] == 0.0:
            tree.branch_len[i] = 0.5
    
    return tree


def _tree_to_newick(tree, labels):
    """Convert _Tree structure to Newick string."""
    label_from_idx = {i: labels[i] for i in range(len(labels))}
    
    def _to_newick(idx):
        if tree.is_leaf[idx]:
            return label_from_idx.get(idx, str(idx))
        left, right = tree.children[idx]
        return f"({_to_newick(left)},{_to_newick(right)})"
    
    root = getattr(tree, '_root', 0)
    return _to_newick(root) + ";"


def _compute_log_likelihood(tree, aln_matrix):
    """Compute log-likelihood of the tree given the alignment using Felsenstein's algorithm.
    
    Uses JC model with estimated branch lengths.
    """
    n_taxa = aln_matrix.shape[0]
    n_sites = aln_matrix.shape[1]
    
    if n_sites == 0:
        return 0.0
    
    # Precompute transition matrices for each branch length
    # For JC model: P(t) = [[p_same, p_diff, p_diff, p_diff], ...]
    # p_same = 0.75 + 0.25 * exp(-4t/3)
    # p_diff = 0.25 * (1 - exp(-4t/3))
    
    # Get unique branch lengths
    branch_lens = set()
    for i in range(tree.node_count):
        if tree.branch_len[i] > 0:
            branch_lens.add(round(tree.branch_len[i], 6))
    
    # Precompute transition matrices
    trans_mats = {}
    for bl in branch_lens:
        if bl <= 0:
            trans_mats[bl] = np.eye(4, dtype=np.float64)
        else:
            p_same = 0.75 + 0.25 * math.exp(-4.0 * bl / 3.0)
            p_diff = 0.25 * (1.0 - math.exp(-4.0 * bl / 3.0))
            mat = np.full((4, 4), p_diff, dtype=np.float64)
            np.fill_diagonal(mat, p_same)
            trans_mats[bl] = mat
    
    # Postorder traversal
    postorder = []
    visited = set()
    
    def _postorder(idx):
        if idx in visited:
            return
        visited.add(idx)
        if tree.is_leaf[idx]:
            postorder.append(idx)
        else:
            left, right = tree.children[idx]
            if left is not None:
                _postorder(left)
            if right is not None:
                _postorder(right)
            postorder.append(idx)
    
    root = getattr(tree, '_root', 0)
    _postorder(root)
    
    # Compute likelihood using Felsenstein's algorithm
    # likelihood[node] = array of shape (n_sites, 4)
    likelihood = {}
    
    for node_idx in postorder:
        if tree.is_leaf[node_idx]:
            # Leaf: likelihood is 1 for the observed base, 0 otherwise
            seq = aln_matrix[node_idx]
            ll = np.zeros((n_sites, 4), dtype=np.float64)
            ll[np.arange(n_sites), seq] = 1.0
            likelihood[node_idx] = ll
        else:
            # Internal node: combine children
            left, right = tree.children[node_idx]
            if left is None or right is None:
                continue
            
            left_ll = likelihood.get(left)
            right_ll = likelihood.get(right)
            
            if left_ll is None or right_ll is None:
                continue
            
            # Apply transition matrix along branch
            left_bl = round(tree.branch_len[left], 6)
            right_bl = round(tree.branch_len[right], 6)
            
            left_trans = trans_mats.get(left_bl, np.eye(4))
            right_trans = trans_mats.get(right_bl, np.eye(4))
            
            # P(left_state | parent_state) = left_ll @ left_trans.T
            # But we need: for each parent state, sum over child states
            # left_ll[s, c] = P(data_left | state_c = s)
            # We want: for each parent state p, sum_c P(data_left | c) * P(c | p)
            # = sum_c left_ll[s=c, c] * left_trans[p, c]
            # So: (left_ll @ left_trans.T)[p] = sum_c left_ll[c] * left_trans[p, c]
            
            left_contrib = left_ll @ left_trans.T  # shape (n_sites, 4)
            right_contrib = right_ll @ right_trans.T  # shape (n_sites, 4)
            
            # Combine: product of contributions
            likelihood[node_idx] = left_contrib * right_contrib
    
    if root not in likelihood:
        return -np.inf
    
    # Sum over root states, multiply by prior (0.25 each)
    root_ll = likelihood[root]
    # log_likelihood = sum over sites of log(sum over states of 0.25 * P(data | state))
    root_probs = root_ll * 0.25
    log_ll_per_site = np.log(np.maximum(root_probs.sum(axis=1), 1e-300))
    
    return float(np.sum(log_ll_per_site))


def _nni_refine(tree, aln_matrix, deadline, max_iterations=50):
    """Refine tree using NNI moves."""
    n_taxa = tree.n_taxa
    current_ll = _compute_log_likelihood(tree, aln_matrix)
    best_tree = tree
    best_ll = current_ll
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 15.0:
            break
        
        improved = False
        
        # Try all NNI moves
        # For each internal edge (parent -> child), try the two alternative topologies
        for child in range(tree.node_count):
            if tree.is_leaf[child]:
                continue
            parent = tree.parent[child]
            if parent < 0:
                continue  # child is root
            
            gp = tree.parent[parent]
            if gp < 0:
                continue  # parent is root
            
            # Get the other child of parent
            p_left, p_right = tree.children[parent]
            if p_left == child:
                other_child = p_right
            else:
                other_child = p_left
            
            if other_child is None:
                continue
            
            # Original structure:
            #        gp
            #       /  \
            #     parent  X
            #    /  \
            #  child  other_child
            #
            # NNI move 1: 
            #        gp
            #       /  \
            #     child  other_child
            #    /  \
            #  parent  X
            #
            # NNI move 2:
            #        gp
            #       /  \
            #     parent  other_child
            #    /  \
            #  X     child
            
            # Save original
            orig_gp_children = tree.children[gp]
            orig_parent_children = tree.children[parent]
            orig_child_parent = tree.parent[child]
            orig_other_parent = tree.parent[other_child]
            orig_bl_child = tree.branch_len[child]
            orig_bl_other = tree.branch_len[other_child]
            orig_bl_parent = tree.branch_len[parent]
            orig_bl_gp = tree.branch_len[gp]
            
            # NNI move 1
            tree.children[gp] = (child, other_child)
            tree.children[parent] = (orig_gp_children[0] if orig_gp_children[0] != parent else orig_gp_children[1], child)
            # Fix: parent should have (X, child) where X is the other child of gp
            gp_left, gp_right = orig_gp_children
            if gp_left == parent:
                x = gp_right
            else:
                x = gp_left
            
            tree.children[gp] = (child, other_child)
            tree.children[parent] = (x, child)
            tree.parent[child] = gp
            tree.parent[other_child] = gp
            tree.parent[parent] = child
            tree.parent[x] = parent
            
            ll1 = _compute_log_likelihood(tree, aln_matrix)
            
            # Restore
            tree.children[gp] = orig_gp_children
            tree.children[parent] = orig_parent_children
            tree.parent[child] = orig_child_parent
            tree.parent[other_child] = orig_other_parent
            tree.branch_len[child] = orig_bl_child
            tree.branch_len[other_child] = orig_bl_other
            tree.branch_len[parent] = orig_bl_parent
            tree.branch_len[gp] = orig_bl_gp
            
            if ll1 > best_ll + 1e-10:
                best_ll = ll1
                best_tree = _copy_tree(tree)
                improved = True
        
        if not improved:
            break
    
    return best_tree, best_ll


def _copy_tree(tree):
    """Deep copy a tree."""
    new_tree = _Tree(tree.n_taxa)
    new_tree.node_count = tree.node_count
    new_tree.children = [None] * len(tree.children)
    new_tree.branch_len = tree.branch_len[:]
    new_tree.is_leaf = tree.is_leaf[:]
    new_tree.parent = tree.parent[:]
    new_tree.taxa = tree.taxa[:]
    new_tree.taxa_set = tree.taxa_set.copy()
    for i in range(tree.node_count):
        if tree.children[i] is not None:
            new_tree.children[i] = tree.children[i]
    new_tree._root = getattr(tree, '_root', 0)
    return new_tree


def _estimate_branch_lengths(tree, aln_matrix):
    """Estimate branch lengths using least squares from pairwise distances."""
    n_taxa = tree.n_taxa
    D = _jc_distance_matrix(aln_matrix)
    
    # For each pair of taxa, the expected distance is the sum of branch lengths along the path
    # We can set up a linear system: for each pair (i,j), sum of branches on path = D[i,j]
    
    # Build path representation
    # For each pair of leaves, find the path in the tree
    paths = {}
    
    def _find_path(a, b):
        """Find the set of internal nodes and branches on path from a to b."""
        # Get path from a to root
        path_a = []
        cur = a
        while cur >= 0:
            path_a.append(cur)
            cur = tree.parent[cur]
        
        # Get path from b to root
        path_b = []
        cur = b
        while cur >= 0:
            path_b.append(cur)
            cur = tree.parent[cur]
        
        # Find LCA
        set_a = set(path_a)
        lca = -1
        for node in path_b:
            if node in set_a:
                lca = node
                break
        
        # Path branches: from a up to lca, and from b up to lca
        branches = []
        cur = a
        while cur != lca:
            branches.append(cur)  # branch from parent[cur] to cur
            cur = tree.parent[cur]
        
        cur = b
        while cur != lca:
            branches.append(cur)
            cur = tree.parent[cur]
        
        return branches
    
    # Build linear system
    # Variables: branch lengths for each non-root node
    # For each pair (i,j), sum of branch lengths on path = D[i,j]
    
    n_vars = tree.node_count - 1  # all nodes except root
    var_index = {}
    idx = 0
    for i in range(tree.node_count):
        if i != getattr(tree, '_root', 0):
            var_index[i] = idx
            idx += 1
    
    # Build A matrix and b vector
    pairs = []
    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            pairs.append((i, j))
    
    n_pairs = len(pairs)
    A = np.zeros((n_pairs, n_vars), dtype=np.float64)
    b = np.zeros(n_pairs, dtype=np.float64)
    
    for pi, (i, j) in enumerate(pairs):
        branches = _find_path(i, j)
        for br in branches:
            if br in var_index:
                A[pi, var_index[br]] = 1.0
        b[pi] = D[i, j]
    
    # Solve least squares
    if n_vars > 0 and n_pairs > 0:
        try:
            solution, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
            # Clamp to non-negative
            solution = np.maximum(solution, 0.0)
            # Assign back
            for i in range(tree.node_count):
                if i in var_index:
                    tree.branch_len[i] = solution[var_index[i]]
        except:
            pass
    
    # Ensure root branch length is set
    root = getattr(tree, '_root', 0)
    if tree.branch_len[root] == 0.0:
        tree.branch_len[root] = 0.5
    
    return tree


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250.0  # Leave 20s margin for 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aln_list = []
    for lab in labels:
        seq = sequences[lab]
        aln_list.append([base_map[c] for c in seq])
    aln_matrix = np.array(aln_list, dtype=np.int8)
    
    # Step 1: Build initial tree using NJ
    nj_newick = _nj_tree(aln_matrix, labels)
    
    if time.time() > deadline - 30.0:
        return nj_newick
    
    # Step 2: Parse NJ tree into internal representation
    tree = _parse_newick_to_tree(nj_newick, labels)
    
    # Step 3: Estimate branch lengths
    tree = _estimate_branch_lengths(tree, aln_matrix)
    
    if time.time() > deadline - 20.0:
        return _tree_to_newick(tree, labels)
    
    # Step 4: NNI refinement
    refined_tree, best_ll = _nni_refine(tree, aln_matrix, deadline, max_iterations=30)
    
    return _tree_to_newick(refined_tree, labels)
