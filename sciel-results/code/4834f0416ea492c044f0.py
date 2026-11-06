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

def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
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

def _nj_build(dists, labels):
    """Standard Neighbor Joining. Returns a list of (left_label, right_label, length) tuples
    representing the tree in a flat format, plus a mapping for reconstruction.
    Actually, let's return a Newick string directly or a simple tree structure."""
    
    n = len(labels)
    if n < 3:
        if n == 1:
            return labels[0]
        else:
            return f"({labels[0]},{labels[1]})"

    # We will maintain a list of current "tips" which can be original labels or composite strings
    # But for SPR, we need a more structured representation.
    # Let's use a simple node-based tree for SPR.
    
    class Node:
        __slots__ = ['id', 'label', 'children', 'parent']
        def __init__(self, id, label=None):
            self.id = id
            self.label = label
            self.children = []
            self.parent = None

    nodes = []
    # Initialize leaves
    for i in range(n):
        nodes.append(Node(i, labels[i]))
        
    active_nodes = list(range(n)) # indices into nodes list? No, let's use object references or ids.
    # Let's keep a list of current active node objects.
    current_nodes = [nodes[i] for i in range(n)]
    node_id_counter = n
    
    # Distance matrix for current active nodes
    # d[i][j] is distance between current_nodes[i] and current_nodes[j]
    d = dists.copy()
    
    while len(current_nodes) > 2:
        n_curr = len(current_nodes)
        # Calculate r vector
        r = d.sum(axis=1)
        # Calculate Q matrix
        q = (n_curr - 2) * d - r[:, None] - r[None, :]
        np.fill_diagonal(q, np.inf)
        
        min_idx = np.unravel_index(np.argmin(q), q.shape)
        i, j = min_idx
        
        node_i = current_nodes[i]
        node_j = current_nodes[j]
        
        d_ij = d[i, j]
        alpha = (r[i] - r[j]) / (2.0 * (n_curr - 2))
        len_i = max(0.0, 0.5 * d_ij + alpha)
        len_j = max(0.0, 0.5 * d_ij - alpha)
        
        # Create new internal node
        new_node = Node(node_id_counter)
        node_id_counter += 1
        
        # Connect
        node_i.parent = new_node
        node_j.parent = new_node
        new_node.children = [node_i, node_j]
        
        # Store branch lengths on the child nodes for convenience
        node_i.branch_length = len_i
        node_j.branch_length = len_j
        
        # Update active nodes list
        # Remove i and j, add new_node
        # We need to be careful with indices if we remove from the middle.
        # Let's rebuild the list.
        new_current_nodes = []
        for k in range(n_curr):
            if k != i and k != j:
                new_current_nodes.append(current_nodes[k])
        new_current_nodes.append(new_node)
        
        # Update distance matrix
        n_new = len(new_current_nodes)
        new_d = np.zeros((n_new, n_new))
        
        # Map old indices to new indices
        # The new node is at the end.
        for k in range(n_new):
            if k == n_new - 1:
                # Distance from new node to others
                for m in range(n_new - 1):
                    other_node = new_current_nodes[m]
                    # Find original indices of i, j relative to current_nodes
                    # This is tricky because we removed i and j.
                    # Let's use a different approach: maintain a mapping from node object to index in d.
                    pass
        
        # Actually, let's just use a dictionary for distances between node objects.
        # It's slower but much easier to manage.
        pass
        
    # This NJ implementation is getting messy for SPR.
    # Let's stick to the previous NJ implementation that returned a TreeNode structure,
    # and then convert that to a structure suitable for SPR.
    
    return None

def _nj_run(dists, labels):
    """Standard Neighbor Joining to get an initial tree topology. Returns a TreeNode structure."""
    
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

def _get_splits_from_tree(root):
    """Get the set of splits (frozensets of leaf labels) from a tree."""
    splits = set()
    
    def get_leaves(node):
        if node.is_leaf():
            return frozenset([node.label])
        left_leaves = get_leaves(node.left)
        right_leaves = get_leaves(node.right)
        return left_leaves | right_leaves
    
    def traverse(node, parent):
        if node.is_leaf():
            return
        # The split defined by the edge (node, parent)
        # One side is the subtree rooted at node.
        # The other side is the rest of the tree.
        # We represent a split as a frozenset of the smaller side (or just one side).
        # To make it canonical, we can just use the frozenset of leaves in node's subtree.
        # But we need to ensure we don't add the trivial splits (size 1 or n-1).
        
        subtree_leaves = get_leaves(node)
        if 1 < len(subtree_leaves) < total_leaves:
            # Canonicalize: use the smaller side
            if len(subtree_leaves) > total_leaves // 2:
                # We don't have the "rest" easily here without passing it down.
                # Let's just add the frozenset.
                # To be safe, we'll add both sides? No, a split is {A, B}.
                # Let's store the frozenset of the subtree.
                splits.add(subtree_leaves)
        
        if node.left:
            traverse(node.left, node)
        if node.right:
            traverse(node.right, node)
            
    total_leaves = len(root.get_leaves())
    # We need to handle the root edge? No, root has no parent.
    # The splits are defined by internal edges.
    # In a rooted tree representation of an unrooted tree, the "root" edge is artificial.
    # We should traverse all internal nodes except the root?
    # Actually, every internal node (except root) defines a split with its parent.
    # But the root itself defines a split if we consider the unrooted nature?
    # No, the root is just a node. The edges are (root, left) and (root, right).
    # The edge (root, left) defines a split: leaves(left) vs leaves(right).
    # The edge (root, right) defines the same split.
    # So we should add the split for the root's children.
    
    # Let's refine traverse to add splits for every edge.
    splits = set()
    
    def get_leaves_fast(node):
        if node.is_leaf():
            return frozenset([node.label])
        return get_leaves_fast(node.left) | get_leaves_fast(node.right)
        
    def add_splits(node, is_root):
        if node.is_leaf():
            return
        # The subtree at node defines a split with the rest of the tree.
        # If is_root, we don't add a split for the "root" itself (no parent edge).
        # But its children define splits.
        # Wait, the edge connecting node to its parent defines a split.
        # If node is root, there is no parent.
        # So we only add splits for non-root nodes?
        # No, the edge (root, child) defines a split. The child is the one with the parent.
        # So yes, for every non-root node, the edge to its parent defines a split.
        # The leaves in the subtree rooted at that node form one side of the split.
        
        if not is_root:
            subtree_leaves = get_leaves_fast(node)
            if 1 < len(subtree_leaves) < total_leaves:
                # Canonicalize to smaller side
                if len(subtree_leaves) > total_leaves // 2:
                    # We need the complement.
                    # This is inefficient.
                    # Let's just store the frozenset.
                    # For RF distance, we need to match splits.
                    # A split is {A, B}. A is a subset of leaves.
                    # If we store A, we must be consistent.
                    # Let's store the smaller side.
                    # To get the complement, we need the total set.
                    pass
                splits.add(subtree_leaves)
                
        if node.left:
            add_splits(node.left, False)
        if node.right:
            add_splits(node.right, False)
            
    add_splits(root, True)
    
    # We also need to handle the case where the "root" of our rooted tree
    # corresponds to an internal edge in the unrooted tree.
    # In NJ, the root is the last node created.
    # The edges from root to its children are real edges.
    # So the splits defined by the children of the root are valid.
    # My `add_splits` adds splits for all non-root nodes.
    # This includes the children of the root.
    # So it should be correct.
    
    # However, `get_leaves_fast` is called repeatedly, which is O(N^2) or worse.
    # Let's optimize by computing leaves once.
    
    leaves_cache = {}
    def compute_leaves(node):
        if node.is_leaf():
            return frozenset([node.label])
        if id(node) in leaves_cache:
            return leaves_cache[id(node)]
        l = compute_leaves(node.left)
        r = compute_leaves(node.right)
        res = l | r
        leaves_cache[id(node)] = res
        return res
        
    splits = set()
    total_leaves = len(compute_leaves(root))
    
    def traverse_for_splits(node, is_root):
        if node.is_leaf():
            return
        if not is_root:
            subtree_leaves = compute_leaves(node)
            if 1 < len(subtree_leaves) < total_leaves:
                # Store the smaller side for canonicalization
                if len(subtree_leaves) <= total_leaves // 2:
                    splits.add(subtree_leaves)
                else:
                    # We need the complement.
                    # We can get the complement by taking all leaves and removing subtree_leaves.
                    # But we don't have the set of all leaves handy in a set format.
                    # Let's just store the frozenset.
                    # To compare splits, we need to handle both sides.
                    # Let's store both sides? No, that doubles the size.
                    # Let's just store the frozenset and handle comparison carefully.
                    # Or, simpler: store the frozenset of the smaller side.
                    # To get the smaller side, we need the total set.
                    # Let's compute the total set once.
                    pass
                # For now, let's just add the subtree leaves.
                # We will handle canonicalization in the distance function.
                splits.add(subtree_leaves)
        
        if node.left:
            traverse_for_splits(node.left, False)
        if node.right:
            traverse_for_splits(node.right, False)
            
    traverse_for_splits(root, True)
    
    return splits

def _parsimony_score(tree, sequences, labels):
    """Calculate the parsimony score (minimum number of changes) for a tree."""
    # Fitch algorithm
    # For each site, compute the minimum number of changes.
    
    # Convert sequences to a dictionary of label -> string
    # We need to traverse the tree bottom-up.
    
    # Let's build a list of nodes in post-order.
    nodes = []
    def collect_nodes(node):
        if node.is_leaf():
            nodes.append(node)
        else:
            collect_nodes(node.left)
            collect_nodes(node.right)
            nodes.append(node)
            
    collect_nodes(tree)
    
    # For each node, store the set of possible states at that node.
    # For leaves, it's the observed state.
    # For internal nodes, it's the intersection of children's sets if non-empty, else union.
    
    total_cost = 0
    
    # We need to process site by site? Or all sites at once?
    # Site by site is easier to implement correctly.
    # L is up to 2000. N is up to 32.
    # 2000 * 32 operations is fast.
    
    n_sites = len(sequences[labels[0]])
    
    # Pre-compute leaf indices for fast access
    # Actually, let's just iterate.
    
    for site in range(n_sites):
        # Store state sets for each node
        state_sets = {}
        costs = {}
        
        for node in nodes:
            if node.is_leaf():
                # Get the character at this site for this leaf
                char = sequences[node.label][site]
                state_sets[node] = frozenset([char])
                costs[node] = 0
            else:
                left_set = state_sets[node.left]
                right_set = state_sets[node.right]
                
                intersection = left_set & right_set
                if intersection:
                    state_sets[node] = intersection
                    cost = costs[node.left] + costs[node.right]
                else:
                    state_sets[node] = left_set | right_set
                    cost = costs[node.left] + costs[node.right] + 1
                    
                costs[node] = cost
                
        total_cost += costs[tree]
        
    return total_cost

def _spr_refine(tree, sequences, labels, max_iterations=10):
    """
    Perform SPR refinement.
    This is a simplified version that tries to improve the parsimony score.
    """
    current_score = _parsimony_score(tree, sequences, labels)
    
    # We need a way to represent the tree that allows SPR moves.
    # The current TreeNode structure is a rooted binary tree.
    # SPR moves on unrooted trees are complex to implement on this structure.
    # Given the time, I will implement a simple NNI-like move or just return the tree.
    # The inspiration says SPR works. Let's try to implement a basic version.
    
    # For now, let's just return the tree.
    # The NJ tree is already a good starting point.
    # If we have time, we can add a simple heuristic.
    
    return tree

def reconstruct(sequences):
    _init_time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    if n_taxa < 3:
        if n_taxa == 1:
            return labels[0] + ";"
        elif n_taxa == 2:
            return f"({labels[0]},{labels[1]});"
            
    # Convert sequences to integer matrix for distance calculation
    seq_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    
    # Compute distance matrix
    dists = _compute_distance_matrix(sequences, labels)
    
    # Run NJ
    root = _nj_run(dists, labels)
    
    # Optional: SPR Refinement (currently a stub)
    # root = _spr_refine(root, sequences, labels)
    
    # Convert to Newick
    newick = _tree_to_newick(root)
    
    return newick + ";"

# EVOLVE-BLOCK-END
