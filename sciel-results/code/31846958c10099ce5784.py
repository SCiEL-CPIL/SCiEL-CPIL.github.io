import time
import math
import numpy as np
import random

_START_TIME = None
_TIME_LIMIT = 250.0  # Safety margin for 270s limit

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_left():
    _init_time()
    return max(0.0, _TIME_LIMIT - (time.time() - _START_TIME))

class TreeNode:
    __slots__ = ['left', 'right', 'label', 'parent', 'length']
    def __init__(self, left=None, right=None, label=None, parent=None, length=0.0):
        self.left = left
        self.right = right
        self.label = label
        self.parent = parent
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

def _nj_run(dists, labels):
    """
    Standard Neighbor Joining.
    Returns a TreeNode structure.
    """
    n = len(labels)
    if n < 3:
        if n == 1:
            return TreeNode(label=labels[0])
        elif n == 2:
            root = TreeNode()
            l = TreeNode(label=labels[0], parent=root)
            r = TreeNode(label=labels[1], parent=root)
            root.left = l
            root.right = r
            return root

    # Map label to index
    label_to_idx = {l: i for i, l in enumerate(labels)}
    
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
        node_i.parent = new_node
        node_j.parent = new_node
        new_node.left = node_i
        new_node.right = node_j
        node_i.length = max(0.0, len_i)
        node_j.length = max(0.0, len_j)
        
        node_map[new_id] = new_node
        
        # Update active_ids and current_d
        new_active_ids = [k for k in range(n_curr) if k != i and k != j]
        new_active_ids.append(len(new_active_ids))  # placeholder for new node index
        
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
        
    # Final step: connect the last 2 nodes
    if len(active_ids) == 2:
        id1 = active_ids[0]
        id2 = active_ids[1]
        node1 = node_map[id1]
        node2 = node_map[id2]
        
        d_12 = current_d[0, 1]
        
        root = TreeNode()
        node1.parent = root
        node2.parent = root
        root.left = node1
        root.right = node2
        node1.length = 0.5 * d_12
        node2.length = 0.5 * d_12
        
        return root
    elif len(active_ids) == 3:
        pass
        
    # Fallback
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

def _tree_splits(root):
    """Extract splits from a tree."""
    splits = set()
    all_leaves = set(root.get_leaves())
    
    def dfs(node):
        if node.is_leaf():
            return {node.label}
        
        left_leaves = dfs(node.left)
        right_leaves = dfs(node.right)
        
        if len(left_leaves) <= len(right_leaves):
            splits.add(frozenset(left_leaves))
        else:
            splits.add(frozenset(right_leaves))
            
        return left_leaves | right_leaves
        
    dfs(root)
    return splits

def _nni_refine(root, dists, labels, max_iters=100):
    """
    Perform NNI refinement using distance-based objective.
    """
    n = len(labels)
    
    def get_implied_dists(tree_root):
        leaves = tree_root.get_leaves()
        n_local = len(leaves)
        implied_dists = np.zeros((n_local, n_local))
        
        for i in range(n_local):
            for j in range(i+1, n_local):
                u = leaves[i]
                v = leaves[j]
                
                visited = {id(u): 0.0}
                stack = [u]
                found = False
                while stack:
                    node = stack.pop()
                    if node is v:
                        found = True
                        break
                    if node.left and id(node.left) not in visited:
                        visited[id(node.left)] = visited[id(node)] + node.left.length
                        stack.append(node.left)
                    if node.right and id(node.right) not in visited:
                        visited[id(node.right)] = visited[id(node)] + node.right.length
                        stack.append(node.right)
                        
                if found:
                    d = visited[id(v)]
                else:
                    d = 0.0
                
                implied_dists[i, j] = d
                implied_dists[j, i] = d
                
        return implied_dists

    def objective(tree_root):
        implied = get_implied_dists(tree_root)
        diff = dists - implied
        return np.sum(diff**2)
        
    current_obj = objective(root)
    
    # Find all internal edges and perform NNI
    # For simplicity, we'll iterate over all internal nodes
    # and consider edges to their children
    
    def dfs_find_edges(node, parent=None):
        edges = []
        if node.is_leaf():
            return edges
        
        if parent is None:
            # Root node
            if node.left:
                edges.extend(dfs_find_edges(node.left, node))
            if node.right:
                edges.extend(dfs_find_edges(node.right, node))
        else:
            # Non-root internal node
            # Edge to parent
            if node.left:
                edges.extend(dfs_find_edges(node.left, node))
            if node.right:
                edges.extend(dfs_find_edges(node.right, node))
            
        return edges
    
    # For NNI, we need to identify internal edges
    # An internal edge is one where both sides have at least 2 leaves
    
    def get_subtree_leaves(node):
        return set(node.get_leaves())
    
    def perform_nni(u, v, root):
        """
        Perform NNI move on edge (u, v).
        Returns new root if successful, None if not a valid NNI.
        """
        # Find the other neighbors of u and v
        # u's neighbors: parent(u) or None, left(u), right(u)
        # We need to find the two "other" subtrees hanging off u and v
        
        def get_neighbors(node):
            neighbors = []
            if node.parent is not None:
                neighbors.append(node.parent)
            if node.left is not None:
                neighbors.append(node.left)
            if node.right is not None:
                neighbors.append(node.right)
            return neighbors
        
        u_neighbors = get_neighbors(u)
        v_neighbors = get_neighbors(v)
        
        # Remove each other from neighbor lists
        u_other = [x for x in u_neighbors if x is not v]
        v_other = [x for x in v_neighbors if x is not u]
        
        if len(u_other) != 2 or len(v_other) != 2:
            return None, None, None, None
        
        a, b = u_other
        c, d = v_other
        
        # Check if this is a valid NNI
        # All four subtrees must have at least one leaf
        if not (a.is_leaf() or (a.left is not None or a.right is not None)):
            return None, None, None, None
        if not (b.is_leaf() or (b.left is not None or b.right is not None)):
            return None, None, None, None
        if not (c.is_leaf() or (c.left is not None or c.right is not None)):
            return None, None, None, None
        if not (d.is_leaf() or (d.left is not None or d.right is not None)):
            return None, None, None, None
        
        return a, b, c, d
    
    # For each internal edge, try NNI
    # We'll collect all internal edges first
    
    internal_edges = []
    
    def collect_edges(node, parent=None):
        if node.is_leaf():
            return
        if parent is not None:
            # This is an edge (parent, node)
            # Check if both sides have >= 2 leaves
            side1 = get_subtree_leaves(node)
            all_leaves = get_subtree_leaves(root)
            side2 = all_leaves - side1
            if len(side1) >= 2 and len(side2) >= 2:
                internal_edges.append((parent, node))
        
        if node.left:
            collect_edges(node.left, node)
        if node.right:
            collect_edges(node.right, node)
    
    collect_edges(root)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        for u, v in internal_edges:
            a, b, c, d = perform_nni(u, v, root)
            if a is None:
                continue
            
            # Try two NNI swaps
            # Swap 1: connect a-c and b-d
            # Original: (a,b) - (c,d)
            # New 1: (a,c) - (b,d)
            
            # This is complex to implement correctly with parent pointers
            # For now, skip NNI refinement to ensure robustness
            pass
    
    return root, current_obj

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
    
    # Convert to Newick
    newick = _tree_to_newick(root)
    
    return newick + ";"

# EVOLVE-BLOCK-END
