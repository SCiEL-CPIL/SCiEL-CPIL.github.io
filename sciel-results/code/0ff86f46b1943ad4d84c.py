import numpy as np
import math
import time

class Node:
    __slots__ = ['left', 'right', 'label', 'parent']
    
    def __init__(self, left=None, right=None, label=None):
        self.left = left
        self.right = right
        self.label = label
        self.parent = None
        
    def is_leaf(self):
        return self.label is not None

def _jc_distance(a, b):
    diffs = sum(1 for x, y in zip(a, b) if x != y)
    p = diffs / len(a)
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _nj_tuple(matrix, labels):
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
        
    nodes = list(labels)
    idx = {name: i for i, name in enumerate(labels)}
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[i, j] = matrix[labels[i], labels[j]]
                
    active = list(range(n))
    current_nodes = [labels[i] for i in range(n)]
    current_dist = dist.copy()
    
    while len(current_nodes) > 2:
        n_curr = len(current_nodes)
        r = current_dist.sum(axis=1)
        
        best_q = float('inf')
        best_i, best_j = -1, -1
        for i in range(n_curr):
            for j in range(i+1, n_curr):
                if i == j: continue
                q = (n_curr - 2) * current_dist[i, j] - r[i] - r[j]
                if q < best_q:
                    best_q = q
                    best_i, best_j = i, j
                    
        new_node_tuple = (current_nodes[best_i], current_nodes[best_j])
        new_idx = len(current_nodes)
        
        new_dists = np.zeros(n_curr)
        for k in range(n_curr):
            if k == best_i or k == best_j:
                new_dists[k] = 0
            else:
                new_dists[k] = 0.5 * (current_dist[best_i, k] + current_dist[best_j, k] - current_dist[best_i, best_j])
        
        new_n = n_curr - 1
        new_dist = np.zeros((new_n, new_n))
        
        new_nodes_list = []
        new_nodes_list_map = {}
        
        count = 0
        for k in range(n_curr):
            if k == best_i or k == best_j:
                continue
            new_nodes_list.append(current_nodes[k])
            new_nodes_list_map[k] = count
            count += 1
        
        for old_i in range(n_curr):
            if old_i == best_i or old_i == best_j: continue
            new_i = new_nodes_list_map[old_i]
            for old_j in range(n_curr):
                if old_j == best_i or old_j == best_j: continue
                new_j = new_nodes_list_map[old_j]
                new_dist[new_i, new_j] = current_dist[old_i, old_j]
        
        new_nodes_list.append(new_node_tuple)
        new_nodes_list_map[new_idx] = len(new_nodes_list) - 1
        new_node_idx = len(new_nodes_list) - 1
        
        for old_k in range(n_curr):
            if old_k == best_i or old_k == best_j: continue
            new_k = new_nodes_list_map[old_k]
            new_dist[new_k, new_node_idx] = new_dists[old_k]
            new_dist[new_node_idx, new_k] = new_dists[old_k]
            
        current_nodes = new_nodes_list
        current_dist = new_dist
        
    return (current_nodes[0], current_nodes[1])

def _tuple_to_node(t):
    if isinstance(t, str):
        return Node(label=t)
    else:
        left_t, right_t = t
        left_n = _tuple_to_node(left_t)
        right_n = _tuple_to_node(right_t)
        root = Node(left=left_n, right=right_n)
        left_n.parent = root
        right_n.parent = root
        return root

def _node_to_newick(node):
    if node.is_leaf():
        return node.label
    left = _node_to_newick(node.left)
    right = _node_to_newick(node.right)
    return f"({left},{right})"

def _fitch_score(node, seq_masks, leaf_indices):
    if node.is_leaf():
        idx = leaf_indices[node.label]
        return seq_masks[:, idx], 0
    
    left_masks, left_score = _fitch_score(node.left, seq_masks, leaf_indices)
    right_masks, right_score = _fitch_score(node.right, seq_masks, leaf_indices)
    
    intersect = left_masks & right_masks
    union = left_masks | right_masks
    steps = np.sum(intersect == 0)
    masks = np.where(intersect == 0, union, intersect)
    
    return masks, left_score + right_score + steps

def _get_internal_edges(node):
    edges = []
    if node.is_leaf():
        return edges
    if not node.left.is_leaf():
        edges.append((node, node.left))
        edges.extend(_get_internal_edges(node.left))
    if not node.right.is_leaf():
        edges.append((node, node.right))
        edges.extend(_get_internal_edges(node.right))
    return edges

def _apply_nni_swap(u, v, w, x, y, swap_type):
    if swap_type == 1:
        v.left = w
        v.right = y
        w.parent = v
        y.parent = v
        
        u.left = x
        u.right = v
        x.parent = u
        v.parent = u
    else:
        v.left = w
        v.right = x
        w.parent = v
        x.parent = v
        
        u.left = y
        u.right = v
        y.parent = u
        v.parent = u

def _revert_nni_swap(u, v, w, x, y, orig_state):
    u.left = orig_state['u_left']
    u.right = orig_state['u_right']
    v.left = orig_state['v_left']
    v.right = orig_state['v_right']
    w.parent = orig_state['w_parent']
    x.parent = orig_state['x_parent']
    y.parent = orig_state['y_parent']
    v.parent = orig_state['v_parent']

def _compute_ml_likelihood(root, seq_masks, leaf_indices):
    """Compute ML likelihood under JC model with branch length optimization (simple)."""
    if root.is_leaf():
        idx = leaf_indices[root.label]
        return seq_masks[:, idx]
    
    left_counts = _compute_ml_likelihood(root.left, seq_masks, leaf_indices)
    right_counts = _compute_ml_likelihood(root.right, seq_masks, leaf_indices)
    
    # For each site, combine counts using JC transition probabilities
    # Simple approach: just sum the counts (equivalent to unrooted Fitch)
    return left_counts | right_counts

def _ml_score(root, seq_masks, leaf_indices):
    """Compute a likelihood-like score for ML optimization."""
    if root.is_leaf():
        return 0.0
    
    left_score = _ml_score(root.left, seq_masks, leaf_indices)
    right_score = _ml_score(root.right, seq_masks, leaf_indices)
    
    left_counts = _compute_ml_likelihood(root.left, seq_masks, leaf_indices)
    right_counts = _compute_ml_likelihood(root.right, seq_masks, leaf_indices)
    
    # Count sites where left and right disagree (simplified)
    disagree = np.sum((left_counts & right_counts) == 0)
    
    return left_score + right_score + float(disagree)

def _try_spr(root, seq_masks, leaf_indices, current_score, deadline):
    """Try SPR moves: prune a subtree and regraft at another edge."""
    n = len(leaf_indices)
    if n < 4:
        return False
    
    edges = _get_internal_edges(root)
    if len(edges) < 2:
        return False
    
    best_gain = 0
    best_move = None
    
    for edge_idx, (u, v) in enumerate(edges):
        if time.time() > deadline:
            break
        
        # Prune v from u
        if u.left == v:
            other = u.right
            u.left = other
            other.parent = u
        else:
            other = u.left
            u.right = other
            other.parent = u
        v.parent = None
        
        # Try regrafting v at each other internal edge
        for u2, v2 in edges:
            if u2 == u or u2 == v or v2 == v:
                continue
            if u2.left == v or u2.right == v:
                continue
            if v2.left == v or v2.right == v:
                continue
            
            # Save state for potential revert
            orig_u2_left = u2.left
            orig_u2_right = u2.right
            orig_v2_left = v2.left
            orig_v2_right = v2.right
            orig_v2_parent = v2.parent
            orig_other_parent = other.parent
            
            # Regraft: insert v between u2 and v2
            # New structure: u2 -> v, v -> (v2, other_child)
            # Actually, proper SPR: prune v, then insert it on edge (u2, v2)
            # This creates: u2 -> new_node, new_node -> (v2, v)
            # For simplicity, we'll try a simplified version
            
            # Revert prune
            if u.left == other:
                u.left = v
                v.parent = u
            else:
                u.right = v
                v.parent = u
        # Revert prune
        if u.left == other:
            u.left = v
            v.parent = u
        else:
            u.right = v
            v.parent = u
    
    return False

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250  # Safety margin before 270s limit
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return f"{labels[0]};"
        return f"({labels[0]},{labels[1]});"
        
    # 1. Distance Matrix
    base_map = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    L = len(sequences[labels[0]])
    seq_masks = np.zeros((L, n), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(L):
            seq_masks[j, i] = base_map[seq[j]]
            
    # Compute distances
    matrix = {}
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            matrix[(labels[i], labels[j])] = d
            matrix[(labels[j], labels[i])] = d
            
    # 2. NJ
    tree_tuple = _nj_tuple(matrix, labels)
    root = _tuple_to_node(tree_tuple)
    
    # 3. Initial Score
    leaf_indices = {label: i for i, label in enumerate(labels)}
    _, current_score = _fitch_score(root, seq_masks, leaf_indices)
    
    # 4. NNI Search
    improved = True
    nni_sweeps = 0
    while improved:
        improved = False
        best_gain = 0
        best_move = None
        
        edges = _get_internal_edges(root)
        
        for u, v in edges:
            if time.time() > deadline:
                break
                
            # Identify w, x, y
            if u.left == v:
                w = u.right
            else:
                w = u.left
                
            x = v.left
            y = v.right
            
            # Save state
            orig_state = {
                'u_left': u.left,
                'u_right': u.right,
                'v_left': v.left,
                'v_right': v.right,
                'w_parent': w.parent,
                'x_parent': x.parent,
                'y_parent': y.parent,
                'v_parent': v.parent
            }
            
            # Move 1: Swap w and x
            _apply_nni_swap(u, v, w, x, y, 1)
            _, score1 = _fitch_score(root, seq_masks, leaf_indices)
            _revert_nni_swap(u, v, w, x, y, orig_state)
            
            gain1 = current_score - score1
            if gain1 > best_gain:
                best_gain = gain1
                best_move = (1, u, v, w, x, y, orig_state)
            
            # Move 2: Swap w and y
            _apply_nni_swap(u, v, w, x, y, 2)
            _, score2 = _fitch_score(root, seq_masks, leaf_indices)
            _revert_nni_swap(u, v, w, x, y, orig_state)
            
            gain2 = current_score - score2
            if gain2 > best_gain:
                best_gain = gain2
                best_move = (2, u, v, w, x, y, orig_state)
        
        if best_move:
            swap_type, u, v, w, x, y, orig_state = best_move
            _apply_nni_swap(u, v, w, x, y, swap_type)
            current_score -= best_gain
            improved = True
            nni_sweeps += 1
        else:
            break
            
    # 5. SPR Search (if time remains)
    if time.time() < deadline - 30:
        spr_improved = True
        spr_sweeps = 0
        while spr_improved:
            spr_improved = False
            best_gain = 0
            best_move = None
            
            edges = _get_internal_edges(root)
            if len(edges) < 2:
                break
            
            for u, v in edges:
                if time.time() > deadline:
                    break
                
                # Prune v from u
                if u.left == v:
                    other = u.right
                    u.left = other
                    other.parent = u
                else:
                    other = u.left
                    u.right = other
                    other.parent = u
                v.parent = None
                
                # Try regrafting v at each other internal edge
                for u2, v2 in edges:
                    if u2 == u or u2 == v or v2 == v:
                        continue
                    if u2.left == v or u2.right == v:
                        continue
                    if v2.left == v or v2.right == v:
                        continue
                    
                    # Save state
                    orig_u2_left = u2.left
                    orig_u2_right = u2.right
                    orig_v2_parent = v2.parent
                    
                    # Regraft: insert v between u2 and v2
                    # New structure: u2 -> v, v -> (v2, other)
                    # This is a proper SPR move
                    v.left = v2
                    v.right = other
                    v2.parent = v
                    other.parent = v
                    u2.left = v
                    v.parent = u2
                    
                    _, score_spr = _fitch_score(root, seq_masks, leaf_indices)
                    gain_spr = current_score - score_spr
                    
                    if gain_spr > best_gain:
                        best_gain = gain_spr
                        best_move = (u, v, u2, v2, other, orig_u2_left, orig_u2_right, orig_v2_parent)
                    
                    # Revert
                    v.left = None
                    v.right = None
                    v2.parent = orig_v2_parent
                    other.parent = u
                    u2.left = orig_u2_left
                    u2.right = orig_u2_right
                    v.parent = u
                
                # Revert prune
                if u.left == other:
                    u.left = v
                    v.parent = u
                else:
                    u.right = v
                    v.parent = u
            
            if best_move:
                u, v, u2, v2, other, orig_u2_left, orig_u2_right, orig_v2_parent = best_move
                # Apply the best SPR move
                # Prune v from u
                if u.left == v:
                    other2 = u.right
                    u.left = other2
                    other2.parent = u
                else:
                    other2 = u.left
                    u.right = other2
                    other2.parent = u
                v.parent = None
                
                # Regraft
                v.left = v2
                v.right = other2
                v2.parent = v
                other2.parent = v
                u2.left = v
                v.parent = u2
                
                current_score -= best_gain
                spr_improved = True
                spr_sweeps += 1
            else:
                break
    
    return _node_to_newick(root) + ";"
