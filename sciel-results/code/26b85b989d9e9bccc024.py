# EVOLVE-BLOCK-START
import time
import math
import numpy as np
import sys

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

def _seqs_to_array(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    seqs = [sequences[l] for l in labels]
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[mapping[c] for c in s] for s in seqs], dtype=np.uint8)
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
        return labels[0], set()
    if n == 2:
        return f"({labels[0]},{labels[1]})", set()
    
    all_leaves = set(labels)
    nodes = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    leaf_sets = {i: {labels[i]} for i in range(n)}
    
    d = {}
    for i in range(n):
        d[i] = {}
        for j in range(n):
            d[i][j] = dist[i][j]
    
    splits = set()
    
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
        
        new_id = -1 - len(nodes)
        trees[new_id] = f"({trees[a]},{trees[b]})"
        leaf_sets[new_id] = leaf_sets[a] | leaf_sets[b]
        
        side1 = frozenset(leaf_sets[new_id])
        side2 = frozenset(all_leaves - leaf_sets[new_id])
        if tuple(sorted(side1)) < tuple(sorted(side2)):
            split = side1
        else:
            split = side2
        if 1 < len(split) < n - 1:
            splits.add(split)
            
        d[new_id] = {}
        for x in nodes:
            if x == a or x == b:
                continue
            d[new_id][x] = 0.5 * (d[a][x] + d[b][x] - d[a][b])
            d[x][new_id] = d[new_id][x]
            
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_id)
        
    final_tree = "(" + ",".join(trees[x] for x in nodes) + ")"
    return final_tree, splits

# --- Fitch Parsimony ---

def _fitch_score_vectorized(root, data, name_to_idx):
    """Vectorized Fitch score computation."""
    n_sites = data.shape[1]
    
    order = []
    stack = [root]
    while stack:
        node = stack.pop()
        order.append(node)
        for child in node.children:
            stack.append(child)
    
    node_to_pos = {id(node): i for i, node in enumerate(order)}
    states = np.zeros((len(order), n_sites), dtype=np.uint8)
    
    for i in range(len(order) - 1, -1, -1):
        node = order[i]
        if not node.children:
            idx = name_to_idx[node.name]
            states[i, :] = 1 << data[idx, :]
        else:
            child_positions = [node_to_pos[id(c)] for c in node.children]
            inter = states[child_positions[0], :].copy()
            for cp in child_positions[1:]:
                inter &= states[cp, :]
            union = states[child_positions[0], :].copy()
            for cp in child_positions[1:]:
                union |= states[cp, :]
            score_mask = (inter == 0)
            states[i, :] = np.where(score_mask, union, inter)
    
    total_score = 0
    for i in range(len(order) - 1, -1, -1):
        node = order[i]
        if node.children:
            child_positions = [node_to_pos[id(c)] for c in node.children]
            inter = states[child_positions[0], :].copy()
            for cp in child_positions[1:]:
                inter &= states[cp, :]
            total_score += (inter == 0).sum()
    
    return total_score

class _Node:
    __slots__ = ['name', 'parent', 'children']
    def __init__(self, name=None):
        self.name = name
        self.parent = None
        self.children = []

def _parse_newick(newick, labels):
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    root = _Node()
    stack = [root]
    i = 0
    while i < len(s):
        c = s[i]
        if c == '(':
            new_node = _Node()
            stack[-1].children.append(new_node)
            new_node.parent = stack[-1]
            stack.append(new_node)
        elif c == ',':
            pass
        elif c == ')':
            stack.pop()
        elif c == ':':
            while i < len(s) and s[i] not in ',)':
                i += 1
            i -= 1
        else:
            start = i
            while i < len(s) and s[i] not in '(),:':
                i += 1
            name = s[start:i]
            if name in labels:
                stack[-1].name = name
            else:
                stack[-1].name = name
        i += 1
    
    while len(root.children) == 1:
        root = root.children[0]
        root.parent = None
    return root

def _tree_to_newick(root):
    def _node_to_str(node):
        if not node.children:
            return node.name
        else:
            child_strs = [_node_to_str(c) for c in node.children]
            return "(" + ",".join(child_strs) + ")"
    return _node_to_str(root)

def _get_all_nodes(root):
    all_nodes = []
    stack = [root]
    while stack:
        node = stack.pop()
        all_nodes.append(node)
        for child in node.children:
            stack.append(child)
    return all_nodes

def _get_leaves(root):
    leaves = []
    stack = [root]
    while stack:
        node = stack.pop()
        if not node.children:
            leaves.append(node)
        else:
            for child in node.children:
                stack.append(child)
    return leaves

def _get_internal_nodes(root):
    internal = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.children:
            internal.append(node)
            for child in node.children:
                stack.append(child)
    return internal

def _spr_refine(root, data, name_to_idx, time_budget):
    """Subtree Prune and Regraft refinement using Fitch parsimony."""
    n_taxa = len(name_to_idx)
    
    if n_taxa <= 4:
        return root, _fitch_score_vectorized(root, data, name_to_idx)
    
    leaves = _get_leaves(root)
    internal_nodes = _get_internal_nodes(root)
    
    current_score = _fitch_score_vectorized(root, data, name_to_idx)
    best_score = current_score
    
    start_time = time.time()
    max_iterations = 50
    
    for iteration in range(max_iterations):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        
        for leaf in leaves:
            if time.time() - start_time > time_budget:
                break
            
            parent = leaf.parent
            if parent is None:
                continue
            
            # Prune the leaf
            parent.children.remove(leaf)
            leaf.parent = None
            
            # If parent now has only one child, we can't graft there
            # Try regrafting at various internal nodes
            best_local_score = current_score
            best_graft_node = None
            
            # Sample regrafting positions to save time
            if len(internal_nodes) > 8:
                import random
                graft_candidates = random.sample(internal_nodes, min(8, len(internal_nodes)))
            else:
                graft_candidates = internal_nodes
            
            for graft_node in graft_candidates:
                if graft_node == parent:
                    continue
                
                # Regraft
                graft_node.children.append(leaf)
                leaf.parent = graft_node
                
                new_score = _fitch_score_vectorized(root, data, name_to_idx)
                
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_graft_node = graft_node
                
                # Undo regrafting
                graft_node.children.remove(leaf)
                leaf.parent = None
            
            # Apply best regrafting
            if best_graft_node is not None:
                best_graft_node.children.append(leaf)
                leaf.parent = best_graft_node
                current_score = best_local_score
                improved = True
                
                if current_score < best_score:
                    best_score = current_score
            else:
                # Put leaf back
                parent.children.append(leaf)
                leaf.parent = parent
    
    return root, best_score

def _bootstrap_consensus(labels, arr, bootstrap_budget):
    """Run bootstrap resampling to build consensus tree."""
    L = arr.shape[1]
    n = len(labels)
    
    # First, get the full-data NJ tree
    d_full = _jc_matrix(arr)
    tree_full, splits_full = _nj(d_full, labels)
    
    candidates = [(tree_full, splits_full)]
    split_counts = {s: 0 for s in splits_full}
    
    num_bootstraps = 0
    max_bootstraps = 2000
    start_time = time.time()
    
    while num_bootstraps < max_bootstraps and (bootstrap_budget - (time.time() - start_time)) > 5.0:
        indices = np.random.randint(0, L, L)
        arr_boot = arr[:, indices]
        d_boot = _jc_matrix(arr_boot)
        tree_boot, splits_boot = _nj(d_boot, labels)
        candidates.append((tree_boot, splits_boot))
        
        for s in splits_boot:
            split_counts[s] = split_counts.get(s, 0) + 1
        
        num_bootstraps += 1
    
    if len(candidates) <= 1:
        return tree_full
    
    total_bootstraps = len(candidates)
    threshold = 0.5 * total_bootstraps
    
    strong_splits = set()
    for s, c in split_counts.items():
        if c > threshold:
            strong_splits.add(s)
    
    best_idx = 0
    best_score = -1
    second_best_score = float('inf')
    
    for i in range(len(candidates)):
        splits_i = candidates[i][1]
        common = len(splits_i & strong_splits)
        
        if common > best_score:
            best_score = common
            best_idx = i
            second_best_score = len(splits_i.symmetric_difference(candidates[0][1]))
        elif common == best_score:
            rf_to_nj = len(splits_i.symmetric_difference(candidates[0][1]))
            if rf_to_nj < second_best_score:
                best_idx = i
                second_best_score = rf_to_nj
    
    return candidates[best_idx][0]

def reconstruct(sequences):
    global _START
    _START = time.time()
    
    labels, arr = _seqs_to_array(sequences)
    n = len(labels)
    name_to_idx = {l: i for i, l in enumerate(labels)}
    
    # Bootstrap consensus
    bootstrap_budget = 270.0 * _DEADLINE_FRACTION * 0.55
    best_tree_newick = _bootstrap_consensus(labels, arr, bootstrap_budget)
    
    # Parse the consensus tree
    root = _parse_newick(best_tree_newick, labels)
    
    # SPR refinement with remaining time budget
    spr_budget = max(0.0, 270.0 * _DEADLINE_FRACTION - (time.time() - _START) - 10.0)
    
    if spr_budget > 2.0:
        root, final_score = _spr_refine(root, arr, name_to_idx, spr_budget)
        best_tree_newick = _tree_to_newick(root)
    
    return best_tree_newick

# EVOLVE-BLOCK-END
