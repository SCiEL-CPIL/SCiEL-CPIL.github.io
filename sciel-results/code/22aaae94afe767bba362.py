import numpy as np

def _to_newick(tree):
    if isinstance(tree, str):
        return tree
    return f"({_to_newick(tree[0])},{_to_newick(tree[1])})"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
        
    # Convert sequences to bitmask array
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    S = len(sequences[labels[0]])
    seq_matrix = np.zeros((S, n), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j in range(S):
            seq_matrix[j, i] = char_map[seq[j]]
            
    bitmasks = 1 << seq_matrix
    
    def make_balanced_tree(labels):
        if len(labels) == 1:
            return labels[0]
        mid = len(labels) // 2
        left = make_balanced_tree(labels[:mid])
        right = make_balanced_tree(labels[mid:])
        return (left, right)
        
    current_tree = make_balanced_tree(labels)
    
    def score_tree(tree):
        if isinstance(tree, str):
            idx = int(tree[1:])
            return bitmasks[:, idx], 0
        left_mask, left_cost = score_tree(tree[0])
        right_mask, right_cost = score_tree(tree[1])
        intersection = left_mask & right_mask
        union = left_mask | right_mask
        cost = left_cost + right_cost + np.sum(intersection == 0)
        mask = np.where(intersection != 0, intersection, union)
        return mask, int(cost)
        
    def get_nni_neighbors(tree):
        def find_edges(node, parent=None, sibling=None):
            if isinstance(node, tuple):
                yield parent, node, sibling
                yield from find_edges(node[0], node, node[1])
                yield from find_edges(node[1], node, node[0])
        
        for parent, node, sibling in find_edges(tree):
            if parent is None:
                continue
            if not isinstance(node, tuple):
                continue
            C1, C2 = node
            S = sibling
            
            def replace_edge(t, target_parent, target_node, replacement):
                if isinstance(t, str):
                    return t
                left, right = t
                new_left = replace_edge(left, target_parent, target_node, replacement)
                new_right = replace_edge(right, target_parent, target_node, replacement)
                if t == target_parent:
                    if left == target_node:
                        return (replacement, right)
                    elif right == target_node:
                        return (left, replacement)
                return (new_left, new_right)
            
            new_node_1 = ((S, C1), C2)
            neighbor_1 = replace_edge(tree, parent, node, new_node_1)
            yield neighbor_1
            
            new_node_2 = ((S, C2), C1)
            neighbor_2 = replace_edge(tree, parent, node, new_node_2)
            yield neighbor_2

    current_score = score_tree(current_tree)[1]
    
    for _ in range(50):
        improved = False
        for neighbor in get_nni_neighbors(current_tree):
            score = score_tree(neighbor)[1]
            if score < current_score:
                current_tree = neighbor
                current_score = score
                improved = True
        if not improved:
            break
            
    return _to_newick(current_tree)
