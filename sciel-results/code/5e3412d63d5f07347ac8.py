"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
'''Parsimony-based phylogenetic reconstruction using Stepwise Addition.
   Uses Fitch's algorithm for parsimony scoring, optimized with NumPy bitmasks.
   This approach is distinct from distance-based methods like NJ.'''
import numpy as np

def _sequences_to_bitmasks(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    char_to_bit = {'A': 1, 'C': 2, 'G': 4, 'T': 8}
    
    X = np.zeros((n_sites, n_taxa), dtype=np.uint8)
    for i, label in enumerate(labels):
        for j, char in enumerate(sequences[label]):
            X[j, i] = char_to_bit[char]
            
    return X, labels

def _fitch_score(tree, X):
    if isinstance(tree, int):
        return 0, X[:, tree]
    
    left_score, left_states = _fitch_score(tree[0], X)
    right_score, right_states = _fitch_score(tree[1], X)
    
    inter = left_states & right_states
    cost = np.sum(inter == 0)
    union = left_states | right_states
    states = np.where(inter == 0, union, inter)
    
    return left_score + right_score + cost, states

def _get_branches(tree):
    """Return a list of (parent_tuple, child) pairs."""
    branches = []
    def traverse(node, parent=None):
        if isinstance(node, int):
            if parent is not None:
                branches.append((parent, node))
        else:
            if parent is not None:
                branches.append((parent, node))
            traverse(node[0], node)
            traverse(node[1], node)
    traverse(tree)
    return branches

def _insert_leaf(tree, parent, child, new_leaf):
    """Insert new_leaf on the branch (parent, child)."""
    def replace(node):
        if isinstance(node, int):
            return node
        left, right = node
        new_left = replace(left)
        new_right = replace(right)
        # If this node is the parent, and one of its children is the child
        if node == parent:
            if left == child:
                return (new_right, (new_left, new_leaf))
            elif right == child:
                return (new_left, (new_right, new_leaf))
        return (new_left, new_right)
    return replace(tree)

def _tree_to_newick(tree, labels):
    if isinstance(tree, int):
        return labels[tree]
    left, right = tree
    return f"({_tree_to_newick(left, labels)},{_tree_to_newick(right, labels)})"

def reconstruct(sequences):
    X, labels = _sequences_to_bitmasks(sequences)
    n_taxa = len(labels)
    
    if n_taxa < 2:
        return labels[0] + ";" if labels else ";"
    
    # Start with first 3 taxa
    current_tree = (0, (1, 2))
    
    for new_idx in range(3, n_taxa):
        best_score = float('inf')
        best_tree = current_tree
        
        branches = _get_branches(current_tree)
        for parent, child in branches:
            candidate_tree = _insert_leaf(current_tree, parent, child, new_idx)
            score, _ = _fitch_score(candidate_tree, X)
            if score < best_score:
                best_score = score
                best_tree = candidate_tree
        
        current_tree = best_tree
        
    return _tree_to_newick(current_tree, labels) + ";"

# EVOLVE-BLOCK-END
