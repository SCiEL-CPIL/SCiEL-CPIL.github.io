"""Greedy Parsimony Construction for phylogenetic topology recovery."""
# EVOLVE-BLOCK-START
import math
import numpy as np


def _parsimony_score(tree_internal, seqs_int, n_taxa):
    """
    Calculate the total parsimony score for a given tree.
    tree_internal: list of (left, right) tuples. 
                   Index k in list corresponds to global node index n_taxa + k.
    seqs_int: list of integer arrays, shape (n_taxa, L)
    """
    L = seqs_int[0].shape[0]
    
    # Masks for leaves: shape (n_taxa, L), values 0..15
    leaf_masks = (1 << seqs_int).astype(np.uint32)
    
    num_nodes = 2 * n_taxa - 1
    node_masks = np.zeros((num_nodes, L), dtype=np.uint32)
    
    # Set leaf masks
    node_masks[:n_taxa, :] = leaf_masks
    
    total_cost = 0
    # Process internal nodes in reverse order of their index in the list
    # This ensures children are processed before parents
    for k in range(len(tree_internal) - 1, -1, -1):
        left, right = tree_internal[k]
        node_idx = n_taxa + k
        
        mask_l = node_masks[left, :]
        mask_r = node_masks[right, :]
        
        # Intersection
        inter = mask_l & mask_r
        # Union
        union = mask_l | mask_r
        
        # Cost is 1 where inter is 0
        cost = (inter == 0).astype(np.uint32)
        
        # New mask is inter if inter != 0, else union
        new_mask = np.where(cost == 1, union, inter)
        
        node_masks[node_idx, :] = new_mask
        total_cost += np.sum(cost)
        
    return total_cost


def _build_tree_greedy(sequences, labels):
    """
    Build a tree using greedy parsimony insertion.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Convert sequences to int arrays
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs_int = []
    for lab in labels:
        s = sequences[lab]
        arr = np.array([char_map[c] for c in s], dtype=np.uint8)
        seqs_int.append(arr)
    seqs_int = np.array(seqs_int) # shape (n, L)
    
    # Start with 3 taxa
    # Tree: 3:(0,1), 4:(3,2)
    tree_internal = [(0, 1), (3, 2)]
    
    for next_idx in range(3, n):
        # Get edges
        # We need to find the root and traverse
        n_leaves_current = next_idx
        total_nodes_current = 2 * n_leaves_current - 1
        
        is_child = [False] * total_nodes_current
        for l, r in tree_internal:
            is_child[l] = True
            is_child[r] = True
        
        root = -1
        for i in range(total_nodes_current):
            if not is_child[i]:
                root = i
                break
        
        edges = []
        stack = [(root, -1)]
        while stack:
            node, parent = stack.pop()
            if parent != -1:
                edges.append((parent, node))
            if node >= n_leaves_current:
                l, r = tree_internal[node - n_leaves_current]
                stack.append((l, node))
                stack.append((r, node))
        
        best_edge = None
        best_score = float('inf')
        best_new_tree = None
        
        for parent, child in edges:
            # Create new tree
            new_tree = [tuple(t) for t in tree_internal]
            
            # Parent must be internal
            if parent < n_leaves_current:
                continue # Leaf parent, invalid edge for this representation? 
                         # Actually, if parent is leaf, it's a terminal edge, 
                         # but in binary tree, leaves don't have children.
                         # So parent must be internal.
            
            p_idx = parent - n_leaves_current
            l, r = new_tree[p_idx]
            
            new_node_global = n_leaves_current + len(new_tree)
            
            if l == child:
                new_tree[p_idx] = (new_node_global, r)
            elif r == child:
                new_tree[p_idx] = (l, new_node_global)
            else:
                continue
            
            new_tree.append((child, next_idx))
            
            score = _parsimony_score(new_tree, seqs_int[:next_idx+1], next_idx + 1)
            
            if score < best_score:
                best_score = score
                best_new_tree = new_tree
                best_edge = (parent, child)
                
        if best_new_tree is not None:
            tree_internal = best_new_tree
        else:
            # Fallback
            last_idx = len(tree_internal) - 1
            l, r = tree_internal[last_idx]
            new_node_global = next_idx + len(tree_internal)
            tree_internal[last_idx] = (new_node_global, r)
            tree_internal.append((l, next_idx))
            
    # Convert to Newick
    n_leaves = n
    def get_node(idx):
        if idx < n_leaves:
            return labels[idx]
        else:
            l, r = tree_internal[idx - n_leaves]
            return f"({get_node(l)},{get_node(r)})"
    
    is_child = [False] * (2 * n_leaves - 1)
    for l, r in tree_internal:
        is_child[l] = True
        is_child[r] = True
        
    root = -1
    for i in range(2 * n_leaves - 1):
        if not is_child[i]:
            root = i
            break
            
    return get_node(root) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    return _build_tree_greedy(sequences, labels)
# EVOLVE-BLOCK-END
