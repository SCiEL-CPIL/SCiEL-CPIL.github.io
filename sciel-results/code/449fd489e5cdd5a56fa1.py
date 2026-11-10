"""Phylogenetic topology recovery using Maximum Parsimony with SPR refinement."""
import math
import numpy as np
from itertools import combinations


def _parsimony_score(tree_nodes, char_matrix, n, sites):
    """Compute parsimony score using Fitch's algorithm."""
    total_score = 0
    for s in range(sites):
        sets = {}
        children = {}
        for node, (l, r) in tree_nodes.items():
            children[node] = (l, r)
        
        visited = set()
        def postorder(node):
            nonlocal total_score
            if node in visited:
                return
            if node < n:
                sets[node] = (char_matrix[node, s],)
                visited.add(node)
                return
            l, r = children[node]
            postorder(l)
            postorder(r)
            visited.add(node)
            sl = sets[l]
            sr = sets[r]
            if sl[0] in sr:
                sets[node] = (sl[0],)
            else:
                sets[node] = sl + sr
                total_score += 1
        
        all_children = set()
        for node, (l, r) in tree_nodes.items():
            all_children.add(l)
            all_children.add(r)
        root = None
        for node in tree_nodes:
            if node not in all_children:
                root = node
                break
        
        if root is not None:
            postorder(root)
    
    return total_score


def _reconstruct_parity(sequences, labels):
    """Maximum Parsimony with stepwise addition + SPR refinement."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    sites = len(sequences[labels[0]])
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    char_matrix = np.zeros((n, sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(sites):
            char_matrix[i, j] = enc[seq[j]]
    
    # Stepwise addition
    best3 = None
    best3_score = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                score = np.sum(char_matrix[i] != char_matrix[j]) + \
                        np.sum(char_matrix[i] != char_matrix[k]) + \
                        np.sum(char_matrix[j] != char_matrix[k])
                if score < best3_score:
                    best3_score = score
                    best3 = (i, j, k)
    
    i, j, k = best3
    tree_nodes = {}
    next_internal = n
    node_jk = next_internal
    next_internal += 1
    tree_nodes[node_jk] = (j, k)
    root = next_internal
    next_internal += 1
    tree_nodes[root] = (i, node_jk)
    
    remaining = [x for x in range(n) if x not in (i, j, k)]
    
    for taxon in remaining:
        best_edge = None
        best_score = float('inf')
        
        parent_map = {}
        for node, (l, r) in tree_nodes.items():
            parent_map[l] = node
            parent_map[r] = node
        
        edges = []
        for node, (l, r) in tree_nodes.items():
            edges.append((node, l))
            edges.append((node, r))
        
        for parent, child in edges:
            old_tree = dict(tree_nodes)
            new_node_id = next_internal
            tree_nodes[new_node_id] = (taxon, child)
            if parent in tree_nodes:
                l, r = tree_nodes[parent]
                if l == child:
                    tree_nodes[parent] = (new_node_id, r)
                else:
                    tree_nodes[parent] = (l, new_node_id)
            
            score = _parsimony_score(tree_nodes, char_matrix, n, sites)
            
            if score < best_score:
                best_score = score
                best_edge = (parent, child, new_node_id)
            
            tree_nodes = dict(old_tree)
        
        parent, child, new_node_id = best_edge
        tree_nodes[new_node_id] = (taxon, child)
        if parent in tree_nodes:
            l, r = tree_nodes[parent]
            if l == child:
                tree_nodes[parent] = (new_node_id, r)
            else:
                tree_nodes[parent] = (l, new_node_id)
        
        next_internal += 1
    
    # SPR refinement
    def spr_refine(tree_nodes, max_iter=15):
        best_score = _parsimony_score(tree_nodes, char_matrix, n, sites)
        improved = True
        iterations = 0
        while improved and iterations < max_iter:
            improved = False
            iterations += 1
            next_internal = n + len(tree_nodes)
            
            parent_map = {}
            for node, (l, r) in tree_nodes.items():
                parent_map[l] = node
                parent_map[r] = node
            
            for leaf in range(n):
                if leaf not in parent_map:
                    continue
                parent = parent_map[leaf]
                if parent not in tree_nodes:
                    continue
                
                old_tree = dict(tree_nodes)
                del tree_nodes[parent]
                
                new_parent_map = {}
                for node, (cl, cr) in tree_nodes.items():
                    new_parent_map[cl] = node
                    new_parent_map[cr] = node
                
                edges_now = []
                for node, (cl, cr) in tree_nodes.items():
                    edges_now.append((node, cl))
                    edges_now.append((node, cr))
                
                best_regraft_score = float('inf')
                best_regraft = None
                
                for p2, c2 in edges_now:
                    new_node_id = next_internal
                    state_before = dict(tree_nodes)
                    tree_nodes[new_node_id] = (leaf, c2)
                    l2, r2 = tree_nodes[p2]
                    if l2 == c2:
                        tree_nodes[p2] = (new_node_id, r2)
                    else:
                        tree_nodes[p2] = (l2, new_node_id)
                    
                    score = _parsimony_score(tree_nodes, char_matrix, n, sites)
                    
                    if score < best_regraft_score:
                        best_regraft_score = score
                        best_regraft = (p2, c2, new_node_id)
                    
                    tree_nodes = dict(state_before)
                
                if best_regraft is not None:
                    p2, c2, new_node_id = best_regraft
                    tree_nodes[new_node_id] = (leaf, c2)
                    l2, r2 = tree_nodes[p2]
                    if l2 == c2:
                        tree_nodes[p2] = (new_node_id, r2)
                    else:
                        tree_nodes[p2] = (l2, new_node_id)
                    next_internal += 1
                    if best_regraft_score < best_score:
                        best_score = best_regraft_score
                        improved = True
                    else:
                        tree_nodes = dict(old_tree)
                else:
                    tree_nodes = dict(old_tree)
        
        return tree_nodes
    
    tree_nodes = spr_refine(tree_nodes, max_iter=15)
    
    # Build Newick
    all_children = set()
    for node, (l, r) in tree_nodes.items():
        all_children.add(l)
        all_children.add(r)
    root = None
    for node in tree_nodes:
        if node not in all_children:
            root = node
            break
    
    def to_newick(node):
        if node < n:
            return labels[node]
        l, r = tree_nodes[node]
        return f"({to_newick(l)},{to_newick(r)})"
    
    return to_newick(root) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    return _reconstruct_parity(sequences, labels)
