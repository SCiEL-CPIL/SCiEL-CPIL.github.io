"""Phylogenetic topology recovery using parsimony with TBR refinement."""
# EVOLVE-BLOCK-START
import math
import random

def _parse_newick(newick):
    """Parse Newick string into a tree structure."""
    newick = newick.strip().rstrip(';').strip()
    
    def parse(s, pos=0):
        if pos >= len(s):
            return None, pos
        if s[pos] == '(':
            pos += 1
            children = []
            while True:
                child, pos = parse(s, pos)
                if child is None:
                    break
                children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    break
                else:
                    break
            if pos < len(s) and s[pos] == ')':
                pos += 1
            if pos < len(s) and s[pos] not in ',)':
                while pos < len(s) and s[pos] not in ',)':
                    pos += 1
            return ('node', children), pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            label = s[start:pos].strip()
            return ('leaf', label), pos
    
    tree, _ = parse(newick)
    return tree

def _newick_from_tree(node):
    """Convert tree node to Newick string."""
    if node[0] == 'leaf':
        return node[1]
    children = [_newick_from_tree(child) for child in node[1]]
    return '(' + ','.join(children) + ')'

def _fitch_parity(node, sequences, site_idx):
    """Fitch parsimony for a single site. Returns (state_set, cost)."""
    if node[0] == 'leaf':
        char = sequences[node[1]][site_idx]
        return ({char}, 0)
    
    child_results = [_fitch_parity(child, sequences, site_idx) for child in node[1]]
    
    intersection = set(child_results[0][0])
    for states, _ in child_results[1:]:
        intersection = intersection & states
    
    cost = sum(c for _, c in child_results)
    
    if intersection:
        return (intersection, cost)
    else:
        union = set()
        for states, _ in child_results:
            union = union | states
        return (union, cost + 1)

def _parsimony_cost(tree, sequences):
    """Compute total parsimony cost."""
    n_sites = len(next(iter(sequences.values())))
    total = 0
    for site_idx in range(n_sites):
        _, cost = _fitch_parity(tree, sequences, site_idx)
        total += cost
    return total

def _generate_spr_moves(tree_newick):
    """Generate SPR moves (subset of TBR)."""
    tree = _parse_newick(tree_newick)
    moves = []
    
    def spr(node):
        if node[0] == 'leaf':
            return
        for i in range(len(node[1])):
            for j in range(i + 1, len(node[1])):
                # Swap children i and j
                new_children = list(node[1])
                new_children[i], new_children[j] = new_children[j], new_children[i]
                new_node = ('node', new_children)
                moves.append(_newick_from_tree(new_node))
            for child in node[1]:
                spr(child)
    
    spr(tree)
    return moves

def _tbr_refine(tree, sequences, max_rounds=10):
    """TBR refinement."""
    current_tree = tree
    current_cost = _parsimony_cost(_parse_newick(current_tree), sequences)
    
    for _ in range(max_rounds):
        best_cost = current_cost
        best_tree = current_tree
        improved = False
        
        for move in _generate_spr_moves(current_tree):
            cost = _parsimony_cost(_parse_newick(move), sequences)
            if cost < best_cost:
                best_cost = cost
                best_tree = move
                improved = True
        
        if not improved:
            break
        
        current_tree = best_tree
        current_cost = best_cost
    
    return current_tree

def _star_tree(labels):
    """Create a star tree."""
    return '(' + ','.join(labels) + ')'

def _greedy_addition(sequences, labels):
    """Build tree using greedy parsimony addition."""
    if len(labels) <= 2:
        if len(labels) == 1:
            return labels[0]
        return f"({labels[0]},{labels[1]})"
    
    # Start with 3 taxa
    tree = f"({labels[0]},{labels[1]},{labels[2]})"
    
    for label in labels[3:]:
        best_cost = float('inf')
        best_tree = None
        
        # Try inserting at each possible position
        candidates = _generate_insertions(tree, label)
        
        for cand in candidates:
            cost = _parsimony_cost(_parse_newick(cand), sequences)
            if cost < best_cost:
                best_cost = cost
                best_tree = cand
        
        tree = best_tree
    
    return tree

def _generate_insertions(tree_newick, new_label):
    """Generate all possible trees by inserting new_label."""
    tree = _parse_newick(tree_newick)
    candidates = []
    
    def insert(node):
        if node[0] == 'leaf':
            new_node = ('node', [node, ('leaf', new_label)])
            candidates.append(_newick_from_tree(new_node))
        else:
            for i in range(len(node[1])):
                child = node[1][i]
                new_child = ('node', [child, ('leaf', new_label)])
                new_children = [node[1][j] if j != i else new_child for j in range(len(node[1]))]
                new_node = ('node', new_children)
                candidates.append(_newick_from_tree(new_node))
            for child in node[1]:
                insert(child)
    
    insert(tree)
    return list(set(candidates))

def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree using parsimony with TBR refinement.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(next(iter(sequences.values())))
    
    if n_taxa == 1:
        return labels[0]
    if n_taxa == 2:
        return f"({labels[0]},{labels[1]})"
    if n_taxa == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # Build initial tree
    if n_taxa <= 12:
        tree = _greedy_addition(sequences, labels)
    else:
        tree = _star_tree(labels)
    
    # Refine with TBR if time allows
    if n_sites <= 400:
        max_rounds = 10 if n_taxa <= 10 else 5
        tree = _tbr_refine(tree, sequences, max_rounds)
    
    return tree
# EVOLVE-BLOCK-END
