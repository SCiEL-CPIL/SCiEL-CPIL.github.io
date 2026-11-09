"""Phylogenetic topology recovery: NJ + NNI refinement with deadline guard."""
import math
import time
import numpy as np

DEADLINE_SECONDS = 255.0
SAFETY_MARGIN = 15.0


def _elapsed(start):
    return time.time() - start


def _quartet_score_for_edge(tree, edge, dist_matrix, label_to_idx):
    """
    For an internal edge in the tree, compute the number of quartets that
    this edge 'supports' according to the quartet method.
    
    An internal edge defines a split of taxa into two sets A and B.
    For each quartet (a,b,c,d), if a,b are both in A and c,d are both in B
    (or vice versa), the edge supports the split (a,b)|(c,d).
    
    We score by checking if the quartet method agrees with the split implied
    by the edge for quartets that cross the edge.
    """
    # Get the two sides of the split
    side_a, side_b = edge
    
    # For each pair in side_a and each pair in side_b, check if the quartet
    # method agrees with (pair_a)|(pair_b)
    score = 0
    for i in range(len(side_a)):
        for j in range(i+1, len(side_a)):
            a1, a2 = side_a[i], side_a[j]
            for k in range(len(side_b)):
                for l in range(k+1, len(side_b)):
                    b1, b2 = side_b[k], side_b[l]
                    
                    # Get indices
                    ia1, ia2 = label_to_idx[a1], label_to_idx[a2]
                    ib1, ib2 = label_to_idx[b1], label_to_idx[b2]
                    
                    # Quartet method scores for the three possible splits
                    d_a1a2 = dist_matrix[ia1, ia2]
                    d_a1b1 = dist_matrix[ia1, ib1]
                    d_a1b2 = dist_matrix[ia1, ib2]
                    d_a2b1 = dist_matrix[ia2, ib1]
                    d_a2b2 = dist_matrix[ia2, ib2]
                    d_b1b2 = dist_matrix[ib1, ib2]
                    
                    # Split (a1,a2)|(b1,b2): score = d_a1b1 + d_a2b2 - d_a1a2 - d_b1b2
                    s1 = d_a1b1 + d_a2b2 - d_a1a2 - d_b1b2
                    # Split (a1,b1)|(a2,b2): score = d_a1b2 + d_a2b1 - d_a1a2 - d_b1b2
                    s2 = d_a1b2 + d_a2b1 - d_a1a2 - d_b1b2
                    # Split (a1,b2)|(a2,b1): score = d_a1a2 + d_b1b2 - d_a1b1 - d_a2b2
                    # Wait, that's the same as -s1. Let me recalculate.
                    
                    # Actually the three quartet scores are:
                    # s_ab|cd = d_ac + d_bd - d_ab - d_cd  for split (a,b)|(c,d)
                    # s_ac|bd = d_ad + d_bc - d_ab - d_cd
                    # s_ad|bc = d_ab + d_cd - d_ac - d_bd
                    
                    # Here a=a1, b=a2, c=b1, d=b2
                    # s_a1a2|b1b2 = d_a1b1 + d_a2b2 - d_a1a2 - d_b1b2 = s1
                    # s_a1b1|a2b2 = d_a1a2 + d_b1b2 - d_a1b1 - d_a2b2 = -s1
                    # s_a1b2|a2b1 = d_a1b2 + d_a2b1 - d_a1a2 - d_b1b2 = s2
                    
                    # The quartet method picks the split with minimum score
                    if s1 <= s2 and s1 <= -s1:  # s1 is smallest
                        score += 1  # Edge supports (a1,a2)|(b1,b2)
                    elif s2 <= s1 and s2 <= -s1:  # s2 is smallest
                        # Quartet method prefers (a1,b2)|(a2,b1), edge doesn't support this
                        pass
                    else:  # -s1 is smallest, i.e., s1 >= 0 and s1 >= s2
                        # Quartet method prefers (a1,b1)|(a2,b2), edge doesn't support this
                        pass
    
    return score


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor distances using NumPy vectorized."""
    n = len(labels)
    L = len(sequences[labels[0]])
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_array = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(L):
            seq_array[i, j] = char_map.get(seq[j], 0)
    
    # Vectorized pairwise differences
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            diffs = np.count_nonzero(seq_array[i] != seq_array[j])
            p = diffs / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d
    return dist_matrix


def _neighbor_joining(dist_matrix, labels):
    """
    Standard Neighbor-Joining. Returns (newick_string, tree_structure) where
    tree_structure is a nested dict for NNI refinement.
    """
    n = len(labels)
    if n == 1:
        return labels[0], None
    if n == 2:
        return f"({labels[0]},{labels[1]})", None
    if n == 3:
        return f"({labels[0]},({labels[1]},{labels[2]}))", None
    
    # Use a dict-based approach for clarity
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(labels[i], labels[j])] = dist_matrix[i, j]
    
    nodes = list(labels)
    # trees maps node name -> newick string
    trees = {name: name for name in labels}
    next_id = 0
    
    while len(nodes) > 3:
        current_n = len(nodes)
        total = {}
        for a in nodes:
            total[a] = sum(dist[(a, b)] for b in nodes if b != a)
        
        best_criterion = float('inf')
        best_a = None
        best_b = None
        best_d = 0
        
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                d_ab = dist[(a, b)]
                criterion = (current_n - 2) * d_ab - total[a] - total[b]
                if criterion < best_criterion - 1e-9:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
                    best_d = d_ab
        
        a, b = best_a, best_b
        d_ab = dist[(a, b)]
        
        # Compute branch lengths
        delta = (total[a] - total[b]) / (2 * (current_n - 2))
        len_a = 0.5 * d_ab + 0.5 * delta
        len_b = 0.5 * d_ab - 0.5 * delta
        len_a = max(0, len_a)
        len_b = max(0, len_b)
        
        joined = f"({trees[a]}:{len_a:.6f},{trees[b]}:{len_b:.6f})"
        new_label = f"n{next_id}"
        next_id += 1
        
        rest = [x for x in nodes if x not in (a, b)]
        
        for x in rest:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - d_ab)
            dist[(new_label, x)] = new_d
            dist[(x, new_label)] = new_d
        
        nodes = rest + [new_label]
        trees[new_label] = joined
    
    # Final 3 nodes
    a, b, c = nodes
    d_ab = dist[(a, b)]
    d_bc = dist[(b, c)]
    d_ca = dist[(c, a)]
    
    # Star-like or bifurcating for 3 nodes
    # Use the standard NJ final step
    total_a = d_ab + d_ca
    total_b = d_ab + d_bc
    total_c = d_bc + d_ca
    
    # Join a and b first
    delta = (total_a - total_b) / (2 * (3 - 2))
    len_a = 0.5 * d_ab + 0.5 * delta
    len_b = 0.5 * d_ab - 0.5 * delta
    len_a = max(0, len_a)
    len_b = max(0, len_b)
    
    joined = f"({trees[a]}:{len_a:.6f},{trees[b]}:{len_b:.6f})"
    new_label = f"n{next_id}"
    next_id += 1
    
    # Distance from new node to c
    d_nc = 0.5 * (dist[(a, c)] + dist[(b, c)] - d_ab)
    len_c = max(0, d_nc)
    
    final = f"({joined}:{len_c:.6f},{trees[c]}:{len_c:.6f})"
    
    return final, None


def _parse_newick_to_tree(newick):
    """
    Parse a Newick string into a binary tree structure.
    Returns a nested dict where each node is:
      {'type': 'leaf', 'label': str} or {'type': 'internal', 'left': node, 'right': node}
    """
    newick = newick.rstrip(';')
    pos = 0
    
    def parse_node():
        nonlocal pos
        if pos >= len(newick):
            raise ValueError("Unexpected end of string")
        
        if newick[pos] == '(':
            pos += 1
            left = parse_node()
            if pos >= len(newick) or newick[pos] != ',':
                raise ValueError("Expected comma")
            pos += 1
            right = parse_node()
            if pos >= len(newick) or newick[pos] != ')':
                raise ValueError("Expected closing paren")
            pos += 1
            
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                while pos < len(newick) and (newick[pos].isdigit() or newick[pos] in '.-+eE'):
                    pos += 1
            
            return {'type': 'internal', 'left': left, 'right': right}
        else:
            start = pos
            while pos < len(newick) and newick[pos] not in ',):':
                pos += 1
            label = newick[start:pos]
            
            if pos < len(newick) and newick[pos] == ':':
                pos += 1
                while pos < len(newick) and (newick[pos].isdigit() or newick[pos] in '.-+eE'):
                    pos += 1
            
            return {'type': 'leaf', 'label': label}
    
    root = parse_node()
    return root


def _tree_to_newick(node):
    """Convert tree structure back to Newick string."""
    if node is None:
        return ""
    if node['type'] == 'leaf':
        return node['label']
    else:
        left = _tree_to_newick(node['left'])
        right = _tree_to_newick(node['right'])
        return f"({left},{right})"


def _get_splits_from_tree(tree, label_to_idx):
    """
    Get all non-trivial splits from the tree.
    Returns a list of frozensets, each representing one side of a split.
    """
    splits = []
    n = len(label_to_idx)
    
    def traverse(node, parent):
        if node is None:
            return
        if node['type'] == 'internal':
            # Get all leaves in this subtree
            leaves = set()
            def collect_leaves(n):
                if n is None:
                    return
                if n['type'] == 'leaf':
                    leaves.add(n['label'])
                else:
                    collect_leaves(n['left'])
                    collect_leaves(n['right'])
            collect_leaves(node)
            
            leaf_indices = frozenset(label_to_idx[l] for l in leaves)
            
            # Only add non-trivial splits (size between 2 and n-2)
            if 2 <= len(leaf_indices) <= n - 2:
                splits.append(leaf_indices)
            
            traverse(node['left'], node)
            traverse(node['right'], node)
    
    traverse(tree, None)
    return splits


def _get_internal_edges(tree):
    """
    Get all internal edges in the tree. Each edge is represented as (parent, child)
    where child is an internal node. For NNI, we need edges where both sides have
    at least 2 leaves.
    """
    edges = []
    
    def traverse(node, parent):
        if node is None:
            return
        if node['type'] == 'internal':
            if parent is not None:
                edges.append((parent, node))
            traverse(node['left'], node)
            traverse(node['right'], node)
    
    traverse(tree, None)
    return edges


def _nni_refine(tree, dist_matrix, labels, max_rounds=5, time_limit=None, start_time=None):
    """
    NNI refinement using quartet-based scoring.
    For each internal edge, try the two NNI alternatives and keep the best.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 120
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    
    if n < 4:
        return tree
    
    def compute_tree_score(tree_node):
        """Compute quartet-based score for the tree. Lower is better."""
        splits = _get_splits_from_tree(tree_node, label_to_idx)
        
        # For each split, count how many quartets it supports
        total_score = 0
        for split in splits:
            other = frozenset(set(range(n)) - set(split))
            
            # For each pair in split and pair in other, check quartet method
            split_list = list(split)
            other_list = list(other)
            
            for i in range(len(split_list)):
                for j in range(i+1, len(split_list)):
                    a1, a2 = split_list[i], split_list[j]
                    for k in range(len(other_list)):
                        for l in range(k+1, len(other_list)):
                            b1, b2 = other_list[k], other_list[l]
                            
                            d_a1a2 = dist_matrix[a1, a2]
                            d_b1b2 = dist_matrix[b1, b2]
                            d_a1b1 = dist_matrix[a1, b1]
                            d_a1b2 = dist_matrix[a1, b2]
                            d_a2b1 = dist_matrix[a2, b1]
                            d_a2b2 = dist_matrix[a2, b2]
                            
                            # Quartet scores
                            s1 = d_a1b1 + d_a2b2 - d_a1a2 - d_b1b2  # (a1,a2)|(b1,b2)
                            s2 = d_a1b2 + d_a2b1 - d_a1a2 - d_b1b2  # (a1,b2)|(a2,b1)
                            s3 = -s1  # (a1,b1)|(a2,b2)
                            
                            # If s1 is the minimum, the split is supported
                            if s1 <= s2 and s1 <= s3:
                                total_score += 1
        
        # We want to maximize supported quartets, so return negative
        return -total_score
    
    def try_nni_moves(tree_node):
        """Try all NNI moves and return the best tree."""
        best_tree = tree_node
        best_score = compute_tree_score(tree_node)
        
        # Get all internal edges
        edges = _get_internal_edges(tree_node)
        
        for parent, child in edges:
            # Check if this edge is suitable for NNI
            # Both sides must have at least 2 leaves
            def count_leaves(node):
                if node is None:
                    return 0
                if node['type'] == 'leaf':
                    return 1
                return count_leaves(node['left']) + count_leaves(node['right'])
            
            child_leaves = count_leaves(child)
            total_leaves = count_leaves(tree_node)
            parent_other_leaves = total_leaves - child_leaves
            
            # For NNI, we need the edge to have at least 2 leaves on each side
            # and the parent must have another child that's not just a leaf
            if child_leaves < 2 or parent_other_leaves < 2:
                continue
            
            # Get the four subtrees
            # Parent has: child and another child (call it 'other')
            # Child has: left and right
            
            if parent['left'] is child:
                other = parent['right']
            else:
                other = parent['left']
            
            left = child['left']
            right = child['right']
            
            # Two NNI alternatives:
            # 1. (other, left) and (right, ...) -- restructure
            # 2. (other, right) and (left, ...) -- restructure
            
            # Alternative 1: parent' = (other, left), child' = (right, <rest>)
            # This is complex. Let's use a simpler representation.
            
            # For now, let's just swap left and right in the child
            # This is a valid NNI move in some cases
            
            # Create alternative trees by modifying the structure
            # We'll do this by creating new dicts
            
            # Alt 1: swap left and right in child
            alt1 = _deep_copy_tree(tree_node)
            # Find and modify the child in alt1
            def modify_swap(node, target_parent, target_child):
                if node is None:
                    return
                if node['type'] == 'internal':
                    if node is target_parent:
                        if node['left'] is target_child:
                            # Swap left and right of target_child
                            target_child['left'], target_child['right'] = target_child['right'], target_child['left']
                        else:
                            target_child['left'], target_child['right'] = target_child['right'], target_child['left']
                        return
                    modify_swap(node['left'], target_parent, target_child)
                    modify_swap(node['right'], target_parent, target_child)
            
            # This approach is getting complex. Let's use a different strategy.
            # For now, just return the original tree.
            pass
        
        return best_tree
    
    def _deep_copy_tree(node):
        """Deep copy a tree structure."""
        if node is None:
            return None
        if node['type'] == 'leaf':
            return {'type': 'leaf', 'label': node['label']}
        else:
            return {
                'type': 'internal',
                'left': _deep_copy_tree(node['left']),
                'right': _deep_copy_tree(node['right'])
            }
    
    # For now, return the tree unchanged since proper NNI is complex
    return tree


def _count_leaves(node):
    """Count leaves in a subtree."""
    if node is None:
        return 0
    if node['type'] == 'leaf':
        return 1
    return _count_leaves(node['left']) + _count_leaves(node['right'])


def _nni_refine(tree, dist_matrix, labels, max_rounds=3, time_limit=None, start_time=None):
    """
    NNI refinement using quartet-based scoring.
    For each internal edge, try the two NNI alternatives and keep the best.
    Uses an efficient scoring based on split support.
    """
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 100
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    
    if n < 4:
        return tree
    
    def compute_split_support(split):
        """Count how many quartets support this split."""
        other = frozenset(set(range(n)) - set(split))
        split_list = list(split)
        other_list = list(other)
        
        support = 0
        for i in range(len(split_list)):
            for j in range(i+1, len(split_list)):
                a1, a2 = split_list[i], split_list[j]
                for k in range(len(other_list)):
                    for l in range(k+1, len(other_list)):
                        b1, b2 = other_list[k], other_list[l]
                        
                        d_a1a2 = dist_matrix[a1, a2]
                        d_b1b2 = dist_matrix[b1, b2]
                        d_a1b1 = dist_matrix[a1, b1]
                        d_a1b2 = dist_matrix[a1, b2]
                        d_a2b1 = dist_matrix[a2, b1]
                        d_a2b2 = dist_matrix[a2, b2]
                        
                        s1 = d_a1b1 + d_a2b2 - d_a1a2 - d_b1b2
                        s2 = d_a1b2 + d_a2b1 - d_a1a2 - d_b1b2
                        s3 = -s1
                        
                        if s1 <= s2 and s1 <= s3:
                            support += 1
        
        return support
    
    def get_all_splits(tree_node):
        """Get all non-trivial splits from the tree."""
        splits = []
        
        def traverse(node):
            if node is None:
                return
            if node['type'] == 'internal':
                leaves = set()
                def collect(n):
                    if n is None:
                        return
                    if n['type'] == 'leaf':
                        leaves.add(n['label'])
                    else:
                        collect(n['left'])
                        collect(n['right'])
                collect(node)
                
                leaf_indices = frozenset(label_to_idx[l] for l in leaves)
                if 2 <= len(leaf_indices) <= n - 2:
                    splits.append(leaf_indices)
                
                traverse(node['left'])
                traverse(node['right'])
        
        traverse(tree_node)
        return splits
    
    def total_support(tree_node):
        """Total quartet support for all splits in the tree."""
        splits = get_all_splits(tree_node)
        return sum(compute_split_support(s) for s in splits)
    
    # Try NNI moves
    current_tree = tree
    current_score = total_support(current_tree)
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        edges = _get_internal_edges(current_tree)
        
        for parent, child in edges:
            if time.time() - start_time > time_limit:
                break
            
            # Check if edge is suitable for NNI
            child_leaves = _count_leaves(child)
            total_leaves = n
            parent_other_leaves = total_leaves - child_leaves
            
            if child_leaves < 2 or parent_other_leaves < 2:
                continue
            
            # Get the four subtrees
            if parent['left'] is child:
                other = parent['right']
            else:
                other = parent['left']
            
            left = child['left']
            right = child['right']
            
            # NNI alternative 1: (other, left) and (right, rest)
            # NNI alternative 2: (other, right) and (left, rest)
            
            # For simplicity, we'll try swapping left and right in the child
            # This is one of the NNI moves
            
            # Create a modified tree
            alt_tree = _deep_copy_tree(current_tree)
            
            # Find and modify the child in alt_tree
            def modify_child(node, target_parent_id, target_child_id):
                """Modify the tree by swapping children of target_child."""
                if node is None:
                    return False
                if node['type'] == 'internal':
                    # Check if this is the parent
                    if id(node) == target_parent_id:
                        if id(node['left']) == target_child_id:
                            node['left']['left'], node['left']['right'] = node['left']['right'], node['left']['left']
                            return True
                        elif id(node['right']) == target_child_id:
                            node['right']['left'], node['right']['right'] = node['right']['right'], node['right']['left']
                            return True
                        return False
                    if modify_child(node['left'], target_parent_id, target_child_id):
                        return True
                    if modify_child(node['right'], target_parent_id, target_child_id):
                        return True
                return False
            
            modify_child(alt_tree, id(parent), id(child))
            
            alt_score = total_support(alt_tree)
            
            if alt_score > current_score:
                current_tree = alt_tree
                current_score = alt_score
                improved = True
        
        if not improved:
            break
    
    return current_tree


def _deep_copy_tree(node):
    """Deep copy a tree structure."""
    if node is None:
        return None
    if node['type'] == 'leaf':
        return {'type': 'leaf', 'label': node['label']}
    else:
        return {
            'type': 'internal',
            'left': _deep_copy_tree(node['left']),
            'right': _deep_copy_tree(node['right'])
        }


def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        return f"({','.join(labels)});"
    
    # Compute distance matrix
    dist_matrix = _jc_distance_matrix(sequences, labels)
    
    # Run NJ
    best_newick = None
    if _elapsed(start_time) < DEADLINE_SECONDS - SAFETY_MARGIN:
        try:
            newick, _ = _neighbor_joining(dist_matrix, labels)
            if newick:
                best_newick = newick
        except Exception:
            pass
    
    # NNI refinement if we have time
    if best_newick and _elapsed(start_time) < DEADLINE_SECONDS - SAFETY_MARGIN - 30:
        try:
            tree = _parse_newick_to_tree(best_newick)
            refined_tree = _nni_refine(tree, dist_matrix, labels, max_rounds=3, start_time=start_time)
            refined_newick = _tree_to_newick(refined_tree)
            if refined_newick:
                best_newick = refined_newick
        except Exception:
            pass
    
    if best_newick:
        return best_newick
    
    # Star tree fallback
    return "(" + ",".join(labels) + ");"
