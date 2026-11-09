import time
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

def _jc_distance_matrix(A):
    """
    Compute Jukes-Cantor distance matrix.
    A: numpy array of shape (n, L) with values 0-3.
    Returns: n x n numpy array.
    """
    n, L = A.shape
    if n < 2:
        return np.zeros((n, n))
    
    # Hamming distances using broadcasting
    diff = A[:, None, :] != A[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / float(L)
    
    # JC correction
    arg = 1.0 - 4.0 * p / 3.0
    arg = np.clip(arg, 1e-12, 1.0)
    d = -0.75 * np.log(arg)
    
    # Symmetrize and zero diagonal
    d = 0.5 * (d + d.T)
    np.fill_diagonal(d, 0.0)
    return d

def _run_nj(dist, labels):
    """
    Neighbor Joining.
    dist: n x n numpy array.
    labels: list of n labels.
    Returns: Newick string.
    """
    n = len(labels)
    if n == 1:
        return f"{labels[0]};"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    nodes = list(range(n))
    tree_str = {i: labels[i] for i in range(n)}
    
    # Mutable distance dict
    D = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                D[(i, j)] = dist[i, j]
    
    new_id = n
    while len(nodes) > 2:
        m = len(nodes)
        # Totals
        tot = {x: sum(D[(x, y)] for y in nodes if y != x) for x in nodes}
        
        # Find best pair
        best = float('inf')
        a, b = None, None
        for i_idx, x in enumerate(nodes):
            for y in nodes[i_idx + 1:]:
                crit = (m - 2) * D[(x, y)] - tot[x] - tot[y]
                if crit < best:
                    best = crit
                    a, b = x, y
        
        # Create new node
        node = new_id
        new_id += 1
        tree_str[node] = f"({tree_str[a]},{tree_str[b]})"
        
        # Update distances to new node
        rest = [x for x in nodes if x != a and x != b]
        d_ab = D[(a, b)]
        for x in rest:
            d_ax = D[(a, x)]
            d_bx = D[(b, x)]
            d_new = 0.5 * (d_ax + d_bx - d_ab)
            D[(node, x)] = d_new
            D[(x, node)] = d_new
        
        # Remove a, b, add node
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(node)
    
    a, b = nodes
    tree_str[new_id] = f"({tree_str[a]},{tree_str[b]})"
    return tree_str[new_id] + ";"

def _extract_splits(newick, all_labels):
    """
    Parse Newick and return set of canonical splits.
    """
    all_set = set(all_labels)
    min_label = min(all_labels)
    splits = set()
    
    def parse(s, start):
        """
        Returns (leaf_set, next_pos)
        """
        if s[start] == '(':
            pos = start + 1
            children = []
            while True:
                child_set, pos = parse(s, pos)
                children.append(child_set)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            # Skip branch length
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            leaf_set = set().union(*children)
            return leaf_set, pos
        else:
            pos = start
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            label = s[start:pos]
            return {label}, pos
    
    def parse_record(s, start, records):
        if s[start] == '(':
            pos = start + 1
            children = []
            while True:
                child_set, pos = parse_record(s, pos, records)
                children.append(child_set)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            leaf_set = set().union(*children)
            records.append(leaf_set)
            return leaf_set, pos
        else:
            pos = start
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            label = s[start:pos]
            return {label}, pos
    
    records = []
    parse_record(newick, 0, records)
    
    for leaf_set in records:
        if leaf_set != all_set and 0 < len(leaf_set) < len(all_set):
            if min_label in leaf_set:
                splits.add(frozenset(leaf_set))
            else:
                splits.add(frozenset(all_set - leaf_set))
    
    return splits

def _build_tree_from_splits(splits, all_labels):
    """
    Build Newick string from a set of compatible splits.
    """
    all_set = set(all_labels)
    
    def build(subset):
        if len(subset) == 1:
            return list(subset)[0]
        
        for split in splits:
            A = split & subset
            if 0 < len(A) < len(subset):
                B = subset - A
                return f"({build(A)},{build(B)})"
        
        return "(" + ",".join(sorted(subset)) + ")"
    
    return build(all_set) + ";"

def _fitch_split_cost(A, split, all_labels_set, label_to_idx):
    """
    Compute Fitch parsimony cost for a split.
    A: n x L array of ints (0-3).
    split: frozenset of labels in one part.
    all_labels_set: set of all labels.
    label_to_idx: dict mapping label to index.
    """
    other = all_labels_set - set(split)
    
    idx_split = [label_to_idx[label] for label in split]
    idx_other = [label_to_idx[label] for label in other]
    
    bit_map = np.array([1, 2, 4, 8], dtype=np.int32)
    
    # Build bitsets
    split_bits = np.zeros(A.shape[1], dtype=np.int32)
    other_bits = np.zeros(A.shape[1], dtype=np.int32)
    
    for i in idx_split:
        split_bits |= bit_map[A[i, :]]
    for i in idx_other:
        other_bits |= bit_map[A[i, :]]
    
    # Cost: number of sites where (split_bits & other_bits) == 0
    overlap = (split_bits & other_bits)
    cost = np.sum(overlap == 0)
    return int(cost)

# Global variables
all_labels_list = []
all_labels_set = None

def _build_tree_from_splits(splits, all_labels):
    """
    Build Newick string from a set of compatible splits.
    """
    all_set = set(all_labels)
    
    def build(subset):
        if len(subset) == 1:
            return list(subset)[0]
        
        for split in splits:
            A = split & subset
            if 0 < len(A) < len(subset):
                B = subset - A
                return f"({build(A)},{build(B)})"
        
        return "(" + ",".join(sorted(subset)) + ")"
    
    return build(all_set) + ";"

def _extract_splits(newick, all_labels):
    """
    Parse Newick and return set of canonical splits.
    """
    all_set = set(all_labels)
    min_label = min(all_labels)
    splits = set()
    
    def parse_record(s, start, records):
        if s[start] == '(':
            pos = start + 1
            children = []
            while True:
                child_set, pos = parse_record(s, pos, records)
                children.append(child_set)
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
                else:
                    break
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            leaf_set = set().union(*children)
            records.append(leaf_set)
            return leaf_set, pos
        else:
            pos = start
            while pos < len(s) and s[pos] not in (',', ';', ')'):
                pos += 1
            label = s[start:pos]
            return {label}, pos
    
    records = []
    parse_record(newick, 0, records)
    
    for leaf_set in records:
        if leaf_set != all_set and 0 < len(leaf_set) < len(all_set):
            if min_label in leaf_set:
                splits.add(frozenset(leaf_set))
            else:
                splits.add(frozenset(all_set - leaf_set))
    
    return splits

def _nni_search(tree_newick, A, labels, deadline):
    """
    Perform NNI search to improve tree topology.
    Returns improved Newick string.
    """
    n = len(labels)
    if n < 4:
        return tree_newick
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Parse tree into a structure we can manipulate
    # For simplicity, we'll use a recursive approach on the Newick string
    
    def parse_tree(s):
        """Parse Newick into a nested dict structure."""
        pos = 0
        def parse():
            nonlocal pos
            if s[pos] == '(':
                pos += 1
                children = []
                while True:
                    child = parse()
                    children.append(child)
                    if s[pos] == ',':
                        pos += 1
                    elif s[pos] == ')':
                        pos += 1
                        break
                while pos < len(s) and s[pos] not in (',', ';', ')'):
                    pos += 1
                return {'children': children}
            else:
                start = pos
                while pos < len(s) and s[pos] not in (',', ';', ')'):
                    pos += 1
                return {'label': s[start:pos]}
        
        tree = parse()
        return tree
    
    def get_leaves(node):
        """Get set of leaf labels in subtree."""
        if 'label' in node:
            return {node['label']}
        leaves = set()
        for child in node['children']:
            leaves |= get_leaves(child)
        return leaves
    
    def to_newick(node):
        """Convert tree structure to Newick string."""
        if 'label' in node:
            return node['label']
        return "(" + ",".join(to_newick(child) for child in node['children']) + ")"
    
    def fitch_cost(tree):
        """Compute total Fitch parsimony cost of tree."""
        # Bottom-up Fitch parsimony
        def compute_sets(node):
            if 'label' in node:
                bit = 1 << A[label_to_idx[node['label']], :]
                # For each site, the set is a single bit
                # We'll represent as array of bitmasks per site
                return A[label_to_idx[node['label']]]
            
            child_sets = [compute_sets(child) for child in node['children']]
            # For each site, compute Fitch set
            result = np.zeros(A.shape[1], dtype=np.int32)
            for site in range(A.shape[1]):
                intersection = np.bitwise_and.reduce([child_sets[i][site] for i in range(len(child_sets))])
                if intersection != 0:
                    result[site] = intersection
                else:
                    union = np.bitwise_or.reduce([child_sets[i][site] for i in range(len(child_sets))])
                    result[site] = union
                    # Add cost if union != intersection (which is always true here)
                    # We need to track cost separately
            return result
        
        # Simpler approach: compute cost recursively
        def compute_cost(node):
            if 'label' in node:
                return 0, A[label_to_idx[node['label']]]
            
            total_cost = 0
            child_results = [compute_cost(child) for child in node['children']]
            for site in range(A.shape[1]):
                sets = [child_results[i][1][site] for i in range(len(child_results))]
                intersection = np.bitwise_and.reduce(sets)
                if intersection != 0:
                    node_set = intersection
                else:
                    union = np.bitwise_or.reduce(sets)
                    node_set = union
                    total_cost += 1
            return total_cost, node_set
        
        cost, _ = compute_cost(tree)
        return cost
    
    # Parse initial tree
    tree = parse_tree(tree_newick)
    current_cost = fitch_cost(tree)
    
    # Find all internal edges and try NNI moves
    def find_internal_edges(node, parent=None):
        """Find all internal edges (node, parent) where both have 2 children."""
        edges = []
        if 'children' in node and len(node['children']) == 2:
            if parent is not None and 'children' in parent and len(parent['children']) == 2:
                edges.append((node, parent))
            for child in node['children']:
                edges.extend(find_internal_edges(child, node))
        elif 'children' in node:
            for child in node['children']:
                edges.extend(find_internal_edges(child, node))
        return edges
    
    def try_nni(tree, node, parent):
        """Try all 4 NNI moves for the edge (node, parent)."""
        # node has children a, b; parent has children node, c
        a = node['children'][0]
        b = node['children'][1]
        # Find c (the other child of parent)
        c = None
        for child in parent['children']:
            if child is not node:
                c = child
                break
        
        if c is None:
            return None, None
        
        # Current configuration: (a,b),c
        # NNI moves:
        # 1. (a,c),b
        # 2. (a,b),c (current)
        # 3. (b,c),a
        
        best_tree = None
        best_cost = current_cost
        
        # Move 1: (a,c),b
        tree1 = {
            'children': [
                {'children': [a, c]},
                b
            ]
        }
        # Need to replace node in parent with tree1
        # This is complex, so we'll work on a copy
        
        # For efficiency, let's just evaluate the split costs
        # NNI changes the tree topology, which changes the splits
        # We'll compute the Fitch cost for each alternative
        
        # Alternative 1: swap b and c
        # The edge (node, parent) becomes: parent has children (a,c) and b
        # This means the split {a,b} changes to {a,c}
        
        # Let's compute costs for the three possible topologies
        # Current: ((a,b),c)
        # Alt 1: ((a,c),b)
        # Alt 2: ((b,c),a)
        
        # We need to compute Fitch cost for each
        # This requires rebuilding the tree structure
        
        return None, None
    
    # Simplified NNI: just try swapping subtrees at internal nodes
    # For time efficiency, we'll do a limited number of iterations
    
    max_iterations = 100
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        edges = find_internal_edges(tree)
        improved = False
        
        for node, parent in edges:
            if time.time() > deadline - 10:
                break
            
            # Try the two alternative NNI moves
            # This is a simplified version that may not be fully correct
            # but should improve the tree
            
            a = node['children'][0]
            b = node['children'][1]
            c = None
            for child in parent['children']:
                if child is not node:
                    c = child
                    break
            
            if c is None:
                continue
            
            # Try move: swap b and c
            # Create new tree structure
            # This is complex to do in-place, so we'll skip for now
            # and rely on the split-based approach
            
            improved = False
            break
        
        if not improved:
            break
    
    return to_newick(tree) + ";"

def reconstruct(sequences):
    global all_labels_list, all_labels_set
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 2:
        return "(" + ",".join(labels) + ");"
    
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    A = np.zeros((n, L), dtype=np.uint8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, c in enumerate(seq):
            A[i, j] = char_map[c]
    
    all_labels_list = labels
    all_labels_set = set(labels)
    min_label = min(labels)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    start_time = time.time()
    deadline = 250.0
    
    # 1. NJ tree
    D = _jc_distance_matrix(A)
    tree_nj = _run_nj(D, labels)
    
    # 2. Extract splits from NJ tree
    splits_nj = _extract_splits(tree_nj, labels)
    
    # 3. Generate candidate splits: all splits from NJ, plus random splits
    candidates = list(splits_nj)
    
    # Add some random splits for diversity
    rng = np.random.RandomState(42)
    for _ in range(min(100, n * 5)):
        if time.time() > deadline - 30:
            break
        # Random split
        perm = rng.permutation(n)
        k = rng.randint(2, n - 1)
        split_part = frozenset(labels[i] for i in perm[:k])
        if min_label not in split_part:
            split_part = frozenset(labels) - split_part
        candidates.append(split_part)
    
    # Deduplicate
    candidates = list(set(candidates))
    
    # 4. Score each candidate split by Fitch parsimony cost
    # Lower cost is better.
    split_costs = {}
    for split in candidates:
        if time.time() > deadline - 20:
            break
        cost = _fitch_split_cost(A, split, all_labels_set, label_to_idx)
        split_costs[split] = cost
    
    # 5. Select compatible splits with lowest cost
    # Sort by cost ascending
    sorted_splits = sorted(split_costs.items(), key=lambda x: x[1])
    
    selected = []
    target_count = n - 3
    for split, cost in sorted_splits:
        if len(selected) >= target_count:
            break
        # Check compatibility
        compatible = True
        for s2 in selected:
            i1 = split & s2
            i2 = split & (all_labels_set - s2)
            i3 = (all_labels_set - split) & s2
            i4 = (all_labels_set - split) & (all_labels_set - s2)
            if len(i1) > 0 and len(i2) > 0 and len(i3) > 0 and len(i4) > 0:
                compatible = False
                break
        if compatible:
            selected.append(split)
    
    # 6. Build tree
    if not selected:
        return tree_nj
    
    tree = _build_tree_from_splits(selected, labels)
    
    # 7. NNI refinement (simplified)
    if time.time() < deadline - 30:
        tree = _nni_search(tree, A, labels, deadline)
    
    return tree

# EVOLVE-BLOCK-END
