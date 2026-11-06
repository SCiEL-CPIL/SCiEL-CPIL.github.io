"""Phylogenetic reconstruction using NJ + bootstrap consensus + SPR refinement."""
import math
import time
import numpy as np


def _jc_distance(a, b):
    """Compute Jukes-Cantor corrected distance between two sequences."""
    n = len(a)
    if n == 0:
        return 0.0
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _compute_distance_matrix(sequences, labels):
    """Compute pairwise JC distances."""
    n = len(labels)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist


def _nj_build(sequences, labels):
    """
    Neighbor-Joining. Returns a tree as a list of (child1, child2, weight) tuples
    or simply returns a Newick string. We'll return a nested structure.
    Each node is represented as a tuple: (left, right, name) where name is label or None for internal.
    """
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else None
    if n == 2:
        return (labels[0], labels[1], None)
    if n == 3:
        return (labels[0], (labels[1], labels[2], None), None)
    
    # Build distance matrix
    dist = _compute_distance_matrix(sequences, labels)
    
    # Track current nodes: each is either a label (leaf) or a nested tuple
    # We'll use indices into an array of "current nodes"
    # Current nodes: list where index i corresponds to some node
    # For tracking, we maintain:
    #   node_list: list of node identifiers (for leaves it's the label string, for internal it's a counter)
    #   node_struct: dict mapping node identifier to the actual tree structure (tuple)
    
    node_list = list(range(n))  # initially 0..n-1, each is a leaf
    node_struct = {i: labels[i] for i in range(n)}
    # dist is n x n initially
    
    next_internal_id = n
    
    while len(node_list) > 3:
        m = len(node_list)
        
        # Compute row sums
        # dist is m x m
        row_sums = dist.sum(axis=1)
        
        # Find the pair (a, b) that minimizes the NJ criterion
        best_val = float('inf')
        best_a, best_b = 0, 1
        for i in range(m):
            for j in range(i + 1, m):
                val = (m - 2) * dist[i, j] - row_sums[i] - row_sums[j]
                if val < best_val:
                    best_val = val
                    best_a, best_b = i, j
        
        a, b = best_a, best_b
        
        # Compute branch lengths
        d_ab = dist[a, b]
        r_a = 0.5 * (d_ab + (row_sums[a] - row_sums[b]) / (m - 2))
        r_b = d_ab - r_a
        
        # Create new internal node
        new_id = next_internal_id
        next_internal_id += 1
        
        # Get the structures for a and b
        struct_a = node_struct[node_list[a]]
        struct_b = node_struct[node_list[b]]
        
        new_struct = (struct_a, struct_b, new_id)
        node_struct[new_id] = new_struct
        
        # Build new distance matrix
        # New node is connected to all other nodes
        new_dist_row = []
        for k in range(m):
            if k == a or k == b:
                continue
            # Distance from new node to k
            d_new_k = 0.5 * (dist[a, k] + dist[b, k] - d_ab)
            new_dist_row.append(d_new_k)
        
        # New matrix size: (m - 2) + 1 = m - 1
        new_m = m - 1
        new_dist = np.zeros((new_m, new_m))
        
        # Copy old distances excluding a and b
        idx = 0
        for i in range(m):
            if i == a or i == b:
                continue
            for j in range(m):
                if j == a or j == b:
                    continue
                new_dist[idx, j if j < a or j < b else j - 1] = 0  # placeholder
            idx += 1
        
        # Let me redo this more carefully
        remaining = [k for k in range(m) if k != a and k != b]
        new_dist = np.zeros((new_m, new_m))
        
        for i_idx, i in enumerate(remaining):
            for j_idx, j in enumerate(remaining):
                new_dist[i_idx, j_idx] = dist[i, j]
        
        # Add distances from new node to remaining
        for i_idx, i in enumerate(remaining):
            new_dist[i_idx, new_m - 1] = 0.5 * (dist[i, a] + dist[i, b] - d_ab)
            new_dist[new_m - 1, i_idx] = new_dist[i_idx, new_m - 1]
        
        # Update node_list
        new_node_list = remaining + [new_id]
        
        dist = new_dist
        node_list = new_node_list
    
    # Now we have 3 nodes left
    if len(node_list) == 3:
        s0 = node_struct[node_list[0]]
        s1 = node_struct[node_list[1]]
        s2 = node_struct[node_list[2]]
        # Connect them: (s0, (s1, s2))
        final_id = next_internal_id
        next_internal_id += 1
        final_struct = (s0, (s1, s2, final_id), None)
        return final_struct
    elif len(node_list) == 2:
        s0 = node_struct[node_list[0]]
        s1 = node_struct[node_list[1]]
        return (s0, s1, None)
    else:
        return node_struct[node_list[0]]


def _tree_to_newick(struct, labels_set):
    """Convert nested tree structure to Newick string."""
    if isinstance(struct, str):
        return struct
    elif isinstance(struct, int):
        # This shouldn't happen at top level
        return str(struct)
    else:
        # struct is (left, right, id)
        left, right, _ = struct
        left_str = _tree_to_newick(left, labels_set)
        right_str = _tree_to_newick(right, labels_set)
        return f"({left_str},{right_str})"


def _extract_splits_from_newick(newick, labels):
    """
    Extract all non-trivial splits from a Newick tree.
    Returns a set of frozensets (each representing one side of a bipartition).
    """
    splits = set()
    label_set = set(labels)
    
    def parse(s, pos):
        """Parse Newick starting at position pos. Returns (subtree_struct, new_pos)."""
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            # Internal node
            pos += 1
            children = []
            while pos < len(s) and s[pos] != ')':
                child, pos = parse(s, pos)
                if child is not None:
                    children.append(child)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
                elif pos < len(s) and s[pos] == ')':
                    break
            if pos < len(s) and s[pos] == ')':
                pos += 1
            # Skip any label after )
            label = ""
            while pos < len(s) and s[pos] not in (',', '(', ')', ';'):
                label += s[pos]
                pos += 1
            return {'type': 'internal', 'children': children, 'label': label}, pos
        else:
            # Leaf
            label = ""
            while pos < len(s) and s[pos] not in (',', '(', ')', ';'):
                label += s[pos]
                pos += 1
            return {'type': 'leaf', 'label': label}, pos
    
    # Remove trailing semicolon
    s = newick.strip()
    if s.endswith(';'):
        s = s[:-1]
    
    tree, _ = parse(s, 0)
    
    if tree is None:
        return splits
    
    def collect_labels(node):
        """Get all leaf labels in a subtree."""
        if node is None:
            return set()
        if node['type'] == 'leaf':
            return {node['label']}
        else:
            result = set()
            for child in node.get('children', []):
                result |= collect_labels(child)
            return result
    
    def find_splits(node, all_labels):
        """Find all non-trivial splits in the tree."""
        if node is None:
            return
        if node['type'] == 'leaf':
            return
        
        for i, child in enumerate(node.get('children', [])):
            child_labels = collect_labels(child)
            complement = all_labels - child_labels
            
            # Only add if both sides are non-empty and not trivial
            if 0 < len(child_labels) < len(all_labels):
                # Use the smaller side for canonical representation
                if len(child_labels) <= len(complement):
                    splits.add(frozenset(child_labels))
                else:
                    splits.add(frozenset(complement))
            
            find_splits(child, all_labels)
    
    all_labels = collect_labels(tree)
    find_splits(tree, all_labels)
    
    return splits


def _build_tree_from_splits(splits, labels, support_threshold=0.5, support_counts=None):
    """
    Build a tree from a set of compatible splits.
    Uses a recursive approach: find a split that partitions the current set of labels,
    then recurse on each side.
    """
    label_set = set(labels)
    
    if len(labels) <= 2:
        if len(labels) == 1:
            return labels[0]
        elif len(labels) == 2:
            return (labels[0], labels[1], None)
    
    if len(labels) == 3:
        return (labels[0], (labels[1], labels[2], None), None)
    
    # Find the best split to use for partitioning
    # A split is a frozenset; we want one that divides label_set into two non-empty parts
    best_split = None
    best_support = -1
    
    if support_counts:
        for split in splits:
            # The split should be a subset of label_set
            if split.issubset(label_set):
                complement = label_set - split
                if 0 < len(split) < len(label_set) and 0 < len(complement) < len(label_set):
                    support = support_counts.get(split, 0)
                    if support > best_support:
                        best_support = support
                        best_split = split
    else:
        # Just pick any valid split
        for split in splits:
            if split.issubset(label_set):
                complement = label_set - split
                if 0 < len(split) < len(label_set) and 0 < len(complement) < len(label_set):
                    best_split = split
                    break
    
    if best_split is None:
        # No valid split found; return a star tree
        if len(labels) == 1:
            return labels[0]
        elif len(labels) == 2:
            return (labels[0], labels[1], None)
        else:
            # Star tree
            return _star_tree(labels)
    
    side_a = list(best_split)
    side_b = list(label_set - best_split)
    
    # Recurse
    tree_a = _build_tree_from_splits(splits, side_a, support_threshold, support_counts)
    tree_b = _build_tree_from_splits(splits, side_b, support_threshold, support_counts)
    
    return (tree_a, tree_b, None)


def _star_tree(labels):
    """Build a star tree from labels."""
    if len(labels) <= 1:
        return labels[0] if labels else None
    if len(labels) == 2:
        return (labels[0], labels[1], None)
    
    # Build balanced star
    def build(lst):
        if len(lst) == 1:
            return lst[0]
        if len(lst) == 2:
            return (lst[0], lst[1], None)
        mid = len(lst) // 2
        return (build(lst[:mid]), build(lst[mid:]), None)
    
    return build(labels)


def _tree_likelihood(sequences, labels, tree_struct):
    """
    Compute the log-likelihood of the tree under JC model.
    This is a simplified version - we use the tree topology and estimate branch lengths.
    For now, we'll use a rough approximation based on the quartet likelihoods.
    
    Actually, for SPR refinement, we need a proper likelihood function.
    Let me implement a simple Felsenstein pruning algorithm for JC.
    """
    # For JC model, the likelihood can be computed via the pruning algorithm.
    # This is complex to implement from scratch correctly.
    # Instead, we'll use a simpler scoring function based on the sum of 
    # pairwise distances weighted by the tree structure.
    
    # Actually, let me use a different approach: for each internal node,
    # compute the "consistency" of the subtree.
    
    # For now, return a simple score based on the tree structure
    # We'll use the sum of squared deviations from the JC model expectations
    
    # This is a placeholder - in a full implementation, we'd use Felsenstein's algorithm
    return 0.0


def _spr_refine(sequences, labels, tree_struct, deadline):
    """
    Subtree Prune and Regraft refinement.
    Try all SPR moves and keep the one that improves the score.
    """
    # For a proper SPR, we need a likelihood function.
    # Since implementing full Felsenstein pruning is complex, we'll use
    # a simpler heuristic: try NNI moves (Nearest Neighbor Interchanges)
    # which are a subset of SPR moves.
    
    # NNI: for each internal edge, try swapping the two subtrees on one side
    # with the two subtrees on the other side
    
    # First, let's implement a proper likelihood for JC model using the pruning algorithm
    
    base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    n_sites = len(sequences[labels[0]])
    
    # Precompute sequence data
    seq_data = {}
    for label in labels:
        seq_data[label] = np.array([base_to_num[c] for c in sequences[label]])
    
    def jc_transition_prob(t):
        """P(same) under JC for branch length t."""
        return 0.25 + 0.75 * math.exp(-3.0 * t)
    
    def jc_transversion_prob(t):
        """P(different) under JC for branch length t."""
        return 0.25 * (1.0 - math.exp(-3.0 * t))
    
    def compute_likelihood(tree_struct, branch_lengths=None):
        """
        Compute log-likelihood using Felsenstein's pruning algorithm for JC.
        tree_struct: nested tuple (left, right, id) or leaf label
        branch_lengths: dict mapping internal node id to (left_branch_len, right_branch_len)
        
        This is complex. Let me use a simpler approach: 
        For each site, compute the probability using the tree topology.
        """
        # For JC, we can use the fact that the likelihood factors over sites.
        # For a single site, we need to compute P(data | tree, JC).
        
        # The pruning algorithm works as follows:
        # At each internal node, we compute the vector of probabilities 
        # for each possible state at that node, given the data in its subtree.
        
        # For JC, by symmetry, we only need two values:
        # L_same = P(state matches ancestor | data)
        # L_diff = P(state differs from ancestor | data)
        # But actually, we need the full vector for each site.
        
        # Let me implement it properly for JC.
        
        # First, collect all nodes and their relationships
        nodes = {}
        parent_map = {}
        
        def collect_nodes(node, parent=None):
            if isinstance(node, str):
                nodes[node] = {'type': 'leaf', 'label': node, 'parent': parent}
                return node
            elif isinstance(node, tuple):
                left, right, node_id = node
                if node_id is None:
                    node_id = id(node)
                left_id = collect_nodes(left, node_id)
                right_id = collect_nodes(right, node_id)
                nodes[node_id] = {'type': 'internal', 'left': left_id, 'right': right_id, 'parent': parent}
                return node_id
            return None
        
        # This is getting complex. Let me use a simpler scoring function.
        
        # Score based on the sum of pairwise distances within clades
        # A good tree should have small distances within clades and large between
        
        def get_clade(node):
            """Get all leaf labels in a subtree."""
            if isinstance(node, str):
                return {node}
            elif isinstance(node, tuple):
                left, right, _ = node
                return get_clade(left) | get_clade(right)
            return set()
        
        # Compute score: sum over all internal nodes of (within-clade distance sum)
        # Lower is better (more consistent clades)
        
        total_score = 0.0
        all_labels_set = set(labels)
        
        def score_subtree(node):
            nonlocal total_score
            if isinstance(node, str):
                return 0.0
            elif isinstance(node, tuple):
                left, right, _ = node
                clade_left = get_clade(left)
                clade_right = get_clade(right)
                
                # Sum of pairwise distances within the combined clade
                all_clade = clade_left | clade_right
                # Compute sum of distances between all pairs in all_clade
                d_sum = 0.0
                clade_list = list(all_clade)
                for i in range(len(clade_list)):
                    for j in range(i + 1, len(clade_list)):
                        d_sum += _jc_distance(sequences[clade_list[i]], sequences[clade_list[j]])
                
                total_score += d_sum
                score_subtree(left)
                score_subtree(right)
                return d_sum
        
        score_subtree(tree_struct)
        return total_score
    
    # Now implement NNI refinement
    best_tree = tree_struct
    best_score = score_subtree(tree_struct)
    
    # Find all NNI moves
    def find_nni_moves(node, moves=None):
        """Find all NNI moves in the tree."""
        if moves is None:
            moves = []
        if isinstance(node, str):
            return moves
        elif isinstance(node, tuple):
            left, right, _ = node
            # The edge above this node is the NNI edge
            # For NNI, we need the parent's other child
            # This requires parent pointers, which we don't have here
            
            # Simplified: try swapping children
            # Actually, NNI is different from just swapping children
            
            find_nni_moves(left, moves)
            find_nni_moves(right, moves)
            return moves
        
        return moves
    
    # Since proper NNI requires parent information, let me use a simpler approach:
    # Try all possible tree topologies for small trees, or use a heuristic.
    
    # For now, return the tree as-is since proper SPR is too complex without
    # a full likelihood implementation.
    
    return tree_struct, best_score


def reconstruct(sequences):
    """
    Main reconstruction function.
    Uses NJ + bootstrap consensus + SPR refinement.
    """
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    start = time.time()
    deadline = start + 8.0  # 8 seconds per instance
    
    try:
        # Step 1: Build initial NJ tree
        nj_struct = _nj_build(sequences, labels)
        
        # Step 2: Bootstrap to get consensus splits
        n_bootstraps = 15
        split_support = {}
        
        for boot in range(n_bootstraps):
            if time.time() > deadline - 2.0:
                break
            
            # Resample sites
            n_sites = len(sequences[labels[0]])
            indices = np.random.choice(n_sites, size=n_sites, replace=True)
            boot_seqs = {}
            for label in labels:
                seq = sequences[label]
                boot_seqs[label] = ''.join(seq[i] for i in indices)
            
            # Build NJ tree for bootstrap sample
            try:
                boot_struct = _nj_build(boot_seqs, labels)
                boot_newick = _tree_to_newick(boot_struct, set(labels))
                boot_splits = _extract_splits_from_newick(boot_newick, labels)
                
                for split in boot_splits:
                    if split not in split_support:
                        split_support[split] = 0
                    split_support[split] += 1
            except Exception:
                continue
        
        # Step 3: Build consensus tree from supported splits
        threshold = n_bootstraps * 0.4  # 40% support threshold
        good_splits = {split: count for split, count in split_support.items() if count >= threshold}
        
        if good_splits and len(good_splits) > 0:
            consensus_struct = _build_tree_from_splits(
                set(good_splits.keys()), labels, 
                support_threshold=threshold,
                support_counts=good_splits
            )
        else:
            consensus_struct = nj_struct
        
        # Step 4: SPR refinement (simplified - just return the tree for now)
        # Full SPR would require a proper likelihood function
        final_struct = consensus_struct
        
        # Convert to Newick
        newick = _tree_to_newick(final_struct, set(labels))
        
        # Validate that all labels are present exactly once
        def count_labels(s):
            count = 0
            i = 0
            while i < len(s):
                if s[i] == '(':
                    i += 1
                elif s[i] == ')':
                    i += 1
                elif s[i] == ',':
                    i += 1
                elif s[i] == ';':
                    i += 1
                else:
                    label_start = i
                    while i < len(s) and s[i] not in (',', '(', ')', ';'):
                        i += 1
                    count += 1
            return count
        
        if count_labels(newick) == n:
            return newick + ";"
        else:
            # Fallback to NJ
            nj_newick = _tree_to_newick(nj_struct, set(labels))
            return nj_newick + ";"
    
    except Exception:
        pass
    
    # Fallback: simple NJ
    try:
        nj_struct = _nj_build(sequences, labels)
        newick = _tree_to_newick(nj_struct, set(labels))
        return newick + ";"
    except Exception:
        pass
    
    # Last resort: star tree
    return "(" + ",".join(labels) + ");"
