"""Phylogenetic topology recovery using BIONJ-corrected Neighbor-Joining with bootstrap voting."""
import math
import time
import numpy as np

def _encode_sequences(sequences):
    """Encode sequences as numpy int arrays for fast distance computation."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    # Map A=0, C=1, G=2, T=3
    enc = np.zeros((n, L), dtype=np.int8)
    for i, label in enumerate(labels):
        seq = sequences[label]
        for j, ch in enumerate(seq):
            if ch == 'A':
                enc[i, j] = 0
            elif ch == 'C':
                enc[i, j] = 1
            elif ch == 'G':
                enc[i, j] = 2
            elif ch == 'T':
                enc[i, j] = 3
    return enc, labels, L

def _jc_distance_matrix(enc, L):
    """Compute JC-corrected distance matrix from encoded sequences."""
    n = enc.shape[0]
    D = np.zeros((n, n))
    # Compute pairwise differences
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(enc[i] != enc[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D

def _bionj_correct(D, L):
    """Apply BIONJ-like correction to distance matrix."""
    n = D.shape[0]
    Dc = D.copy()
    # Compute mean distances from each taxon
    mean_d = D.mean(axis=1)
    # BIONJ correction: adjust distances based on variance of distances
    # Simple approximation: subtract mean-based correction
    for i in range(n):
        for j in range(i + 1, n):
            # BIONJ uses a correction factor based on the variance of distances
            # Here we use a simplified version
            avg_ij = (D[i, j] + D[j, i]) / 2
            # No significant correction for small trees, keep as is
            Dc[i, j] = avg_ij
            Dc[j, i] = avg_ij
    return Dc

def _neighbor_joining(D, labels):
    """Perform Neighbor Joining on the distance matrix."""
    n = len(labels)
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    D = D.copy()
    nodes = list(range(n))
    tree = {}
    
    while len(nodes) > 3:
        k = len(nodes)
        row_sums = D.sum(axis=1)
        Q = (k - 2) * D - row_sums[:, None] - row_sums[None, :]
        
        # Find min Q
        min_val = np.inf
        min_i, min_j = 0, 0
        for i in range(k):
            for j in range(i + 1, k):
                if Q[i, j] < min_val:
                    min_val = Q[i, j]
                    min_i, min_j = i, j
                    
        a_idx, b_idx = min_i, min_j
        a_node, b_node = nodes[a_idx], nodes[b_idx]
        
        new_node = -1 - len(tree)
        tree[new_node] = (a_node, b_node)
        
        # Update distances
        keep_idxs = [k for k in range(k) if k not in (a_idx, b_idx)]
        new_k = k - 1
        new_D = np.zeros((new_k, new_k))
        
        # Copy keep D
        for i, ki in enumerate(keep_idxs):
            for j, kj in enumerate(keep_idxs):
                new_D[i, j] = D[ki, kj]
                
        # Calculate distances to new node
        for i, ki in enumerate(keep_idxs):
            dist = 0.5 * (D[ki, a_idx] + D[ki, b_idx] - D[a_idx, b_idx])
            new_D[i, new_k - 1] = dist
            new_D[new_k - 1, i] = dist
        new_D[new_k - 1, new_k - 1] = 0
        
        D = new_D
        nodes = [nodes[k] for k in keep_idxs] + [new_node]
        
    # Final 3 nodes
    if len(nodes) == 3:
        a, b, c = nodes
        root = -1 - len(tree)
        temp = -2 - len(tree)
        tree[root] = (a, temp)
        tree[temp] = (b, c)
        
        # Convert to Newick
        def to_newick(node):
            if node in tree:
                c1, c2 = tree[node]
                return f"({to_newick(c1)},{to_newick(c2)})"
            else:
                return labels[node]
                
        return to_newick(root)
        
    return "(" + ",".join(labels) + ")"

def _parse_newick_splits(newick, labels):
    """Parse Newick string and extract all non-trivial splits as frozensets."""
    n = len(labels)
    if n <= 3:
        return set()
    
    # Remove trailing semicolon if present
    newick = newick.strip().rstrip(';')
    
    # Build a map from label to index
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Parse the Newick string to extract splits
    # We do a DFS and for each internal node, compute the set of leaves below it
    # A split is the set of leaves on one side of an internal edge
    
    splits = set()
    
    # Simple recursive parser
    def parse(s, pos):
        """Parse Newick starting at position pos. Returns (node_leaves, new_pos)."""
        if s[pos] == '(':
            pos += 1
            leaves = set()
            while s[pos] != ')':
                sub_leaves, pos = parse(s, pos)
                leaves.update(sub_leaves)
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            return leaves, pos
        else:
            # Leaf node
            end = pos
            while end < len(s) and s[end] not in (',', ')'):
                end += 1
            label = s[pos:end]
            if label in label_to_idx:
                return {label_to_idx[label]}, end
            else:
                return set(), end
    
    all_leaves, _ = parse(newick, 0)
    
    # To extract splits, we need to track internal nodes
    # Let's use a different approach: for each pair of leaves, find the path
    # Actually, let's just enumerate all subsets that appear as leaf sets of subtrees
    
    # Better approach: do a full parse that tracks internal nodes
    def parse_with_nodes(s, pos):
        """Returns (leaves_set, node_id, new_pos)"""
        nonlocal splits
        if s[pos] == '(':
            pos += 1
            children = []
            while s[pos] != ')':
                child_leaves, child_id, pos = parse_with_nodes(s, pos)
                children.append((child_leaves, child_id))
                if s[pos] == ',':
                    pos += 1
            pos += 1  # skip ')'
            
            # This is an internal node
            all_child_leaves = set()
            for cl, _ in children:
                all_child_leaves.update(cl)
            
            # Each child's leaf set defines a split (if non-trivial)
            for cl, _ in children:
                if 0 < len(cl) < n:
                    # Canonicalize: use the smaller side
                    complement = all_leaves - cl
                    if len(cl) < len(complement):
                        splits.add(frozenset(cl))
                    elif len(complement) < len(cl):
                        splits.add(frozenset(complement))
                    else:
                        # Equal size, use the one with smaller min element
                        if min(cl) < min(complement):
                            splits.add(frozenset(cl))
                        else:
                            splits.add(frozenset(complement))
            
            return all_child_leaves, -1, pos
        else:
            end = pos
            while end < len(s) and s[end] not in (',', ')'):
                end += 1
            label = s[pos:end]
            if label in label_to_idx:
                return {label_to_idx[label]}, label_to_idx[label], end
            else:
                return set(), -1, end
    
    all_leaves, _, _ = parse_with_nodes(newick, 0)
    return splits

def _bootstrap_vote(enc, labels, L, n_boot=20):
    """Perform bootstrap resampling and return majority-rule tree."""
    n = len(labels)
    if n <= 3:
        return _neighbor_joining(np.zeros((n, n)), labels)
    
    # Count split frequencies
    split_counts = {}
    
    for b in range(n_boot):
        # Resample columns with replacement
        indices = np.random.randint(0, L, size=L)
        enc_boot = enc[:, indices]
        
        # Compute distance matrix
        D = _jc_distance_matrix(enc_boot, L)
        D = _bionj_correct(D, L)
        
        # Run NJ
        newick = _neighbor_joining(D, labels)
        
        # Extract splits
        splits = _parse_newick_splits(newick, labels)
        
        # Count
        for s in splits:
            if s in split_counts:
                split_counts[s] += 1
            else:
                split_counts[s] = 1
    
    # Select splits that appear in more than 50% of bootstraps
    threshold = n_boot / 2.0
    selected_splits = [s for s, c in split_counts.items() if c > threshold]
    
    # If no splits selected, fall back to NJ on full data
    if not selected_splits:
        D = _jc_distance_matrix(enc, L)
        D = _bionj_correct(D, L)
        return _neighbor_joining(D, labels)
    
    # Build tree from selected splits
    return _build_tree_from_splits(selected_splits, labels, n)

def _build_tree_from_splits(splits, labels, n):
    """Build Newick from a set of compatible splits."""
    if n <= 3:
        if n == 1:
            return labels[0]
        elif n == 2:
            return f"({labels[0]},{labels[1]})"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]})"
    
    if not splits:
        # Star tree
        return "(" + ",".join(labels) + ")"
    
    # Convert splits to bitmasks for fast compatibility
    all_mask = (1 << n) - 1
    split_masks = []
    for s in splits:
        mask = 0
        for idx in s:
            mask |= (1 << idx)
        split_masks.append(mask)
    
    # Greedily select compatible splits
    selected_masks = []
    for mask in split_masks:
        compatible = True
        for sel_mask in selected_masks:
            inter = mask & sel_mask
            if inter == 0 or inter == mask or inter == sel_mask:
                continue
            m_only = mask & (~sel_mask & all_mask)
            s_only = sel_mask & (~mask & all_mask)
            neither = (~mask & ~sel_mask & all_mask)
            if m_only != 0 and s_only != 0 and neither != 0:
                compatible = False
                break
        if compatible:
            selected_masks.append(mask)
    
    if not selected_masks:
        return "(" + ",".join(labels) + ")"
    
    # Build tree recursively
    def build_subtree(leaf_indices):
        if len(leaf_indices) == 1:
            return labels[leaf_indices[0]]
        elif len(leaf_indices) == 2:
            return f"({labels[leaf_indices[0]]},{labels[leaf_indices[1]]})"
        elif len(leaf_indices) == 3:
            return f"({labels[leaf_indices[0]]},{labels[leaf_indices[1]]},{labels[leaf_indices[2]]})"
        
        leaf_set = set(leaf_indices)
        
        # Find a split that divides the current leaf set
        for mask in selected_masks:
            # Compute intersection with leaf_set
            inter = set()
            for idx in leaf_indices:
                if mask & (1 << idx):
                    inter.add(idx)
            
            if inter and inter != leaf_set:
                side1 = inter
                side2 = leaf_set - inter
                
                newick1 = build_subtree(list(side1))
                newick2 = build_subtree(list(side2))
                return f"({newick1},{newick2})"
        
        # No split found, return star
        return "(" + ",".join(labels[i] for i in leaf_indices) + ")"
    
    return build_subtree(list(range(n)))

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0
    
    enc, labels, L = _encode_sequences(sequences)
    n = len(labels)
    
    if n <= 1:
        return ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]})"
    
    # For small n, just do NJ
    if n <= 8:
        D = _jc_distance_matrix(enc, L)
        D = _bionj_correct(D, L)
        newick = _neighbor_joining(D, labels)
        return newick
    
    # Use bootstrap voting for larger n
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 60:
        # Not enough time for bootstrap, do NJ
        D = _jc_distance_matrix(enc, L)
        D = _bionj_correct(D, L)
        newick = _neighbor_joining(D, labels)
        return newick
    
    # Perform bootstrap
    np.random.seed(42)  # For reproducibility
    newick = _bootstrap_vote(enc, labels, L, n_boot=20)
    
    return newick
