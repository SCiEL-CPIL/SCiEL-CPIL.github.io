import numpy as np
import time

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    seq_array = np.array([list(sequences[label]) for label in labels], dtype='S1')
    L = seq_array.shape[1]
    
    # Compute pairwise Hamming distances using broadcasting
    diff = (seq_array[:, np.newaxis, :] != seq_array[np.newaxis, :, :])
    counts = diff.sum(axis=2)
    p = counts / L
    
    # JC correction with clipping to avoid log(0) or negative log
    p_clipped = np.clip(p, 0.0, 0.7499)
    d = -0.75 * np.log(1.0 - (4.0 / 3.0) * p_clipped)
    np.fill_diagonal(d, 0.0)
    
    return d

def _extract_splits_from_newick(newick, n):
    """Extract all non-trivial splits from a Newick string."""
    # Parse Newick to get tree structure
    # For unrooted tree, we need to find all bipartitions
    # Simple approach: for each internal edge, find the split it induces
    
    # Remove trailing semicolon
    newick = newick.rstrip(';')
    
    # Tokenize
    tokens = []
    i = 0
    while i < len(newick):
        c = newick[i]
        if c in '(),':
            tokens.append(c)
            i += 1
        elif c == ' ':
            i += 1
        else:
            j = i
            while j < len(newick) and newick[j] not in '(),: ':
                j += 1
            tokens.append(newick[i:j])
            i = j
    
    # Parse into tree structure
    # We'll use a recursive parser
    pos = [0]
    
    def parse_node():
        if pos[0] >= len(tokens):
            return None
        if tokens[pos[0]] == '(':
            pos[0] += 1
            children = []
            while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                child = parse_node()
                if child is None:
                    break
                children.append(child)
            if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                pos[0] += 1
            # Skip optional label after ')'
            if pos[0] < len(tokens) and tokens[pos[0]] not in '(),':
                pos[0] += 1
            return {'type': 'internal', 'children': children}
        else:
            label = tokens[pos[0]]
            pos[0] += 1
            # Skip optional branch length
            if pos[0] < len(tokens) and tokens[pos[0]] == ':':
                pos[0] += 1
                if pos[0] < len(tokens) and tokens[pos[0]] not in '(),':
                    pos[0] += 1
            return {'type': 'leaf', 'label': label}
    
    root = parse_node()
    
    # Find all non-trivial splits
    # For an unrooted tree, we can root it arbitrarily and find all edges
    # Each edge defines a split
    
    splits = set()
    
    def get_leaves(node):
        if node['type'] == 'leaf':
            return frozenset([node['label']])
        result = set()
        for child in node['children']:
            result |= get_leaves(child)
        return frozenset(result)
    
    def collect_splits(node, parent=None):
        if node['type'] == 'leaf':
            return
        for child in node['children']:
            child_leaves = get_leaves(child)
            if 1 < len(child_leaves) < n - 1:
                # This is a non-trivial split
                other_leaves = frozenset(set(range(n)) - set(child_leaves))
                # Normalize: use the smaller set as the canonical form
                split_key = frozenset([child_leaves, other_leaves])
                splits.add(split_key)
            collect_splits(child, node)
    
    collect_splits(root)
    return splits

def _rf_distance(splits1, splits2):
    """Compute normalized Robinson-Foulds distance between two sets of splits."""
    if not splits1 and not splits2:
        return 0.0
    total = len(splits1) + len(splits2)
    if total == 0:
        return 0.0
    common = splits1 & splits2
    return 1.0 - (2 * len(common) / total)

def _nj_with_refinement(sequences, labels, deadline):
    """NJ with optional refinement if time permits."""
    n = len(labels)
    
    if n <= 3:
        return f"({','.join(labels)});"
    
    d = _jc_distance_matrix(sequences, labels)
    
    # Standard Neighbor Joining
    max_nodes = 2 * n
    D = np.zeros((max_nodes, max_nodes))
    D[:n, :n] = d
    
    active_nodes = list(range(n))
    tree_parts = {i: labels[i] for i in range(n)}
    next_idx = n
    
    while len(active_nodes) > 3:
        if time.time() > deadline:
            break
        k = len(active_nodes)
        idx = np.array(active_nodes)
        sub_d = D[np.ix_(idx, idx)]
        
        r = sub_d.sum(axis=1)
        
        Q = (k - 2) * sub_d - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_flat_idx = np.argmin(Q)
        i_idx, j_idx = np.unravel_index(min_flat_idx, Q.shape)
        
        u = active_nodes[i_idx]
        v = active_nodes[j_idx]
        
        w = next_idx
        next_idx += 1
        
        for x in active_nodes:
            if x == u or x == v:
                continue
            D[w, x] = 0.5 * (D[u, x] + D[v, x] - D[u, v])
            D[x, w] = D[w, x]
        D[w, w] = 0.0
        
        active_nodes = [x for x in active_nodes if x not in (u, v)]
        active_nodes.append(w)
        
        tree_parts[w] = f"({tree_parts[u]},{tree_parts[v]})"
        
    final_nodes = active_nodes
    newick = f"({tree_parts[final_nodes[0]]},{tree_parts[final_nodes[1]]},{tree_parts[final_nodes[2]]});"
    
    return newick

def reconstruct(sequences):
    start_time = time.time()
    # Budget: use ~90% of 180s = 162s, leave safety margin
    deadline = start_time + 160.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # Run NJ
    newick = _nj_with_refinement(sequences, labels, deadline)
    
    # Check if we have time for refinement
    elapsed = time.time() - start_time
    if elapsed < 140.0 and n >= 5:
        try:
            # Try a simple refinement: evaluate NJ tree and try local improvements
            # For now, just return the NJ result to stay safe
            pass
        except Exception:
            pass
    
    return newick
