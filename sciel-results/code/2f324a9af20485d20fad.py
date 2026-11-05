import numpy as np
import math

def _jc_distance_matrix(seqs_np):
    # seqs_np: (n, L) array of ints 0-3
    n, L = seqs_np.shape
    # diff[i,j] = number of mismatches
    # Using broadcasting
    # seqs_np[:, None, :] is (n, 1, L)
    # seqs_np[None, :, :] is (1, n, L)
    diff = (seqs_np[:, None, :] != seqs_np[None, :, :]).sum(axis=2)
    p = diff / L
    # Clip p to avoid log(0) or log(negative)
    # Max p is 1.0. 1 - 4/3 * 1 = -1/3. log is undefined.
    # JC is valid for p < 0.75.
    # If p >= 0.75, distance is infinite/undefined. Cap it.
    p_clipped = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    # Set diagonal to 0
    np.fill_diagonal(d, 0.0)
    return d

def _nj_build(dist_matrix, labels):
    # dist_matrix: (n, n) numpy array
    # labels: list of strings
    n = len(labels)
    if n == 1:
        return labels[0]
    if n == 2:
        return (labels[0], labels[1])
    
    # Use a dict for active nodes to avoid resizing
    # Map internal index to label
    # dist_matrix is fixed size, but we only care about active nodes
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    # We need to update distances.
    # It is easier to maintain a dictionary of distances for active nodes.
    d = {(i, j): dist_matrix[i, j] for i in range(n) for j in range(n) if i != j}
    
    while len(active) > 3:
        n_curr = len(active)
        r = {u: 0.0 for u in active}
        for u in active:
            for v in active:
                if u != v:
                    r[u] += d[(u, v)]
        
        best_val = float('inf')
        best_u = None
        best_v = None
        for i, u in enumerate(active):
            for v in active[i+1:]:
                val = (n_curr - 2) * d[(u, v)] - r[u] - r[v]
                if val < best_val:
                    best_val = val
                    best_u = u
                    best_v = v
        
        u, v = best_u, best_v
        new_id = f"n{len(active)}_{u}_{v}"
        trees[new_id] = (trees[u], trees[v])
        
        rest = [x for x in active if x not in (u, v)]
        for x in rest:
            new_d = 0.5 * (d[(u, x)] + d[(v, x)] - d[(u, v)])
            d[(new_id, x)] = new_d
            d[(x, new_id)] = new_d
            
        active = rest + [new_id]
        
    a, b, c = active
    return (trees[a], trees[b], trees[c])

def _get_leaves(node):
    if isinstance(node, str):
        return {node}
    leaves = set()
    for child in node:
        leaves.update(_get_leaves(child))
    return leaves

def _extract_splits(tree, taxa_set):
    splits = set()
    
    def process_rec(node):
        if isinstance(node, str):
            return
        child_leaves = [_get_leaves(c) for c in node]
        for cl in child_leaves:
            if len(cl) < len(taxa_set) / 2:
                split_key = frozenset(cl)
            else:
                split_key = frozenset(taxa_set - cl)
            splits.add(split_key)
        for c in node:
            process_rec(c)

    process_rec(tree)
    return splits

def _build_consensus(splits, taxa):
    if not splits:
        return tuple(taxa)
    
    sorted_splits = sorted(splits.items(), key=lambda x: x[1], reverse=True)
    
    def build_rec(current_taxa):
        if len(current_taxa) <= 2:
            return tuple(current_taxa)
        
        valid_split = None
        for s, count in sorted_splits:
            s_set = set(s)
            if s_set.issubset(current_taxa):
                comp = current_taxa - s_set
                if comp:
                    valid_split = s_set
                    break
        
        if valid_split is None:
            return tuple(current_taxa)
        
        other_side = current_taxa - valid_split
        left_tree = build_rec(valid_split)
        right_tree = build_rec(other_side)
        
        return (left_tree, right_tree)
        
    return build_rec(set(taxa))

def _to_newick(tree):
    if isinstance(tree, str):
        return tree
    if len(tree) == 2:
        return f"({_to_newick(tree[0])},{_to_newick(tree[1])})"
    else:
        return "(" + ",".join(_to_newick(t) for t in tree) + ")"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    seqs_list = [sequences[l] for l in labels]
    
    # Convert to numpy array
    # A=0, C=1, G=2, T=3
    code = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    arr = np.array([[code[c] for c in s] for s in seqs_list], dtype=np.uint8)
    n, L = arr.shape
    
    # Adaptive K
    if L < 500:
        K = 50
    else:
        K = 20
        
    split_counts = {}
    
    for _ in range(K):
        # Bootstrap
        idx = np.random.randint(0, L, size=L)
        boot_arr = arr[:, idx]
        
        d = _jc_distance_matrix(boot_arr)
        tree = _nj_build(d, labels)
        splits = _extract_splits(tree, set(labels))
        
        for s in splits:
            split_counts[s] = split_counts.get(s, 0) + 1
            
    # Consensus
    # Filter splits with support > 0.5
    threshold = K * 0.5
    filtered_splits = {s: c for s, c in split_counts.items() if c > threshold}
    
    consensus_tree = _build_consensus(filtered_splits, labels)
    newick = _to_newick(consensus_tree)
    
    return newick
