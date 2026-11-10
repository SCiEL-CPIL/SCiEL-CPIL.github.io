"""Neighbour-joining with vectorized JC distances and a deadline guard."""
import time
import math
import numpy as np


def _jc_dist_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using numpy."""
    n = len(labels)
    n_sites = len(sequences[labels[0]])
    # Build a matrix of nucleotide codes: 0=A,1=C,2=G,3=T
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    mat = np.zeros((n, n_sites), dtype=np.int8)
    for i, lab in enumerate(labels):
        seq = sequences[lab]
        for j in range(n_sites):
            mat[i, j] = code_map[seq[j]]
    # Pairwise mismatches
    # For each pair (i,j), count positions where mat[i]!=mat[j]
    # Use broadcasting: mat[:, None, :] vs mat[None, :, :] -> (n,n,n_sites)
    # That's n^2 * n_sites which for n=32, n_sites=2000 is 32^2*2000 = 2,048,000 -> fine
    diff = (mat[:, None, :] != mat[None, :, :]).astype(np.float64)
    mismatches = diff.sum(axis=2)  # (n, n)
    p = mismatches / n_sites
    # JC correction: -3/4 ln(1 - 4p/3)
    np.fill_diagonal(p, 0.0)
    # Clip p to avoid log(0) or negative
    p_clipped = np.clip(p, 0.0, 0.7499)
    dist = -0.75 * np.log(1.0 - 4.0 * p_clipped / 3.0)
    np.fill_diagonal(dist, 0.0)
    return dist


def _nj_tree(dist, labels):
    """Standard neighbour joining. dist is (n,n) numpy array, labels is list of str."""
    n = len(labels)
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    if n == 1:
        return labels[0]

    # trees[i] = newick string for current node i
    trees = list(labels)
    # active nodes
    active = list(range(n))
    # We'll maintain a (n,n) distance matrix; when we merge, we add a new node
    # For simplicity, use a dict-based approach or extend the array.
    # Use a list of dicts? No, let's use a growing array.
    # Actually, let's use a simpler approach: maintain dist as a 2D list of lists, 
    # and a list of active node indices. When merging a,b, create new node index = len(active)+... 
    # Let's use a different strategy: keep dist as numpy array of size (n+n-3, n+n-3) but that's wasteful.
    # Better: use a list-based adjacency.
    
    # Let's use a clean implementation:
    # nodes: list of current node names (strings)
    # trees: dict name -> newick
    # dist: dict (name1, name2) -> distance
    # This is O(n^2) per iteration, total O(n^3) which is fine for n<=32.
    
    nodes = list(labels)
    trees = {name: name for name in labels}
    # Build dist dict from numpy array
    dist_dict = {}
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            d = float(dist[i, j])
            dist_dict[(nodes[i], nodes[j])] = d
            dist_dict[(nodes[j], nodes[i])] = d
    
    while len(nodes) > 3:
        n_cur = len(nodes)
        # Compute row sums
        row_sum = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if a != b:
                    s += dist_dict[(a, b)]
            row_sum[a] = s
        
        # Find best pair
        best_criterion = float('inf')
        best_a = best_b = None
        for i_idx in range(len(nodes)):
            a = nodes[i_idx]
            for j_idx in range(i_idx + 1, len(nodes)):
                b = nodes[j_idx]
                criterion = (n_cur - 2) * dist_dict[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
        
        a, b = best_a, best_b
        # Compute branch lengths
        d_ab = dist_dict[(a, b)]
        len_a = 0.5 * d_ab + (row_sum[a] - row_sum[b]) / (2.0 * (n_cur - 2))
        len_b = d_ab - len_a
        # Clamp to non-negative
        if len_a < 0: len_a = 0.0
        if len_b < 0: len_b = 0.0
        
        # Create new node
        new_name = f"n{len(nodes)}_{a}_{b}"
        trees[new_name] = f"({trees[a]},{trees[b]})"
        
        # Update distances
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            d_new_x = 0.5 * (dist_dict[(a, x)] + dist_dict[(b, x)] - d_ab)
            dist_dict[(new_name, x)] = d_new_x
            dist_dict[(x, new_name)] = d_new_x
        
        nodes = rest + [new_name]
    
    # Final three nodes
    return "(" + ",".join(trees[x] for x in nodes) + ");"


def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 170.0  # Safety margin for 20 instances in 270s wall-clock
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    # Compute distance matrix
    dist = _jc_dist_matrix(sequences, labels)
    
    # Check deadline
    if time.time() > deadline:
        # Fallback: return a star tree
        return "(" + ",".join(labels) + ");"
    
    # Run NJ
    result = _nj_tree(dist, labels)
    
    return result
