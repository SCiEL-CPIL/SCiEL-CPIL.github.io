"""Phylogenetic topology recovery using distance-based methods."""
import time
import math
import numpy as np

_START_TIME = time.time()
_DEADLINE = 265.0  # safety margin before 270s

def _remaining_time():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_jc_distances(sequences):
    """Compute JC-corrected pairwise distance matrix and labels."""
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seq_list = [sequences[lab] for lab in labels]
    
    code_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    codes = np.array([[code_map[c] for c in s] for s in seq_list], dtype=np.int8)
    
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            p = np.mean(codes[i] != codes[j])
            if p >= 0.7499:
                D[i, j] = D[j, i] = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
                D[i, j] = D[j, i] = d
    return D, labels

def _neighbor_joining(D, labels):
    """Neighbor-Joining algorithm. Returns Newick string."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})"
    
    dist = D.copy()
    nodes = list(range(n))
    tree = {i: labels[i] for i in range(n)}
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        # Row sums
        r = np.array([sum(dist[i][j] for j in nodes if j != i) for i in nodes])
        
        best_val = float('inf')
        best_pair = None
        for a_idx in range(m):
            for b_idx in range(a_idx + 1, m):
                a, b = nodes[a_idx], nodes[b_idx]
                q = (m - 2) * dist[a][b] - r[a_idx] - r[b_idx]
                if q < best_val:
                    best_val = q
                    best_pair = (a, b)
        
        if best_pair is None:
            break
        
        a, b = best_pair
        d_ab = dist[a][b]
        delta = (r[nodes.index(a)] - r[nodes.index(b)]) / max(m - 2, 1)
        len_a = max(0.0, 0.5 * d_ab + 0.5 * delta)
        len_b = max(0.0, 0.5 * d_ab - 0.5 * delta)
        
        new_node = next_id
        next_id += 1
        
        for x in nodes:
            if x == a or x == b:
                continue
            nd = 0.5 * (dist[a][x] + dist[b][x] - d_ab)
            dist[a, new_node] = nd
            dist[new_node, a] = nd
            dist[b, new_node] = nd
            dist[new_node, b] = nd
        
        nodes = [x for x in nodes if x != a and x != b] + [new_node]
        tree[new_node] = f"({tree[a]}:{len_a:.4f},{tree[b]}:{len_b:.4f})n{new_node}"
    
    if len(nodes) == 3:
        return f"({tree[nodes[0]]},{tree[nodes[1]]},{tree[nodes[2]]})"
    elif len(nodes) == 2:
        return f"({tree[nodes[0]]},{tree[nodes[1]]})"
    elif len(nodes) == 1:
        return tree[nodes[0]]
    else:
        return "(" + ",".join(tree[x] for x in nodes) + ")"

def reconstruct(sequences):
    if _remaining_time() < 1.0:
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return "(" + ",".join(labels) + ")"
    
    D, labels = _compute_jc_distances(sequences)
    newick = _neighbor_joining(D, labels)
    return newick
