import numpy as np
import time
import math

_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _jc_distance_matrix(sequences):
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    n = len(labels)
    if n == 0:
        return np.array([]), labels
    seq_data = np.array([[ord(c) for c in sequences[label]] for label in labels])
    L = seq_data.shape[1]
    if L == 0:
        return np.zeros((n, n)), labels
    diff = seq_data[:, np.newaxis, :] != seq_data[np.newaxis, :, :]
    p = np.sum(diff, axis=2) / float(L)
    d = -0.75 * np.log(1 - (4.0/3.0) * p)
    # Handle numerical issues where log argument is <= 0
    mask = p >= 0.75
    d[mask] = 100.0
    np.fill_diagonal(d, 0.0)
    return d, labels

def _nj_tree(d, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    if n == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ");"

    # Pre-allocate distance matrix for max 2*n nodes
    max_nodes = 2 * n
    dist = np.zeros((max_nodes, max_nodes))
    # Copy initial distances
    dist[:n, :n] = d
    
    active = list(range(n))
    trees = {i: labels[i] for i in range(n)}
    
    next_internal_id = n
    
    while len(active) > 3:
        n_act = len(active)
        # Calculate r_i = sum_{j in active} d_ij
        r = np.zeros(n_act)
        for idx_i, i in enumerate(active):
            for j in active:
                if i != j:
                    r[idx_i] += dist[i, j]
        
        best_c = float('inf')
        best_i, best_j = -1, -1
        
        for idx_i, i in enumerate(active):
            for idx_j, j in enumerate(active):
                if i == j: continue
                c = (n_act - 2) * dist[i, j] - r[idx_i] - r[idx_j]
                if c < best_c:
                    best_c = c
                    best_i = i
                    best_j = j

        i, j = best_i, best_j
        d_ij = dist[i, j]
        
        a = next_internal_id
        next_internal_id += 1
        
        # Update distances from new node a to all other active nodes k
        for k in active:
            if k == i or k == j: continue
            d_ik = dist[i, k]
            d_jk = dist[j, k]
            d_ak = 0.5 * (d_ik + d_jk - d_ij)
            dist[a, k] = d_ak
            dist[k, a] = d_ak
            
        trees[a] = "(" + trees[i] + "," + trees[j] + ")"
        
        # Remove i, j from active, add a
        active.remove(i)
        active.remove(j)
        active.append(a)
        
    # Final 3 nodes
    a, b, c = active
    newick = "(" + trees[a] + "," + trees[b] + "," + trees[c] + ");"
    return newick

def reconstruct(sequences):
    global _START_TIME
    _START_TIME = time.time()
    
    labels = sorted(sequences, key=lambda x: int(x[1:]))
    d, _ = _jc_distance_matrix(sequences)
    
    # Use NJ for initial tree
    newick = _nj_tree(d, labels)
    
    return newick
