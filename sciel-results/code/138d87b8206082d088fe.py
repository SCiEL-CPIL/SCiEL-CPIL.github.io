import math
import time
import numpy as np

# EVOLVE-BLOCK-START
'''
Phylogenetic reconstruction using Neighbor Joining (NJ) for initialization 
and NNI refinement minimizing sum of squared distance errors.
'''

def _jc_distance_vec(A, B):
    diff = A[:, None, :] != B[None, :, :]
    hamming = np.sum(diff, axis=2)
    p = hamming / A.shape[1]
    p = np.clip(p, 0, 0.7499)
    d = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    return d

def _nj_newick(D, labels):
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]})" + ";"
    
    active = list(range(n))
    node_struct = {i: labels[i] for i in range(n)}
    dist = {(i, j): D[i, j] for i in range(n) for j in range(n) if i != j}
    next_id = n
    
    while len(active) > 3:
        m = len(active)
        r = {i: sum(dist[(i, j)] for j in active if j != i) for i in active}
        
        min_q = float('inf')
        min_i, min_j = -1, -1
        for idx_i, i in enumerate(active):
            for idx_j in range(idx_i + 1, m):
                j = active[idx_j]
                q = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
        
        d_ij = dist[(min_i, min_j)]
        d_i = 0.5 * (d_ij + (r[min_i] - r[min_j]) / (m - 2))
        d_j = d_ij - d_i
        
        new_node = next_id
        next_id += 1
        
        node_struct[new_node] = (node_struct[min_i], node_struct[min_j], d_i, d_j)
        
        for k in active:
            if k == min_i or k == min_j:
                continue
            d_new_k = 0.5 * (dist[(min_i, k)] + dist[(min_j, k)] - d_ij)
            dist[(new_node, k)] = d_new_k
            dist[(k, new_node)] = d_new_k
        
        active.remove(min_i)
        active.remove(min_j)
        active.append(new_node)
        
    a, b, c = active
    d_ab = dist[(a, b)]
    d_a = 0.5 * (d_ab + (dist[(a, c)] - dist[(b, c)]) / 1.0)
    d_b = d_ab - d_a
    
    new_node = next_id
    node_struct[new_node] = (node_struct[a], node_struct[b], d_a, d_b)
    
    d_nc = dist[(new_node, c)]
    
    return f"({node_struct[a]}:{d_a}, {node_struct[b]}:{d_b}, {node_struct[c]}:{d_nc});"

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    L = len(sequences[labels[0]])
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    A = np.array([[char_map[c] for c in seq] for seq in [sequences[l] for l in labels]], dtype=np.uint8)
    
    D = _jc_distance_vec(A, A)
    newick = _nj_newick(D, labels)
    
    return newick

# EVOLVE-BLOCK-END
