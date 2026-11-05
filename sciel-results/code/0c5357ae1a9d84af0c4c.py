import math
import time
import random
from itertools import combinations
import numpy as np

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    # Handle trivial cases
    if n <= 3:
        if n == 3:
            return f"({labels[0]},{labels[1]},{labels[2]});"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]});"

    # 1. Compute Pairwise Distances (JC corrected) using NumPy
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arrays = []
    for label in labels:
        seq = sequences[label]
        arr = np.array([char_map[c] for c in seq], dtype=np.int8)
        seq_arrays.append(arr)
    
    seq_matrix = np.vstack(seq_arrays)
    
    D = np.zeros((n, n), dtype=np.float64)
    L = len(sequences[labels[0]])
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.sum(seq_matrix[i] != seq_matrix[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i][j] = d
            D[j][i] = d

    # 2. Neighbor-Joining Tree Construction
    # Max nodes = 2*n
    max_nodes = 2 * n
    D_curr = np.zeros((max_nodes, max_nodes), dtype=np.float64)
    D_curr[:n, :n] = D
    
    active = list(range(n))
    next_id = n
    
    adj = {i: {} for i in range(max_nodes)}
    
    while len(active) > 2:
        m = len(active)
        if m <= 2:
            break
            
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j:
                    r[i] += D_curr[active[i]][active[j]]
                    
        min_q = float('inf')
        min_i = -1
        min_j = -1
        
        for i in range(m):
            for j in range(i + 1, m):
                q = (m - 2) * D_curr[active[i]][active[j]] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    min_i = i
                    min_j = j
                    
        if min_i == -1:
            break
            
        node_i = active[min_i]
        node_j = active[min_j]
        
        delta_i = r[min_i] / (m - 2)
        delta_j = r[min_j] / (m - 2)
        
        d_ij = D_curr[node_i][node_j]
        
        len_i = 0.5 * d_ij + delta_i - delta_j
        len_j = 0.5 * d_ij - delta_i + delta_j
        
        if len_i < 0: len_i = 0
        if len_j < 0: len_j = 0
        
        new_node = next_id
        next_id += 1
        
        adj[new_node] = {}
        adj[new_node][node_i] = len_i
        adj[new_node][node_j] = len_j
        
        adj[node_i][new_node] = len_i
        adj[node_j][new_node] = len_j
        
        for k in range(m):
            if k == min_i or k == min_j:
                continue
            node_k = active[k]
            d_ik = D_curr[node_i][node_k]
            d_jk = D_curr[node_j][node_k]
            d_new_k = 0.5 * (d_ik + d_jk - d_ij)
            if d_new_k < 0: d_new_k = 0
            
            D_curr[new_node][node_k] = d_new_k
            D_curr[node_k][new_node] = d_new_k
            
        active.remove(node_i)
        active.remove(node_j)
        active.append(new_node)
        
        if time.time() - start_time > 250:
            break

    if len(active) == 2:
        node_i = active[0]
        node_j = active[1]
        d_ij = D_curr[node_i][node_j]
        if d_ij < 0: d_ij = 0
        adj[node_i][node_j] = d_ij
        adj[node_j][node_i] = d_ij

    def newick_from_adj(node, parent):
        neighbors = [nb for nb in adj[node] if nb != parent]
        if not neighbors:
            return labels[node]
        
        children_strs = []
        for nb in neighbors:
            child_str = newick_from_adj(nb, node)
            children_strs.append(child_str)
            
        return "(" + ",".join(children_strs) + ")"
        
    newick = newick_from_adj(0, -1)
    
    return newick + ";"
