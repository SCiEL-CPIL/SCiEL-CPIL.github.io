import numpy as np
import time
import math

def reconstruct(sequences):
    start_time = time.time()
    DEADLINE = 265.0  # Safety margin from 270s wall-clock kill
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n == 0:
        return ""
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"

    # 1. Convert sequences to numpy arrays for fast distance calculation
    seq_list = [sequences[label] for label in labels]
    length = len(seq_list[0])
    
    # Map A,C,G,T to 0,1,2,3
    seq_matrix = np.zeros((n, length), dtype=np.int8)
    for i, seq in enumerate(seq_list):
        for j, char in enumerate(seq):
            if char == 'A':
                seq_matrix[i, j] = 0
            elif char == 'C':
                seq_matrix[i, j] = 1
            elif char == 'G':
                seq_matrix[i, j] = 2
            else:
                seq_matrix[i, j] = 3

    # 2. Compute Pairwise Jukes-Cantor Distances
    diff_matrix = (seq_matrix[:, np.newaxis, :] != seq_matrix[np.newaxis, :, :]).sum(axis=2)
    p_matrix = diff_matrix / length
    p_safe = np.clip(p_matrix, 0, 0.7499)
    log_arg = np.maximum(1.0 - (4.0 / 3.0) * p_safe, 1e-10)
    D = -0.75 * np.log(log_arg)
    np.fill_diagonal(D, 0.0)

    # 3. Neighbor Joining
    active_nodes = list(range(n))
    node_trees = {i: labels[i] for i in range(n)}
    D_active = D.copy()
    next_node_id = n

    while len(active_nodes) > 2:
        if time.time() > DEADLINE - 5:
            break
            
        m = len(active_nodes)
        r = D_active.sum(axis=1)
        Q = (m - 2) * D_active - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        min_val, min_idx_flat = np.unravel_index(np.argmin(Q), Q.shape)
        i, j = min_idx_flat
        if i > j:
            i, j = j, i
            
        a_id = active_nodes[i]
        b_id = active_nodes[j]
        d_ab = D_active[i, j]
        
        c_id = next_node_id
        next_node_id += 1
        node_trees[c_id] = f"({node_trees[a_id]},{node_trees[b_id]})"
        
        new_active_nodes = []
        for k in range(m):
            if k == i or k == j:
                continue
            new_active_nodes.append(active_nodes[k])
        new_active_nodes.append(c_id)
        
        remaining_indices = [k for k in range(m) if k != i and k != j]
        d_c_x = 0.5 * (D_active[i, remaining_indices] + D_active[j, remaining_indices] - d_ab)
        D_sub = D_active[np.ix_(remaining_indices, remaining_indices)]
        
        new_D = np.zeros((m - 1, m - 1))
        new_D[:m-2, :m-2] = D_sub
        new_D[:m-2, m-1] = d_c_x
        new_D[m-1, :m-2] = d_c_x
        new_D[m-1, m-1] = 0.0
        
        D_active = new_D
        active_nodes = new_active_nodes

    # Final assembly
    if len(active_nodes) == 3:
        a_id, b_id, c_id = active_nodes
        new_id_1 = next_node_id
        next_node_id += 1
        node_trees[new_id_1] = f"({node_trees[a_id]},{node_trees[b_id]})"
        new_id_2 = next_node_id
        next_node_id += 1
        node_trees[new_id_2] = f"({node_trees[new_id_1]},{node_trees[c_id]})"
        final_tree = node_trees[new_id_2]
    elif len(active_nodes) == 2:
        a_id, b_id = active_nodes
        final_tree = f"({node_trees[a_id]},{node_trees[b_id]})"
    elif len(active_nodes) == 1:
        final_tree = node_trees[active_nodes[0]]
    else:
        final_tree = ",".join(node_trees[a] for a in active_nodes)
        
    return final_tree + ";"
