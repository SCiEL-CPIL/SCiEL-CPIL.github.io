"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def _jc_distance(a, b):
    diff = np.sum(a != b)
    p = diff / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _bionj_distance_matrix(sequences, labels):
    n = len(labels)
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    int_seqs = [np.array([char_map[c] for c in seq]) for seq in sequences]
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _jc_distance(int_seqs[i], int_seqs[j])
            dist[i, j] = d
            dist[j, i] = d
            
    # BIONJ Correction
    # D_ij = D_ij * (1 + sum_{k != i, j} (1/(N-2)) * (D_ik + D_jk - D_ij) / D_ij)
    # Note: The sum is over k.
    # This can be vectorized or done in loops.
    
    # Let's compute the correction factor for each pair
    # Sum_{k} (D_ik + D_jk - D_ij)
    # = Sum_k D_ik + Sum_k D_jk - (N-2) * D_ij
    
    # Precompute row sums
    row_sums = np.sum(dist, axis=1)
    
    bionj_dist = dist.copy()
    for i in range(n):
        for j in range(i + 1, n):
            if dist[i, j] == 0:
                continue
            # Sum over k != i, j
            # Sum_{k != i, j} D_ik = row_sums[i] - D_ii - D_ij = row_sums[i] - D_ij
            # Sum_{k != i, j} D_jk = row_sums[j] - D_ij
            # Sum_{k != i, j} (D_ik + D_jk - D_ij) = (row_sums[i] - D_ij) + (row_sums[j] - D_ij) - (N-2)*D_ij
            # = row_sums[i] + row_sums[j] - 2*D_ij - (N-2)*D_ij
            # = row_sums[i] + row_sums[j] - N*D_ij
            
            sum_term = (row_sums[i] + row_sums[j] - n * dist[i, j])
            correction = 1.0 + (sum_term / (n - 2)) / dist[i, j]
            bionj_dist[i, j] = dist[i, j] * correction
            bionj_dist[j, i] = bionj_dist[i, j]
            
    return bionj_dist

def _run_nj(dist, labels):
    n = len(labels)
    nodes = list(range(n))
    trees = [f"t{labels[i]}" for i in range(n)]
    current_dist = dist.copy()
    active = list(range(n))
    
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for i_idx, i in enumerate(active):
            r[i_idx] = sum(current_dist[i, j] for j in active if j != i)
        
        best_criterion = float('inf')
        best_i, best_j = -1, -1
        for i_idx in range(m):
            for j_idx in range(i_idx + 1, m):
                i = active[i_idx]
                j = active[j_idx]
                criterion = (m - 2) * current_dist[i, j] - r[i_idx] - r[j_idx]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_i, best_j = i, j
        
        new_node_id = max(active) + 1
        d_ij = current_dist[best_i, best_j]
        r_i = r[active.index(best_i)]
        r_j = r[active.index(best_j)]
        
        len_i = 0.5 * d_ij + (r_i - r_j) / (2 * (m - 2))
        len_j = 0.5 * d_ij - (r_i - r_j) / (2 * (m - 2))
        
        len_i = max(0.0, len_i)
        len_j = max(0.0, len_j)
        
        trees.append(f"({trees[best_i]}:{len_i:.4f},{trees[best_j]}:{len_j:.4f})")
        
        for k in active:
            if k != best_i and k != best_j:
                current_dist[new_node_id, k] = 0.5 * (current_dist[best_i, k] + current_dist[best_j, k] - d_ij)
                current_dist[k, new_node_id] = current_dist[new_node_id, k]
        
        active.remove(best_i)
        active.remove(best_j)
        active.append(new_node_id)
        
    if len(active) == 3:
        a, b, c = active
        d_ab = current_dist[a, b]
        d_ac = current_dist[a, c]
        d_bc = current_dist[b, c]
        
        len_a = max(0.0, (d_ab + d_ac - d_bc) / 2.0)
        len_b = max(0.0, (d_ab + d_bc - d_ac) / 2.0)
        len_c = max(0.0, (d_ac + d_bc - d_ab) / 2.0)
        
        final_tree = f"({trees[a]}:{len_a:.4f},{trees[b]}:{len_b:.4f},{trees[c]}:{len_c:.4f})"
        return final_tree
    else:
        return "(" + ",".join(trees[i] for i in active) + ")"

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 260.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    dist = _bionj_distance_matrix(sequences, labels)
    newick = _run_nj(dist, labels)
    
    return newick
# EVOLVE-BLOCK-END
