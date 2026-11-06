import math
import numpy as np

def _jc_distance(a, b):
    diff = np.sum(a != b)
    p = diff / len(a)
    if p >= 0.75:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _variance(d, n_sites):
    if d == 0:
        return 1e-10
    p = 0.75 * (1.0 - math.exp(-4.0 * d / 3.0))
    if p < 1e-10 or p > 1.0 - 1e-10:
        return 1e-10
    var_p = p * (1.0 - p) / n_sites
    deriv = 1.0 / (1.0 - 4.0 * p / 3.0)
    return var_p * deriv * deriv

def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    n_sites = len(sequences[labels[0]])
    
    seqs = np.array([list(sequences[label]) for label in labels])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seqs = np.array([[base_map[c] for c in seq] for seq in seqs])
    
    D = {}
    V = {}
    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            d = _jc_distance(seqs[i], seqs[j])
            D[(i, j)] = D[(j, i)] = d
            V[(i, j)] = V[(j, i)] = _variance(d, n_sites)
            
    nodes = list(range(n_taxa))
    trees = {i: str(i) for i in range(n_taxa)}
    
    while len(nodes) > 3:
        n = len(nodes)
        best_q = float('inf')
        best_pair = None
        
        for idx_i, i in enumerate(nodes):
            for idx_j, j in enumerate(nodes):
                if idx_i >= idx_j: continue
                
                v_ij = V[(i, j)]
                sum_i = 0.0
                sum_j = 0.0
                for k in nodes:
                    if k == i or k == j: continue
                    v_ik = V[(i, k)]
                    v_jk = V[(j, k)]
                    sum_i += D[(i, k)] / (v_ik + v_ij)
                    sum_j += D[(j, k)] / (v_jk + v_ij)
                
                q = (n - 2) * D[(i, j)] - sum_i - sum_j
                if q < best_q:
                    best_q = q
                    best_pair = (i, j)
                    
        if best_pair is None:
            break
            
        i, j = best_pair
        new_label = f"n{len(nodes)}_{i}_{j}"
        
        trees[new_label] = f"({trees[i]},{trees[j]})"
        
        rest = [x for x in nodes if x not in (i, j)]
        for k in rest:
            D[(new_label, k)] = D[(k, new_label)] = 0.5 * (D[(i, k)] + D[(j, k)] - D[(i, j)])
            V[(new_label, k)] = V[(k, new_label)] = 0.25 * (V[(i, k)] + V[(j, k)] + V[(i, j)])
            
        nodes = rest + [new_label]
        
    return "(" + ",".join(trees[x] for x in nodes) + ");"
