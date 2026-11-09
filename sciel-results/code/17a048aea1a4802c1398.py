import time
import numpy as np

def _jc_distance_matrix(sequences):
    labels = sorted(sequences.keys())
    n = len(labels)
    aln = np.array([list(sequences[lab]) for lab in labels])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    aln_int = np.vectorize(base_map.get)(aln).astype(np.int8)
    n_sites = aln_int.shape[1]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            diff = (aln_int[i] != aln_int[j]).sum()
            p = diff / n_sites
            if p >= 0.75:
                D[i,j] = 1.0
            else:
                D[i,j] = -0.75 * np.log(1 - 4/3 * p)
            D[j,i] = D[i,j]
    return D, labels

def _nj_tree(D, labels):
    n = len(labels)
    if n == 1:
        return {
            'root': 0,
            'children': {0: []},
            'parent': {0: -1},
            'leaf_map': {0: labels[0]},
            'lengths': {}
        }
    if n == 2:
        return {
            'root': 2,
            'children': {2: [0, 1], 0: [], 1: []},
            'parent': {0: 2, 1: 2, 2: -1},
            'leaf_map': {0: labels[0], 1: labels[1]},
            'lengths': {(2, 0): D[0,1]/2, (2, 1): D[0,1]/2}
        }

    active = list(range(n))
    D_cur = D.copy()
    next_id = n
    children = {i: [] for i in range(n)}
    parent = {i: -1 for i in range(n)}
    leaf_map = {i: labels[i] for i in range(n)}
    lengths = {}
    
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for i in range(m):
            for j in range(m):
                r[i] += D_cur[active[i], active[j]]
        
        min_q = float('inf')
        i_idx, j_idx = -1, -1
        for i in range(m):
            for j in range(i+1, m):
                q = (m-2) * D_cur[active[i], active[j]] - r[i] - r[j]
                if q < min_q:
                    min_q = q
                    i_idx, j_idx = i, j
        
        i_node = active[i_idx]
        j_node = active[j_idx]
        d_ij = D_cur[i_node, j_node]
        
        delta = 0.5 * (r[i_idx] - r[j_idx])
        len_i = 0.5 * d_ij + delta
        len_j = 0.5 * d_ij - delta
        
        new_node = next_id
        next_id += 1
        
        children[new_node] = [i_node, j_node]
        parent[i_node] = new_node
        parent[j_node] = new_node
        parent[new_node] = -1
        lengths[(new_node, i_node)] = len_i
        lengths[(new_node, j_node)] = len_j
        
        for k in active:
            if k == i_node or k == j_node:
                continue
            D_cur[new_node, k] = 0.5 * (D_cur[i_node, k] + D_cur[j_node, k] - d_ij)
            D_cur[k, new_node] = D_cur[new_node, k]
            
        active.remove(i_node)
        active.remove(j_node)
        active.append(new_node)
        
    # Final 3 nodes
    a, b, c = active
    dab = D_cur[a,b]
    dac = D_cur[a,c]
    dbc = D_cur[b,c]
    
    # Pick closest pair
    if dab <= dac and dab <= dbc:
        p1, p2, other = a, b, c
    elif dac <= dab and dac <= dbc:
        p1, p2, other = a, c, b
    else:
        p1, p2, other = b, c, a
        
    mid = next_id
    next_id += 1
    root = next_id
    next_id += 1
    
    d12 = D_cur[p1, p2]
    d13 = D_cur[p1, other]
    d23 = D_cur[p2, other]
    
    L_mp1 = 0.5 * (d12 + d13 - d23)
    L_mp2 = 0.5 * (d12 + d23 - d13)
    L_rm_plus_L_ro = 0.5 * (d13 + d23 - d12)
    L_rm = 0.5 * L_rm_plus_L_ro
    L_ro = 0.5 * L_rm_plus_L_ro
    
    children[mid] = [p1, p2]
    parent[p1] = mid
    parent[p2] = mid
    lengths[(mid, p1)] = L_mp1
    lengths[(mid, p2)] = L_mp2
    
    children[root] = [mid, other]
    parent[mid] = root
    parent[other] = root
    lengths[(root, mid)] = L_rm
    lengths[(root, other)] = L_ro
    parent[root] = -1
    
    return {
        'root': root,
        'children': children,
        'parent': parent,
        'leaf_map': leaf_map,
        'lengths': lengths
    }

def reconstruct(sequences):
    D, labels = _jc_distance_matrix(sequences)
    n = len(labels)
    if n <= 1:
        return labels[0] if n == 1 else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    
    tree = _nj_tree(D, labels)
    
    def to_newick(node):
        if node in tree['leaf_map']:
            return tree['leaf_map'][node]
        children = tree['children'][node]
        if not children:
            return ""
        child_strs = []
        for child in children:
            c_str = to_newick(child)
            l = tree['lengths'].get((node, child), 0.0)
            if l > 0.001:
                c_str += f":{l:.4f}"
            child_strs.append(c_str)
        return "(" + ",".join(child_strs) + ")"
        
    newick = to_newick(tree['root'])
    return newick + ";"
