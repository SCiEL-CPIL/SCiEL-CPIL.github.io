import math
import time
import numpy as np

def _jc_distance(a, b):
    diffs = np.sum(a != b)
    p = diffs / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_bionj_distances(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = [sequences[label] for label in labels]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(seqs[i], seqs[j])
            dist[i, j] = d
            dist[j, i] = d
            
    # BIONJ correction
    if n > 3:
        bionj_dist = dist.copy()
        for i in range(n):
            for j in range(i+1, n):
                s = 0.0
                for k in range(n):
                    if k != i and k != j:
                        s += dist[i, k] + dist[j, k] - dist[i, j]
                bionj_dist[i, j] = dist[i, j] + 0.5 * s / (n - 3)
                bionj_dist[j, i] = bionj_dist[i, j]
        return labels, bionj_dist, dist
    else:
        return labels, dist, dist

def _nj_tree(dist, bionj_dist=None):
    n = len(dist)
    if n <= 1:
        return None, None, None
    if n == 2:
        return [(0, 1, 0.0)], 1, 2 # Root at 1? No, unrooted.
    
    active = list(range(n))
    current_dist = dist.copy()
    tree_edges = []
    
    next_node = n
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for i, u in enumerate(active):
            for j, v in enumerate(active):
                if i != j:
                    r[i] += current_dist[u, v]
        
        best_val = float('inf')
        best_i, best_j = -1, -1
        for i in range(m):
            for j in range(i+1, m):
                u, v = active[i], active[j]
                val = (m - 2) * current_dist[u, v] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        u, v = active[best_i], active[best_j]
        
        l_u = 0.5 * current_dist[u, v] + (r[best_i] - r[best_j]) / (2.0 * (m - 2))
        l_v = current_dist[u, v] - l_u
        
        w = next_node
        next_node += 1
        
        for k in range(m):
            if k == best_i or k == best_j:
                continue
            x = active[k]
            current_dist[w, x] = 0.5 * (current_dist[u, x] + current_dist[v, x] - current_dist[u, v])
            current_dist[x, w] = current_dist[w, x]
        
        tree_edges.append((u, w, l_u))
        tree_edges.append((v, w, l_v))
        
        active.remove(u)
        active.remove(v)
        active.append(w)
        
    a, b, c = active
    center = next_node
    next_node += 1
    
    d_ab = current_dist[a, b]
    d_bc = current_dist[b, c]
    d_ca = current_dist[c, a]
    
    l_a = 0.5 * (d_ab + d_ca - d_bc)
    l_b = 0.5 * (d_ab + d_bc - d_ca)
    l_c = 0.5 * (d_ca + d_bc - d_ab)
    
    tree_edges.append((a, center, l_a))
    tree_edges.append((b, center, l_b))
    tree_edges.append((c, center, l_c))
    
    return tree_edges, center, next_node

def _build_newick(tree_edges, root, labels):
    adj = {}
    for u, v, l in tree_edges:
        if u not in adj: adj[u] = []
        if v not in adj: adj[v] = []
        adj[u].append((v, l))
        adj[v].append((u, l))
    
    def dfs(node, parent):
        children = [x for x in adj[node] if x[0] != parent]
        if not children:
            if node < len(labels):
                return labels[node]
            else:
                return "" # Should not happen for leaves
        
        child_strs = []
        for child, l in children:
            sub_str = dfs(child[0], node)
            child_strs.append(sub_str)
        
        return "(" + ",".join(child_strs) + ")"
    
    return dfs(root, -1) + ";"

def reconstruct(sequences):
    start_time = time.time()
    deadline = 230.0
    
    labels, bionj_dist, jc_dist = _compute_bionj_distances(sequences)
    n = len(labels)
    
    if n <= 3:
        if n == 1:
            return f"{labels[0]};"
        elif n == 2:
            return f"({labels[0]},{labels[1]});"
        else:
            return f"({labels[0]},{labels[1]},{labels[2]});"
    
    # Use BIONJ distances for NJ
    tree_edges, root, next_node = _nj_tree(bionj_dist)
    
    if time.time() > deadline:
        return _build_newick(tree_edges, root, labels)
    
    return _build_newick(tree_edges, root, labels)
