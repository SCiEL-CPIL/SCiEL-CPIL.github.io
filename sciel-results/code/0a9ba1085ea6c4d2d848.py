"""Phylogenetic topology recovery via Maximum Likelihood under Jukes-Cantor."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_MAX_PER_INSTANCE = 12.0

def _jc_distance(p):
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _compute_ml(nodes, root, n_taxa, seq_matrix):
    """
    Compute log-likelihood under JC model using Felsenstein's algorithm.
    nodes: list where nodes[i] for i >= n_taxa is (left, right)
    root: index of root
    seq_matrix: (n_sites, n_taxa) array with values 0-3
    """
    n_sites = seq_matrix.shape[0]
    n_nodes = len(nodes)
    
    # Estimate uniform branch length
    n = n_taxa
    total_dist = 0.0
    count = 0
    for i in range(n):
        for j in range(i+1, n):
            p = np.mean(seq_matrix[:, i] != seq_matrix[:, j])
            total_dist += p
            count += 1
    avg_pdist = total_dist / count if count > 0 else 0.1
    edge_len = avg_pdist / max(1.0, math.log2(n)) if avg_pdist > 0 else 0.1
    edge_len = min(max(edge_len, 0.01), 1.0)
    
    exp_term = math.exp(-4.0 * edge_len / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 - 0.25 * exp_term
    
    L = np.zeros((n_nodes, n_sites, 4), dtype=np.float64)
    
    for i in range(n_taxa):
        obs = seq_matrix[:, i]
        for a in range(4):
            L[i, obs == a, a] = 1.0
    
    # Post-order traversal
    order = []
    def dfs(v):
        if v < n_taxa:
            order.append(v)
        else:
            left, right = nodes[v]
            dfs(left)
            dfs(right)
            order.append(v)
    dfs(root)
    
    for v in order:
        if v < n_taxa:
            continue
        left, right = nodes[v]
        left_sum = L[left].sum(axis=1)
        right_sum = L[right].sum(axis=1)
        for a in range(4):
            lc = p_diff * left_sum + (p_same - p_diff) * L[left, :, a]
            rc = p_diff * right_sum + (p_same - p_diff) * L[right, :, a]
            L[v, :, a] = lc * rc
    
    root_probs = L[root].sum(axis=1)
    root_probs = np.maximum(root_probs, 1e-300)
    return float(np.sum(np.log(root_probs)))

def _nj_tree(sequences, labels):
    n = len(labels)
    enc = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.array([[enc[c] for c in sequences[label]] for label in labels]).T
    
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            p = np.mean(seq_matrix[:, i] != seq_matrix[:, j])
            d = _jc_distance(p)
            D[i, j] = d
            D[j, i] = d
    
    nodes = list(range(n))
    trees = {i: i for i in range(n)}
    next_id = n
    
    while len(nodes) > 3:
        m = len(nodes)
        total = {a: sum(D[a, b] for b in nodes if b != a) for a in nodes}
        best = None
        for idx, a in enumerate(nodes):
            for b in nodes[idx+1:]:
                crit = (m - 2) * D[a, b] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        new_id = next_id
        next_id += 1
        trees[new_id] = (a, b)
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            D[new_id, x] = 0.5 * (D[a, x] + D[b, x] - D[a, b])
            D[x, new_id] = D[new_id, x]
        nodes = rest + [new_id]
    
    a, b, c = nodes
    root = next_id
    next_id += 1
    trees[root] = (a, (b, c))
    
    # Build node list
    node_list = [None] * next_id
    for i in range(n):
        node_list[i] = i
    for nid in range(n, next_id):
        node_list[nid] = trees[nid]
    
    return node_list, root, next_id, seq_matrix

def _nodes_to_newick(nodes, root, n_taxa, labels):
    def rec(v):
        if v < n_taxa:
            return labels[v]
        left, right = nodes[v]
        return f"({rec(left)},{rec(right)})"
    return rec(root) + ";"

def _try_nni(nodes, root, n_taxa, seq_matrix, current_ll):
    """
    Try all NNI moves. Return (new_nodes, new_root, new_ll) if improved, else None.
    """
    n_nodes = len(nodes)
    best_move = None
    best_ll = current_ll
    
    for i in range(n_taxa, n_nodes):
        if i == root:
            continue
        left, right = nodes[i]
        
        # NNI move 1: (left, right) -> (left's left, (left's right, right))
        if left >= n_taxa:
            ll, lr = nodes[left]
            # New structure: (ll, (lr, right))
            new_nodes = list(nodes)
            new_nodes[i] = (ll, (lr, right))
            # Need to check if this is valid (no cycles)
            ll_new = _compute_ml(new_nodes, root, n_taxa, seq_matrix)
            if ll_new > best_ll:
                best_ll = ll_new
                best_move = (new_nodes, root)
        
        # NNI move 2: (left, right) -> ((left, right's left), right's right)
        if right >= n_taxa:
            rl, rr = nodes[right]
            new_nodes = list(nodes)
            new_nodes[i] = ((left, rl), rr)
            ll_new = _compute_ml(new_nodes, root, n_taxa, seq_matrix)
            if ll_new > best_ll:
                best_ll = ll_new
                best_move = (new_nodes, root)
    
    return best_move, best_ll

def reconstruct(sequences):
    start_time = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 2:
            return f"({labels[0]},{labels[1]});"
        return f"({labels[0]});" if n == 1 else ";"
    
    node_list, root, n_nodes, seq_matrix = _nj_tree(sequences, labels)
    
    best_nodes = list(node_list)
    best_root = root
    best_ll = _compute_ml(best_nodes, best_root, n, seq_matrix)
    
    # Hill-climb with NNI
    time_budget = _MAX_PER_INSTANCE - (time.time() - start_time)
    max_iterations = int(time_budget / 0.5)  # ~0.5s per full NNI pass
    
    for iteration in range(max_iterations):
        if time.time() - start_time > _MAX_PER_INSTANCE - 1.0:
            break
        move, new_ll = _try_nni(best_nodes, best_root, n, seq_matrix, best_ll)
        if move is None:
            break
        best_nodes, best_root = move
        best_ll = new_ll
    
    newick = _nodes_to_newick(best_nodes, best_root, n, labels)
    return newick
# EVOLVE-BLOCK-END
