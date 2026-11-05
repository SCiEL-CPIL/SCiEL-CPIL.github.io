import numpy as np
import math
import time

START_TIME = time.time()
TIME_LIMIT = 170.0

def _jc_distance(a, b):
    differences = sum(1 for x, y in zip(a, b) if x != y)
    p = differences / len(a)
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _neighbor_joining(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    tree = {name: set() for name in labels}
    branch_lengths = {}
    
    D = {}
    for i, a in enumerate(labels):
        for b in labels[i+1:]:
            d = _jc_distance(sequences[a], sequences[b])
            D[(a, b)] = d
            D[(b, a)] = d
            
    nodes = list(labels)
    counter = 0
    
    while len(nodes) > 3:
        m = len(nodes)
        r = {i: sum(D[(i, j)] for j in nodes if j != i) for i in nodes}
        
        best_criterion = None
        best_pair = None
        for i_idx, a in enumerate(nodes):
            for b in nodes[i_idx+1:]:
                criterion = (m - 2) * D[(a, b)] - r[a] - r[b]
                if best_criterion is None or criterion < best_criterion:
                    best_criterion = criterion
                    best_pair = (a, b)
                    
        a, b = best_pair
        
        L_ab = 0.5 * D[(a, b)]
        L_a = 0.5 * (D[(a, b)] + (r[a] - r[b]) / (m - 2))
        L_b = 0.5 * (D[(a, b)] - (r[a] - r[b]) / (m - 2))
        
        if L_a < 0: L_a = 0
        if L_b < 0: L_b = 0
        
        new_node = f"n{counter}"
        counter += 1
        
        tree[a].add(new_node)
        tree[b].add(new_node)
        tree[new_node] = {a, b}
        
        branch_lengths[(a, new_node)] = L_a
        branch_lengths[(new_node, a)] = L_a
        branch_lengths[(b, new_node)] = L_b
        branch_lengths[(new_node, b)] = L_b
        
        for x in nodes:
            if x not in (a, b):
                D[(new_node, x)] = 0.5 * (D[(a, x)] + D[(b, x)] - D[(a, b)])
                D[(x, new_node)] = D[(new_node, x)]
                
        nodes.remove(a)
        nodes.remove(b)
        nodes.append(new_node)
        
    if len(nodes) == 3:
        a, b, c = nodes
        tree[a].add(b)
        tree[b].add(a)
        tree[b].add(c)
        tree[c].add(b)
        tree[c].add(a)
        tree[a].add(c)
        
        bl = 0.1
        branch_lengths[(a, b)] = bl
        branch_lengths[(b, a)] = bl
        branch_lengths[(b, c)] = bl
        branch_lengths[(c, b)] = bl
        branch_lengths[(a, c)] = bl
        branch_lengths[(c, a)] = bl
        
    return tree, branch_lengths

def _jc_transition_matrix(t):
    if t < 0: t = 0
    exp_term = math.exp(-4.0 * t / 3.0)
    p_same = 0.25 + 0.75 * exp_term
    p_diff = 0.25 * (1.0 - exp_term)
    return np.array([
        [p_same, p_diff, p_diff, p_diff],
        [p_diff, p_same, p_diff, p_diff],
        [p_diff, p_diff, p_same, p_diff],
        [p_diff, p_diff, p_diff, p_same]
    ])

def _calculate_likelihood(tree, sequences, branch_lengths):
    leaves = [n for n in tree if len(tree[n]) == 1]
    if not leaves:
        return -np.inf
    root = leaves[0]
    
    parent = {root: None}
    children = {root: []}
    stack = [root]
    visited = {root}
    
    while stack:
        node = stack.pop()
        for neighbor in tree[node]:
            if neighbor not in visited:
                parent[neighbor] = node
                children[node].append(neighbor)
                visited.add(neighbor)
                stack.append(neighbor)
                
    S = len(sequences[leaves[0]])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_matrix = np.zeros((S, len(leaves)), dtype=int)
    for i, leaf in enumerate(leaves):
        for j, char in enumerate(sequences[leaf]):
            seq_matrix[j, i] = base_map[char]
            
    likelihoods = {}
    
    def compute_likelihood(node):
        if node in children and not children[node]:
            idx = leaves.index(node)
            L = np.eye(4)[seq_matrix[:, idx]]
            likelihoods[node] = L
            return L
            
        if parent[node] is not None:
            bl = branch_lengths.get((node, parent[node]), branch_lengths.get((parent[node], node), 0.0))
        else:
            bl = 0.0
            
        P = _jc_transition_matrix(bl)
        
        c1, c2 = children[node]
        L1 = compute_likelihood(c1)
        L2 = compute_likelihood(c2)
        
        bl1 = branch_lengths.get((node, c1), branch_lengths.get((c1, node), 0.0))
        bl2 = branch_lengths.get((node, c2), branch_lengths.get((c2, node), 0.0))
        
        P1 = _jc_transition_matrix(bl1)
        P2 = _jc_transition_matrix(bl2)
        
        L_node = np.zeros((S, 4))
        for s in range(4):
            for s1 in range(4):
                for s2 in range(4):
                    L_node[:, s] += P1[s1, s] * P2[s2, s] * L1[:, s1] * L2[:, s2]
                    
        likelihoods[node] = L_node
        return L_node
        
    compute_likelihood(root)
    L_root = likelihoods[root]
    ll = np.sum(L_root) * 0.25
    return np.log(ll) if ll > 0 else -np.inf

def _nni_search(tree, sequences, branch_lengths):
    internal_edges = []
    for u in tree:
        for v in tree[u]:
            if u < v:
                if len(tree[u]) > 1 and len(tree[v]) > 1:
                    internal_edges.append((u, v))
    
    improved = True
    while improved:
        improved = False
        for u, v in internal_edges:
            neighbors_u = [x for x in tree[u] if x != v]
            neighbors_v = [x for x in tree[v] if x != u]
            
            if len(neighbors_u) != 2 or len(neighbors_v) != 2:
                continue
                
            a, b = neighbors_u
            c, d = neighbors_v
            
            tree_new = {k: set(v) for k, v in tree.items()}
            
            tree_new[u].remove(b)
            tree_new[u].add(c)
            tree_new[v].remove(c)
            tree_new[v].add(b)
            tree_new[b].remove(u)
            tree_new[b].add(v)
            tree_new[c].remove(v)
            tree_new[c].add(u)
            
            bl_new = dict(branch_lengths)
            bl_new[(u, c)] = bl_new.get((v, c), bl_new.get((c, v), 0.1))
            bl_new[(c, u)] = bl_new[(u, c)]
            bl_new[(v, b)] = bl_new.get((u, b), bl_new.get((b, u), 0.1))
            bl_new[(b, v)] = bl_new[(v, b)]
            
            ll_new = _calculate_likelihood(tree_new, sequences, bl_new)
            ll_old = _calculate_likelihood(tree, sequences, branch_lengths)
            
            if ll_new > ll_old:
                tree = tree_new
                branch_lengths = bl_new
                improved = True
                break
                
        if improved:
            internal_edges = []
            for u in tree:
                for v in tree[u]:
                    if u < v:
                        if len(tree[u]) > 1 and len(tree[v]) > 1:
                            internal_edges.append((u, v))
                            
    return tree, branch_lengths

def _tree_to_newick(tree, branch_lengths):
    # Pick a leaf as root
    leaves = [n for n in tree if len(tree[n]) == 1]
    root = leaves[0]
    
    def build_newick(node, parent):
        neighbors = [n for n in tree[node] if n != parent]
        if not neighbors:
            return node
        if len(neighbors) == 1:
            # Leaf or single child?
            # In a binary tree, internal nodes have 2 children (excluding parent)
            # Leaves have 0 children
            # But if we root at a leaf, it has 3 neighbors.
            # So root has 3 children.
            # Other internal nodes have 2 children.
            # Leaves have 0.
            child = neighbors[0]
            bl = branch_lengths.get((node, child), branch_lengths.get((child, node), 0.0))
            return f"({build_newick(child, node)}:{bl:.4f})"
        
        # Internal node
        parts = []
        for child in neighbors:
            bl = branch_lengths.get((node, child), branch_lengths.get((child, node), 0.0))
            parts.append(f"{build_newick(child, node)}:{bl:.4f}")
        return "(" + ",".join(parts) + ")"
        
    return build_newick(root, None) + ";"

def reconstruct(sequences):
    global START_TIME, TIME_LIMIT
    if time.time() - START_TIME > TIME_LIMIT:
        labels = sorted(sequences, key=lambda name: int(name[1:]))
        return ",".join(labels) + ";"
        
    tree, branch_lengths = _neighbor_joining(sequences)
    
    # Optional: Optimize branch lengths
    # For now, skip to save time
    
    # NNI Search
    tree, branch_lengths = _nni_search(tree, sequences, branch_lengths)
    
    newick = _tree_to_newick(tree, branch_lengths)
    return newick
