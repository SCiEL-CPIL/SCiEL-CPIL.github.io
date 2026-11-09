import time
import math
import numpy as np
from collections import Counter
import re
import scipy.optimize

# EVOLVE-BLOCK-START

class Node:
    def __init__(self, name=None):
        self.name = name
        self.children = []
        self.parent = None
        self.branch_length = 0.0
        self.id = 0

def _encode_sequences(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    data = np.zeros((n, L), dtype=np.int8)
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            data[i, j] = base_map[ch]
    return data

def _jc_distance_matrix(sequences, labels):
    n = len(labels)
    L = len(sequences[labels[0]])
    codes = np.zeros((n, L, 4), dtype=np.int8)
    base_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i, lab in enumerate(labels):
        s = sequences[lab]
        for j, ch in enumerate(s):
            codes[i, j, base_map[ch]] = 1
    D = np.zeros((n, n), dtype=np.float64)
    for b in range(4):
        M = codes[:, :, b]
        agree = M @ M.T
        D += agree
    D = 4.0 * L - D
    np.fill_diagonal(D, 0.0)
    p = D / float(L)
    p = np.clip(p, 0.0, 0.7499)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D

def _neighbor_joining(D, labels):
    n = len(labels)
    if n == 1:
        return Node(labels[0])
    if n == 2:
        root = Node()
        n1 = Node(labels[0])
        n2 = Node(labels[1])
        root.children = [n1, n2]
        n1.parent = root
        n2.parent = root
        return root
    if n == 3:
        root = Node()
        c1 = Node(labels[0])
        c2 = Node()
        c2.children = [Node(labels[1]), Node(labels[2])]
        c2.children[0].parent = c2
        c2.children[1].parent = c2
        root.children = [c1, c2]
        c1.parent = root
        c2.parent = root
        return root

    dist_mat = D.copy()
    tree_str = {i: labels[i] for i in range(n)}
    active_set = set(range(n))
    nid = n
    
    while len(active_set) > 3:
        m = len(active_set)
        total = np.zeros(n, dtype=np.float64)
        for a in active_set:
            total[a] = np.sum([dist_mat[a, b] for b in active_set if b != a])
        best_crit = float('inf')
        best_a, best_b = None, None
        act_list = sorted(active_set)
        for ia in range(len(act_list)):
            a = act_list[ia]
            for ib in range(ia + 1, len(act_list)):
                b = act_list[ib]
                crit = (m - 2) * dist_mat[a, b] - total[a] - total[b]
                if crit < best_crit:
                    best_crit = crit
                    best_a, best_b = a, b
        a, b = best_a, best_b
        new = nid
        nid += 1
        old_n = dist_mat.shape[0]
        new_dist = np.zeros((old_n + 1, old_n + 1), dtype=np.float64)
        new_dist[:old_n, :old_n] = dist_mat
        rest = [x for x in active_set if x != a and x != b]
        for x in rest:
            val = 0.5 * (dist_mat[a, x] + dist_mat[b, x] - dist_mat[a, b])
            new_dist[new, x] = val
            new_dist[x, new] = val
        dist_mat = new_dist
        tree_str[new] = "(" + tree_str[a] + "," + tree_str[b] + ")"
        active_set.discard(a)
        active_set.discard(b)
        active_set.add(new)
        
    rem = sorted(active_set)
    newick = "(" + ",".join(tree_str[x] for x in rem) + ");"
    
    return _parse_newick_to_nodes(newick, n)

def _parse_newick_to_nodes(newick, n):
    pos = 0
    s = newick.rstrip(';')
    next_id = 0
    def parse():
        nonlocal pos, next_id
        if s[pos] == '(':
            pos += 1
            node = Node()
            while True:
                child = parse()
                node.children.append(child)
                child.parent = node
                if s[pos] == ',':
                    pos += 1
                elif s[pos] == ')':
                    pos += 1
                    break
            return node
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',);':
                pos += 1
            name = s[start:pos].strip()
            node = Node(name)
            return node
    root = parse()
    def assign_ids(node):
        nonlocal next_id
        node.id = next_id
        next_id += 1
        for c in node.children:
            assign_ids(c)
    assign_ids(root)
    return root

def _log_likelihood(tree, data):
    nodes_post = []
    def dfs_post(node):
        for c in node.children:
            dfs_post(c)
        nodes_post.append(node)
    dfs_post(tree)
    
    likelihoods = {}
    L = data.shape[1]
    
    for node in nodes_post:
        if node.name is not None:
            idx = int(node.name[1:])
            v = np.zeros((4, L), dtype=np.float64)
            v[data[idx], np.arange(L)] = 1.0
            likelihoods[node.id] = v
        else:
            c1 = node.children[0]
            c2 = node.children[1]
            t1 = c1.branch_length
            t2 = c2.branch_length
            
            def get_P(t):
                e = math.exp(-4.0 * t / 3.0)
                p_ii = 0.25 + 0.75 * e
                p_ij = 0.25 - (1.0/12.0) * e
                P = np.full((4, 4), p_ij)
                np.fill_diagonal(P, p_ii)
                return P
            
            P1 = get_P(t1)
            P2 = get_P(t2)
            
            v1 = likelihoods[c1.id]
            v2 = likelihoods[c2.id]
            
            A1 = P1 @ v1
            A2 = P2 @ v2
            v_i = A1 * A2
            likelihoods[node.id] = v_i

    v_root = likelihoods[tree.id]
    total_ll = np.sum(0.25 * np.sum(v_root, axis=0))
    return np.log(total_ll)

def _optimize_branch_lengths(tree, data, max_iter=20):
    internal_nodes = [n for n in _traverse(tree) if n.name is None]
    if not internal_nodes:
        return
    
    def objective(branch_lengths):
        for i, node in enumerate(internal_nodes):
            node.branch_length = branch_lengths[i]
        return -_log_likelihood(tree, data)
    
    x0 = np.array([n.branch_length for n in internal_nodes])
    res = scipy.optimize.minimize(objective, x0, method='Powell', options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
    for i, node in enumerate(internal_nodes):
        node.branch_length = max(0.0, res.x[i])

def _traverse(node):
    yield node
    for c in node.children:
        yield from _traverse(c)

def _build_newick_from_nodes(root):
    def to_newick(node):
        if node.name is not None:
            return node.name
        else:
            return "(" + ",".join(to_newick(c) for c in node.children) + ")"
    return to_newick(root) + ";"

def reconstruct(sequences):
    t0 = time.time()
    deadline = 240.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 3:
        return "(" + ",".join(labels) + ");"
    
    try:
        D = _jc_distance_matrix(sequences, labels)
        tree = _neighbor_joining(D, labels)
        
        data = _encode_sequences(sequences, labels)
        _optimize_branch_lengths(tree, data)
        
        newick = _build_newick_from_nodes(tree)
        
        found = re.findall(r't\d+', newick)
        cnt = Counter(found)
        if len(cnt) == n and all(cnt[lab] == 1 for lab in labels):
            return newick
    except Exception:
        pass
        
    return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
