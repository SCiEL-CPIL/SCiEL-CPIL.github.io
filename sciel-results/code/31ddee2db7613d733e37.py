import math
import time
import numpy as np
from collections import defaultdict

BASES = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

def _jc_distance(a, b):
    n = len(a)
    if n == 0:
        return 0.0
    mismatches = sum(1 for x, y in zip(a, b) if x != y)
    p = mismatches / n
    if p >= 0.75:
        return 10.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

def _build_nj_tree(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return [], {0: labels[0]} if n == 1 else {}
    
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = _jc_distance(sequences[labels[i]], sequences[labels[j]])
            dist[i, j] = d
            dist[j, i] = d
            
    nodes = list(range(n))
    trees = {i: labels[i] for i in nodes}
    new_id = n
    edges = []
    active = list(range(n))
    d = dist.copy()
    
    while len(active) > 3:
        m = len(active)
        r = np.zeros(m)
        for i_idx, i in enumerate(active):
            for j_idx, j in enumerate(active):
                if i != j:
                    r[i_idx] += d[i, j]
        
        best_q = float('inf')
        best_i_idx = -1
        best_j_idx = -1
        
        for i_idx in range(m):
            for j_idx in range(i_idx + 1, m):
                i = active[i_idx]
                j = active[j_idx]
                q = (m - 2) * d[i, j] - r[i_idx] - r[j_idx]
                if q < best_q:
                    best_q = q
                    best_i_idx = i_idx
                    best_j_idx = j_idx
        
        i = active[best_i_idx]
        j = active[best_j_idx]
        
        delta = (r[best_i_idx] - r[best_j_idx]) / (2.0 * (m - 2))
        d_i_new = 0.5 * (d[i, j] + delta)
        d_j_new = 0.5 * (d[i, j] - delta)
        
        edges.append((i, new_id, d_i_new))
        edges.append((j, new_id, d_j_new))
        
        for k_idx, k in enumerate(active):
            if k != i and k != j:
                d[new_id, k] = 0.5 * (d[i, k] + d[j, k] - d[i, j])
                d[k, new_id] = d[new_id, k]
        
        new_active = [x for x in active if x not in (i, j)]
        new_active.append(new_id)
        active = new_active
        new_id += 1
        
    if len(active) == 3:
        a, b, c = active
        d_ab = d[a, b]
        d_bc = d[b, c]
        d_ca = d[c, a]
        
        d_a = 0.5 * (d_ab + d_ca - d_bc)
        d_b = 0.5 * (d_ab + d_bc - d_ca)
        d_c = 0.5 * (d_ca + d_bc - d_ab)
        
        center = new_id
        edges.append((a, center, d_a))
        edges.append((b, center, d_b))
        edges.append((c, center, d_c))
        active = [center]
        new_id += 1
        
    return edges, {i: labels[i] for i in range(n)}

class Tree:
    def __init__(self, labels, edges, node_map):
        self.labels = labels
        self.n = len(labels)
        self.edges = edges
        self.node_map = node_map
        
        self.adj = defaultdict(list)
        for u, v, l in edges:
            self.adj[u].append((v, l))
            self.adj[v].append((u, l))
            
        self.leaves = set(node_map.keys())
        self.internal_nodes = set()
        for node in self.adj:
            if node not in self.leaves:
                self.internal_nodes.add(node)
                
        self.root = list(self.leaves)[0]
        self._build_parent_child()
        
    def _build_parent_child(self):
        self.parent = {}
        self.children = defaultdict(list)
        
        visited = set()
        queue = [self.root]
        visited.add(self.root)
        self.parent[self.root] = None
        
        while queue:
            u = queue.pop(0)
            for v, l in self.adj[u]:
                if v not in visited:
                    visited.add(v)
                    self.parent[v] = u
                    self.children[u].append(v)
                    queue.append(v)
                    
        self.branch_lengths = {}
        for child, parent in self.parent.items():
            for v, l in self.adj[child]:
                if v == parent:
                    self.branch_lengths[child] = l
                    break
                    
    def get_likelihood(self, sequences):
        leaves_list = list(self.leaves)
        n_sites = len(sequences[leaves_list[0]])
        
        post_order = []
        def dfs(u):
            for v in self.children[u]:
                dfs(v)
            post_order.append(u)
            
        dfs(self.root)
        
        L = {}
        
        for u in post_order:
            if u in self.leaves:
                label = self.node_map[u]
                seq = sequences[label]
                L[u] = np.zeros((n_sites, 4))
                for s in range(n_sites):
                    base = BASES[seq[s]]
                    L[u][s, base] = 1.0
            else:
                L[u] = np.ones((n_sites, 4))
                
                for v in self.children[u]:
                    length = self.branch_lengths[v]
                    exp_term = math.exp(-4.0 * length / 3.0)
                    p_same = 0.25 + 0.75 * exp_term
                    p_diff = 0.25 * (1.0 - exp_term)
                    
                    sum_Lv = np.sum(L[v], axis=1)
                    factor = p_same - p_diff
                    const = p_diff * sum_Lv
                    
                    for i in range(4):
                        L[u][:, i] += factor * L[v][:, i] + const
                        
        root_L = L[self.root]
        total_prob = np.sum(root_L, axis=1)
        total_prob = np.maximum(total_prob, 1e-100)
        log_likelihood = np.sum(np.log(total_prob))
        
        return log_likelihood

    def apply_spr_move(self, u, v, w):
        p_u = self.parent[u]
        len_u = self.branch_lengths[u]
        len_vw = self.branch_lengths[w]
        
        self.adj[p_u].remove((u, len_u))
        self.adj[u].remove((p_u, len_u))
        
        self.adj[v].remove((w, len_vw))
        self.adj[w].remove((v, len_vw))
        
        len_vu = 0.1
        len_uw = 0.1
        
        self.adj[v].append((u, len_vu))
        self.adj[u].append((v, len_vu))
        self.adj[u].append((w, len_uw))
        self.adj[w].append((u, len_uw))
        
        self._build_parent_child()

def reconstruct(sequences):
    start_time = time.time()
    deadline = start_time + 250.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return ";".join(labels)
        
    edges, node_map = _build_nj_tree(sequences)
    tree = Tree(labels, edges, node_map)
    
    import scipy.optimize
    
    def neg_log_likelihood(branch_lengths_list):
        for child, l in zip(tree.parent.keys(), branch_lengths_list):
            tree.branch_lengths[child] = max(l, 1e-5)
        return -tree.get_likelihood(sequences)
        
    initial_lengths = [tree.branch_lengths[child] for child in tree.parent.keys()]
    
    try:
        result = scipy.optimize.minimize(neg_log_likelihood, initial_lengths, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
        for child, l in zip(tree.parent.keys(), result.x):
            tree.branch_lengths[child] = max(l, 1e-5)
    except:
        pass
    
    best_likelihood = -neg_log_likelihood([tree.branch_lengths[child] for child in tree.parent.keys()])
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline and iteration < 10:
        improved = False
        iteration += 1
        best_move = None
        best_gain = 0
        
        for u in list(tree.parent.keys()):
            if u == tree.root:
                continue
            for v in list(tree.parent.keys()):
                if v == tree.root:
                    continue
                for w in tree.children[v]:
                    # Check if u is in subtree of w
                    # Simple DFS
                    def in_subtree(node, target):
                        if node == target:
                            return True
                        for child in tree.children[node]:
                            if in_subtree(child, target):
                                return True
                        return False
                        
                    if in_subtree(w, u):
                        continue
                    
                    # Save state
                    old_adj = {k: list(v) for k, v in tree.adj.items()}
                    old_parent = dict(tree.parent)
                    old_children = {k: list(v) for k, v in tree.children.items()}
                    old_branches = dict(tree.branch_lengths)
                    
                    tree.apply_spr_move(u, v, w)
                    
                    initial_lengths = [tree.branch_lengths[child] for child in tree.parent.keys()]
                    try:
                        result = scipy.optimize.minimize(neg_log_likelihood, initial_lengths, method='Nelder-Mead', options={'maxiter': 10, 'xatol': 1e-3, 'fatol': 1e-3})
                        for child, l in zip(tree.parent.keys(), result.x):
                            tree.branch_lengths[child] = max(l, 1e-5)
                    except:
                        pass
                        
                    new_likelihood = -neg_log_likelihood([tree.branch_lengths[child] for child in tree.parent.keys()])
                    gain = new_likelihood - best_likelihood
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (u, v, w)
                        best_likelihood = new_likelihood
                    else:
                        tree.adj = old_adj
                        tree.parent = old_parent
                        tree.children = old_children
                        tree.branch_lengths = old_branches
                        tree._build_parent_child()
                        
                    if time.time() > deadline:
                        break
                        
        if best_move:
            improved = True
        else:
            improved = False
            
    def to_newick(node):
        if node in tree.leaves:
            return tree.node_map[node]
        
        children = tree.children[node]
        if len(children) == 1:
            return to_newick(children[0])
            
        child_strings = []
        for child in children:
            length = tree.branch_lengths[child]
            child_str = to_newick(child)
            child_strings.append(f"{child_str}:{length:.4f}")
            
        return "(" + ",".join(child_strings) + ")"
        
    newick = to_newick(tree.root) + ";"
    return newick
