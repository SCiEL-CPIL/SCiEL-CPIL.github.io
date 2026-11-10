"""Phylogenetic topology recovery: NJ with JC distances, then MP/NNI refinement."""
import math
import time
import numpy as np

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 255.0  # safety margin before 270s kill


def _jc_dist_matrix(sequences, labels):
    """Compute Jukes-Cantor corrected distance matrix using numpy."""
    n = len(labels)
    seq_list = [sequences[lb] for lb in labels]
    L = len(seq_list[0])
    arr = np.array([list(s) for s in seq_list])
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = np.sum(arr[i] != arr[j])
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_clean(sequences, labels):
    """Clean Neighbor Joining implementation."""
    n = len(labels)
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]})" + ";"
    
    seqs = [sequences[lb] for lb in labels]
    L = len(seqs[0])
    
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            mismatches = sum(1 for x, y in zip(seqs[i], seqs[j]) if x != y)
            p = mismatches / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    
    active = list(range(n))
    node_subtree = {i: labels[i] for i in range(n)}
    dist = {}
    for i in range(n):
        for j in range(i + 1, n):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
    
    next_node_id = n
    
    while len(active) > 3:
        if time.time() > _DEADLINE:
            break
        
        m = len(active)
        row_sum = {a: 0.0 for a in active}
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i != j:
                    row_sum[a] += dist[(a, b)]
        
        best_criterion = float('inf')
        best_a = active[0]
        best_b = active[1]
        for i, a in enumerate(active):
            for j, b in enumerate(active):
                if i >= j:
                    continue
                criterion = (m - 2) * dist[(a, b)] - row_sum[a] - row_sum[b]
                if criterion < best_criterion:
                    best_criterion = criterion
                    best_a = a
                    best_b = b
        
        a = best_a
        b = best_b
        
        new_node = next_node_id
        next_node_id += 1
        
        node_subtree[new_node] = f"({node_subtree[a]},{node_subtree[b]})"
        
        remaining = [x for x in active if x != a and x != b]
        for x in remaining:
            new_d = 0.5 * (dist[(a, x)] + dist[(b, x)] - dist[(a, b)])
            dist[(new_node, x)] = new_d
            dist[(x, new_node)] = new_d
        
        for x in remaining:
            del dist[(a, x)]
            del dist[(x, a)]
            del dist[(b, x)]
            del dist[(x, b)]
        del dist[(a, b)]
        del dist[(b, a)]
        
        active = remaining + [new_node]
    
    if len(active) == 3:
        a, b, c = active
        return f"({node_subtree[a]},{node_subtree[b]},{node_subtree[c]})" + ";"
    elif len(active) == 2:
        a, b = active
        return f"({node_subtree[a]},{node_subtree[b]})" + ";"
    elif len(active) == 1:
        return node_subtree[active[0]] + ";"
    else:
        return "(" + ",".join(node_subtree[x] for x in active) + ")" + ";"


class _UnrootedTree:
    """Unrooted tree for MP operations."""
    
    def __init__(self, n_leaves, labels):
        self.n_leaves = n_leaves
        self.labels = labels
        self.n_nodes = 2 * n_leaves - 3 if n_leaves >= 3 else n_leaves
        self.adj = [[] for _ in range(self.n_nodes)]
        self.is_leaf = [True] * n_leaves + [False] * (self.n_nodes - n_leaves)
        self.leaf_to_node = {labels[i]: i for i in range(n_leaves)}
        self.node_to_leaf = {i: labels[i] for i in range(n_leaves)}
    
    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
            self.adj[v].append(u)
    
    def remove_edge(self, u, v):
        if v in self.adj[u]:
            self.adj[u].remove(v)
        if u in self.adj[v]:
            self.adj[v].remove(u)
    
    def get_neighbors(self, u):
        return list(self.adj[u])
    
    def to_newick(self):
        if self.n_leaves == 1:
            return self.labels[0] + ";"
        if self.n_leaves == 2:
            return f"({self.labels[0]},{self.labels[1]})" + ";"
        
        visited = set()
        
        def node_to_newick(node, parent_node):
            if node in visited:
                return ""
            visited.add(node)
            
            if self.is_leaf[node]:
                return self.node_to_leaf[node]
            
            child_newicks = []
            for c in self.adj[node]:
                if c != parent_node:
                    cn = node_to_newick(c, node)
                    if cn:
                        child_newicks.append(cn)
            
            if not child_newicks:
                return f"int{node}"
            return "(" + ",".join(child_newicks) + ")"
        
        root = 0
        newick = node_to_newick(root, -1)
        return newick + ";"


def _build_tree_from_newick(newick, labels):
    """Parse Newick into _UnrootedTree."""
    newick = newick.strip().rstrip(';')
    n = len(labels)
    tree = _UnrootedTree(n, labels)
    
    if n <= 2:
        return tree
    
    # Parse and build
    next_internal = n
    stack = []
    
    def parse(s, pos):
        nonlocal next_internal
        if pos >= len(s):
            return None, pos
        
        if s[pos] == '(':
            pos += 1
            child_ids = []
            while True:
                if pos >= len(s):
                    break
                if s[pos] == ')':
                    pos += 1
                    break
                child_id, pos = parse(s, pos)
                if child_id is not None:
                    child_ids.append(child_id)
                if pos < len(s) and s[pos] == ',':
                    pos += 1
            
            node_id = next_internal
            next_internal += 1
            for c in child_ids:
                tree.add_edge(node_id, c)
            
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            
            return node_id, pos
        else:
            start = pos
            while pos < len(s) and s[pos] not in ',)':
                pos += 1
            label = s[start:pos].strip()
            if label in tree.leaf_to_node:
                return tree.leaf_to_node[label], pos
            return -1, pos
    
    root, _ = parse(newick, 0)
    return tree


def _fitch_score(tree, sequences, labels):
    """Compute parsimony score using Fitch optimization."""
    n = tree.n_leaves
    if n <= 2:
        return 0
    
    L = len(sequences[labels[0]])
    state_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    
    # Precompute leaf states
    leaf_states = {}
    for i, lb in enumerate(labels):
        seq = sequences[lb]
        leaf_states[i] = [state_map[ch] for ch in seq]
    
    # Build post-order
    post_order = []
    visited = set()
    
    def dfs(node, parent_node):
        visited.add(node)
        for c in tree.adj[node]:
            if c != parent_node:
                dfs(c, node)
        post_order.append(node)
    
    root = 0
    dfs(root, -1)
    
    total_score = 0
    for site in range(L):
        node_mask = {}
        
        for node in post_order:
            if tree.is_leaf[node]:
                # Find leaf index
                lb = tree.node_to_leaf[node]
                idx = labels.index(lb)
                state = leaf_states[idx][site]
                node_mask[node] = 1 << state
            else:
                child_masks = []
                for c in tree.adj[node]:
                    if c in node_mask:
                        child_masks.append(node_mask[c])
                
                if not child_masks:
                    node_mask[node] = 15
                else:
                    intersection = child_masks[0]
                    for cm in child_masks[1:]:
                        intersection &= cm
                    
                    if intersection:
                        node_mask[node] = intersection
                    else:
                        node_mask[node] = 15
                        total_score += 1
        
        # Downward pass for additional changes
        for node in reversed(post_order):
            if not tree.is_leaf[node]:
                for c in tree.adj[node]:
                    if c in node_mask and node in node_mask:
                        if (node_mask[c] & ~node_mask[node]) != 0:
                            total_score += 1
    
    return total_score


def _nni_refine_mp(tree, sequences, labels, max_sweeps=3, time_budget=15.0):
    """Refine tree using NNI with parsimony scoring."""
    n = tree.n_leaves
    if n < 4:
        return tree
    
    start = time.time()
    current_score = _fitch_score(tree, sequences, labels)
    
    for sweep in range(max_sweeps):
        if time.time() - start > time_budget:
            break
        
        improved = False
        
        for i in range(tree.n_nodes):
            if tree.is_leaf[i]:
                continue
            for j in tree.adj[i]:
                if tree.is_leaf[j] or j <= i:
                    continue
                
                i_neigh = [x for x in tree.adj[i] if x != j]
                j_neigh = [x for x in tree.adj[j] if x != i]
                
                if not i_neigh or not j_neigh:
                    continue
                
                a = i_neigh[0]
                b = j_neigh[0]
                
                # Save
                i_adj = list(tree.adj[i])
                j_adj = list(tree.adj[j])
                a_adj = list(tree.adj[a])
                b_adj = list(tree.adj[b])
                
                # NNI: remove i-j, i-a, j-b; add i-b, j-a
                tree.remove_edge(i, j)
                tree.remove_edge(i, a)
                tree.remove_edge(j, b)
                tree.add_edge(i, b)
                tree.add_edge(j, a)
                
                new_score = _fitch_score(tree, sequences, labels)
                
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    tree.remove_edge(i, b)
                    tree.remove_edge(j, a)
                    tree.add_edge(i, j)
                    tree.add_edge(i, a)
                    tree.add_edge(j, b)
        
        if not improved:
            break
    
    return tree


def reconstruct(sequences):
    """
    Reconstruct phylogenetic tree topology from DNA sequences.
    Uses NJ for initial tree, then MP/NNI refinement.
    """
    start_time = time.time()
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return labels[0] + ";"
        return f"({labels[0]},{labels[1]})" + ";"
    
    if time.time() > _DEADLINE - 10:
        return "(" + ",".join(labels) + ")" + ";"
    
    # NJ initial tree
    newick = _nj_clean(sequences, labels)
    
    if time.time() > _DEADLINE - 20:
        return newick
    
    # Parse and refine with MP
    try:
        tree = _build_tree_from_newick(newick, labels)
        tree = _nni_refine_mp(tree, sequences, labels, max_sweeps=3, time_budget=15.0)
        return tree.to_newick()
    except Exception:
        return newick
# EVOLVE-BLOCK-END
