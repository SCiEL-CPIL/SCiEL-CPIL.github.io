"""ML-based phylogeny reconstruction under Jukes-Cantor with NNI hill-climbing."""
import math
import numpy as np


def _encode(seq):
    out = np.empty(len(seq), dtype=np.int8)
    for i, ch in enumerate(seq):
        c = ord(ch)
        if c == 65:
            out[i] = 0
        elif c == 67:
            out[i] = 1
        elif c == 71:
            out[i] = 2
        else:
            out[i] = 3
    return out


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"(({labels[0]},{labels[1]}),{labels[2]});"

    data = {lab: _encode(sequences[lab]) for lab in labels}
    L = len(sequences[labels[0]])
    data_mat = np.array([data[lab] for lab in labels], dtype=np.int8)

    # ---- NJ initial tree ----
    def _distance(a, b):
        diff = int(np.sum(data[a] != data[b]))
        p = diff / L
        if p >= 0.7499:
            return 3.0
        return -0.75 * math.log(1.0 - 4.0 * p / 3.0)

    nodes = list(labels)
    trees = {name: name for name in labels}
    matrix = {}
    for a in labels:
        for b in labels:
            if a != b:
                matrix[(a, b)] = _distance(a, b)
    while len(nodes) > 3:
        nn = len(nodes)
        total = {a: sum(matrix[(a, b)] for b in nodes if b != a) for a in nodes}
        best = None
        for i, a in enumerate(nodes):
            for b in nodes[i + 1:]:
                criterion = (nn - 2) * matrix[(a, b)] - total[a] - total[b]
                if best is None or criterion < best[0]:
                    best = (criterion, a, b)
        _, a, b = best
        joined = f"({trees[a]},{trees[b]})"
        new = f"n{len(nodes)}_{a}_{b}"
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            matrix[(new, x)] = matrix[(x, new)] = 0.5 * (matrix[(a, x)] + matrix[(b, x)] - matrix[(a, b)])
        nodes = rest + [new]
        trees[new] = joined
    nj_newick = "(" + ",".join(trees[x] for x in nodes) + ");"

    # ---- Parse newick to adjacency + edge lengths ----
    # Build a tree structure. We'll use node IDs: leaves 0..n-1, internal n..
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Parse newick
    def parse(s):
        pos = [0]
        def parse_node():
            if s[pos[0]] == '(':
                pos[0] += 1
                left = parse_node()
                pos[0] += 1  # skip ','
                right = parse_node()
                pos[0] += 1  # skip ')'
                return ('int', [left, right])
            else:
                name = ''
                while pos[0] < len(s) and s[pos[0]] not in (',', ')', ';'):
                    name += s[pos[0]]
                    pos[0] += 1
                return ('leaf', name)
        root = parse_node()
        return root

    root = parse(nj_newick)

    # Assign IDs and build adjacency + edge lengths
    next_id = [n]
    adj = {}
    edge_bl = {}

    def assign(node, parent_id=None, bl=None):
        if node[0] == 'leaf':
            idx = label_to_idx[node[1]]
            adj[idx] = set()
            if parent_id is not None:
                adj[idx].add(parent_id)
                adj[parent_id].add(idx)
                if bl is not None:
                    edge_bl[frozenset([idx, parent_id])] = bl
            return idx
        else:
            nid = next_id[0]
            next_id[0] += 1
            adj[nid] = set()
            if parent_id is not None:
                adj[nid].add(parent_id)
                adj[parent_id].add(nid)
                if bl is not None:
                    edge_bl[frozenset([nid, parent_id])] = bl
            kids = node[1]
            c1 = assign(kids[0], nid, 0.1)
            c2 = assign(kids[1], nid, 0.1)
            return nid

    root_id = assign(root)
    total_nodes = next_id[0]

    # ---- ML scoring ----
    def root_tree(root_id):
        parent = {root_id: None}
        children = {i: [] for i in adj}
        stack = [root_id]
        visited = {root_id}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    children[u].append(v)
                    stack.append(v)
        return parent, children

    def compute_loglik(parent, children):
        post = []
        def postorder(u):
            for v in children[u]:
                postorder(v)
            post.append(u)
        postorder(0)  # root at 0
        profiles = {}
        for u in post:
            if u < n:
                prof = np.zeros((4, L), dtype=np.float64)
                prof[data_mat[u], np.arange(L)] = 1.0
                profiles[u] = prof
            else:
                prof = np.ones((4, L), dtype=np.float64)
                for c in children[u]:
                    Lc = profiles[c]
                    Sc = Lc.sum(axis=0)
                    d = edge_bl.get(frozenset([u, c]), 0.1)
                    q = math.exp(-4.0 * d / 3.0)
                    M = q * Lc + 0.25 * (1.0 - q) * Sc[np.newaxis, :]
                    prof = prof * M
                profiles[u] = prof
        root_prof = profiles[0]
        site_loglik = np.log(root_prof.sum(axis=0) / 4.0)
        site_loglik = np.where(site_loglik > -1e30, site_loglik, -1e30)
        return float(np.sum(site_loglik))

    parent, children = root_tree(0)
    current_ll = compute_loglik(parent, children)

    # Optimize branch lengths
    def optimize_bls(iters=2):
        nonlocal current_ll
        for _ in range(iters):
            for key in list(edge_bl.keys()):
                old = edge_bl[key]
                best_bl = old
                best_ll = current_ll
                for bl in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]:
                    edge_bl[key] = bl
                    ll = compute_loglik(parent, children)
                    if ll > best_ll:
                        best_ll = ll
                        best_bl = bl
                edge_bl[key] = best_bl
                if best_ll > current_ll:
                    current_ll = best_ll
        return current_ll

    current_ll = optimize_bls(2)

    # NNI
    def nni_round():
        nonlocal current_ll, parent, children
        improved = False
        for u in range(n, total_nodes):
            for v in adj[u]:
                if v < n or v <= u:
                    continue
                # u-v internal edge
                if parent.get(v) == u:
                    u_kids = [k for k in children[u] if k != v]
                    if not u_kids or len(children[v]) != 2:
                        continue
                    a = u_kids[0]
                    b, c = children[v]
                    # Try swapping a with b, and a with c
                    for swap_target in [b, c]:
                        # Save
                        orig_adj = {k: set(vv) for k, vv in adj.items()}
                        orig_bl = dict(edge_bl)
                        # Apply
                        adj[u].discard(a)
                        adj[u].add(swap_target)
                        adj[v].discard(swap_target)
                        adj[v].add(a)
                        # Update edge lengths
                        key_ua = frozenset([u, a])
                        key_vt = frozenset([v, swap_target])
                        bl_ua = edge_bl.get(key_ua, 0.1)
                        bl_vt = edge_bl.get(key_vt, 0.1)
                        if key_ua in edge_bl:
                            del edge_bl[key_ua]
                        if key_vt in edge_bl:
                            del edge_bl[key_vt]
                        edge_bl[frozenset([u, swap_target])] = bl_ua
                        edge_bl[frozenset([v, a])] = bl_vt
                        parent, children = root_tree(0)
                        ll = compute_loglik(parent, children)
                        if ll > current_ll:
                            current_ll = ll
                            improved = True
                            # Keep the change
                        else:
                            # Revert
                            adj = {k: set(vv) for k, vv in orig_adj.items()}
                            edge_bl = dict(orig_bl)
                            parent, children = root_tree(0)
        return improved

    for _ in range(3):
        if not nni_round():
            break
        current_ll = optimize_bls(1)

    # ---- Convert to newick ----
    def to_newick(u, is_root=False):
        if u < n:
            return labels[u]
        kids = children[u]
        if not kids:
            return f"n{u}"
        parts = [to_newick(k) for k in kids]
        s = "(" + ",".join(parts) + ")"
        if not is_root:
            d = edge_bl.get(frozenset([u, parent[u]]), 0.1)
            s += f":{d:.4f}"
        return s

    return to_newick(0, True) + ";"
