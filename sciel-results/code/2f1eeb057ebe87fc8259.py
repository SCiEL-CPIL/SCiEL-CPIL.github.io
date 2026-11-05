"""Phylogenetic topology recovery: NJ + SPR refinement with time budgeting."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = time.time() + 260.0  # safety margin under the 270s kill
_EPS = 1e-9


def _time_left():
    return _DEADLINE - time.time()


def _jc_distance(a, b, n):
    """Jukes-Cantor corrected distance between two DNA strings."""
    diff = np.count_nonzero(a != b)
    p = diff / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _build_distance_matrix(sequences, labels):
    """Vectorized pairwise JC distance matrix over label indices."""
    n = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    seqs = [sequences[lab] for lab in labels]
    L = len(seqs[0])
    # encode each sequence as int8 array of 0..3
    enc = np.empty((n, L), dtype=np.int8)
    for i, s in enumerate(seqs):
        # map ACGT -> 0,1,2,3
        ba = s.encode()
        arr = np.frombuffer(ba, dtype=np.int8)
        # A=65 C=67 G=71 T=84
        m = np.zeros(90, dtype=np.int8)
        m[65] = 0; m[67] = 1; m[71] = 2; m[84] = 3
        enc[i] = m[arr]
    # pairwise hamming via broadcasting (n x n x L) is heavy; do per-pair
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            diff = np.count_nonzero(enc[i] != enc[j])
            p = diff / L
            if p >= 0.7499:
                d = 3.0
            else:
                d = -0.75 * math.log(1.0 - 4.0 * p / 3.0)
            D[i, j] = d
            D[j, i] = d
    return D


def _nj_newick(D, labels):
    """Neighbor-joining on distance matrix; returns Newick string over labels."""
    n = len(labels)
    if n == 1:
        return labels[0] + ";"
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    # work with node list; internal nodes get names
    nodes = list(range(n))
    # tree representation: for each current node, a string (subtree) or label
    subtree = {i: labels[i] for i in range(n)}
    # active distance matrix over original indices; we maintain a dict keyed by node id
    # Use a mutable matrix keyed by node ids; new nodes get new ids
    mat = {tuple(p): D[p[0], p[1]] for p in [(i, j) for i in range(n) for j in range(n) if i != j]}
    mat = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[(i, j)] = D[i, j]
    next_id = n
    while len(nodes) > 3:
        m = len(nodes)
        # row sums
        total = {}
        for a in nodes:
            s = 0.0
            for b in nodes:
                if b != a:
                    s += mat[(a, b)]
            total[a] = s
        best = None
        for ii in range(m):
            a = nodes[ii]
            for jj in range(ii + 1, m):
                b = nodes[jj]
                crit = (m - 2) * mat[(a, b)] - total[a] - total[b]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        joined = "(" + subtree[a] + "," + subtree[b] + ")"
        new = next_id
        next_id += 1
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            val = 0.5 * (mat[(a, x)] + mat[(b, x)] - mat[(a, b)])
            mat[(new, x)] = val
            mat[(x, new)] = val
        nodes = rest + [new]
        subtree[new] = joined
    # finish with 3 nodes
    a, b, c = nodes
    return "(" + subtree[a] + "," + subtree[b] + "," + subtree[c] + ");"


def _newick_to_adj(newick, labels):
    """Parse a Newick string into an adjacency dict (unrooted, edges as sets of node ids).
    Returns (adj, node_count) where nodes are integers; leaves map to labels."""
    # tokenize
    s = newick
    # build a tree of nodes; we produce an unrooted adjacency
    # Use a stack-based parser producing a rooted tree then unroot.
    # Represent each node as an int id.
    adj = {}
    node_label = {}
    counter = [0]

    def new_node():
        i = counter[0]
        counter[0] += 1
        adj[i] = set()
        return i

    # We'll parse into a rooted structure then unroot by picking a leaf as root.
    # Simpler: parse producing adjacency directly using a stack of "open" nodes.
    i = 0
    n = len(s)
    stack = []  # list of (node_id, state)
    # We'll build a forest of rooted trees; track parent.
    # Use explicit recursion via stack of iterators is complex; do iterative.
    # Parse characters.
    root = None
    # stack entries: dict with 'node' id and 'children' count and 'done'
    # We'll maintain a list of nodes; when we see '(' we open a new internal node.
    # When we see a label (t0..), it's a leaf.
    # When we see ',', we finish current child and expect next.
    # When we see ')', we close current internal node (it becomes a child of its parent).
    # We'll build parent pointers.
    parent = {}
    # stack of internal nodes being built
    open_stack = []  # each entry: node id
    last_node = None
    i = 0
    while i < n:
        ch = s[i]
        if ch == '(':
            # open internal node
            node = new_node()
            parent[node] = open_stack[-1] if open_stack else None
            open_stack.append(node)
            last_node = node
        elif ch == ',':
            # child boundary; last_node was just completed, expect next sibling
            pass
        elif ch == ';':
            pass
        elif ch == ')':
            # close internal node: it is now a complete child
            node = open_stack.pop()
            last_node = node
        elif ch in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            # read a label token
            j = i
            while j < n and s[j] not in '(),;':
                j += 1
            tok = s[i:j]
            node = new_node()
            parent[node] = open_stack[-1] if open_stack else None
            node_label[node] = tok
            last_node = node
            i = j - 1
        i += 1
    # Now build adjacency from parent pointers
    for node, p in parent.items():
        if p is not None:
            adj[node].add(p)
            adj[p].add(node)
    # Unroot: pick a leaf (node with label) and re-root there.
    # Find a leaf node.
    leaves = [node for node in adj if node in node_label]
    if not leaves:
        # fallback: any node
        leaves = list(adj.keys())
    root = leaves[0]
    # BFS from root to recompute parent, then rebuild adjacency as undirected (already undirected)
    # adjacency is already undirected (we added both directions). Good.
    # Remove any self loops
    for node in list(adj.keys()):
        adj[node].discard(node)
    return adj, node_label


def _sse(tree_adj, node_label, D, labels):
    """Squared-sum of edge lengths for a tree given by adjacency + labels,
    using least-squares edge length estimation. Lower = better fit."""
    # Map labels to indices in D
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    # assign each tree node a D-index if it's a leaf, else None
    node_didx = {}
    for node, lab in node_label.items():
        if lab in lab_to_idx:
            node_didx[node] = lab_to_idx[lab]
    # Build rooted tree for least squares: pick a root leaf
    leaves = [node for node in node_didx]
    if not leaves:
        return float('inf')
    root = leaves[0]
    # BFS to get parent and order
    from collections import deque
    order = []
    par = {root: None}
    q = deque([root])
    while q:
        u = q.popleft()
        order.append(u)
        for v in tree_adj[u]:
            if v not in par:
                par[v] = u
                q.append(v)
    # For each non-root node, compute the "centroid" (mean) of its subtree's leaf positions
    # and the variance, then SSE = sum over edges of (subtree_size * (var_subtree + var_parent_rest))
    # Standard least-squares: for each edge, SSE contribution = n_sub * n_rest / (n_sub + n_rest) * d^2
    # where d is the distance between the two centroids.
    # We need leaf positions in 2D (or use a reference). Use the first leaf as origin.
    # Represent each leaf by its D-vector to all other leaves? That's heavy.
    # Instead use the fact that for least squares with all pairwise distances,
    # the optimal embedding uses the "distance matrix" and the SSE per edge
    # = (n_sub * n_rest / n) * (d(centroid_sub, centroid_rest)^2)
    # We can compute centroids in the space of the distance matrix to a reference.
    # Simpler robust approach: use the "variance" formula.
    # For a subtree S, define its centroid c_S = mean of leaf positions in S.
    # The squared distance between centroid of S and complement is:
    # d^2 = (1/(n_sub*n_rest)) * [ sum_{i in S, j not in S} d(i,j)^2 - (n_sub*n_rest) * (global mean)^2 ... ]
    # This is getting complex. Use a simpler but valid proxy:
    # SSE proxy = sum over edges of (n_sub * n_rest) * (mean_pairwise_d_between_S_and_rest)^2
    # We'll compute mean pairwise distance between S leaves and rest leaves.
    # Precompute leaf D vectors.
    leaf_nodes = list(node_didx.keys())
    leaf_didx = [node_didx[u] for u in leaf_nodes]
    # positions: use D to leaf 0 as coordinate (1D is not enough). Use 2D via two reference leaves.
    # Use reference leaf r0 and r1.
    r0 = leaf_didx[0]
    r1 = leaf_didx[1] if len(leaf_didx) > 1 else leaf_didx[0]
    # coordinate of leaf i: (D[i][r0], D[i][r1])
    coords = {}
    for u in leaf_nodes:
        di = node_didx[u]
        coords[u] = (D[di, r0], D[di, r1])
    # For each edge, find subtree leaf set
    # Precompute for each node the set of leaves in its subtree (rooted at root)
    sub_leaves = {}
    for u in reversed(order):
        s = set()
        if u in node_didx:
            s.add(u)
        for v in tree_adj[u]:
            if v != par[u]:
                s |= sub_leaves.get(v, set())
        sub_leaves[u] = s
    n_total = len(leaf_nodes)
    sse = 0.0
    for u in order:
        if par[u] is None:
            continue
        S = sub_leaves[u]
        if not S:
            continue
        n_sub = len(S)
        n_rest = n_total - n_sub
        if n_sub == 0 or n_rest == 0:
            continue
        # centroid of S
        cx = sum(coords[v][0] for v in S) / n_sub
        cy = sum(coords[v][1] for v in S) / n_sub
        # centroid of rest
        rest = [v for v in leaf_nodes if v not in S]
        cx2 = sum(coords[v][0] for v in rest) / n_rest
        cy2 = sum(coords[v][1] for v in rest) / n_rest
        d2 = (cx - cx2) ** 2 + (cy - cy2) ** 2
        sse += (n_sub * n_rest / n_total) * d2
    return sse


def _spr_refine(newick, D, labels, adj, node_label, max_time):
    """One or more rounds of SPR moves improving SSE. Returns best newick."""
    best_newick = newick
    best_sse = _sse(adj, node_label, D, labels)
    deadline = time.time() + max_time
    # Precompute leaf index map
    lab_to_idx = {lab: i for i, lab in enumerate(labels)}
    # We'll do a limited number of full passes
    passes = 0
    while time.time() < deadline and passes < 4:
        passes += 1
        improved = False
        # For each edge, try pruning a subtree and regrafting
        # Build rooted structure each pass (root at a leaf)
        from collections import deque
        leaves = [node for node in node_label if node_label[node] in lab_to_idx]
        if not leaves:
            break
        root = leaves[0]
        par = {root: None}
        order = []
        q = deque([root])
        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if v not in par:
                    par[v] = u
                    q.append(v)
        # subtree leaf sets
        sub_leaves = {}
        for u in reversed(order):
            s = set()
            if u in lab_to_idx:
                s.add(u)
            for v in adj[u]:
                if v != par[u]:
                    s |= sub_leaves.get(v, set())
            sub_leaves[u] = s
        # Try each edge (parent u, child v): prune v's subtree, regraft at each other edge
        for u in order:
            if par[u] is None:
                continue
            p = par[u]
            # candidate regraft targets: edges not in u's subtree and not the p-u edge
            # We'll try a limited set of targets for speed
            targets = [w for w in order if w != p and w != u and par[w] is not None and w not in sub_leaves[u]]
            if not targets:
                continue
            best_target = None
            best_sse_try = best_sse
            # To speed up, only try a subset of targets
            if len(targets) > 60:
                import random
                targets = random.sample(targets, 60)
            for w in targets:
                # prune u from p, attach to w
                # compute new SSE approximately: only edges on path change
                # For simplicity recompute full SSE (costly) but limited targets
                # Build new adjacency
                new_adj = {k: set(v) for k, v in adj.items()}
                # remove edge p-u
                new_adj[p].discard(u)
                new_adj[u].discard(p)
                # add edge w-u
                new_adj[w].add(u)
                new_adj[u].add(w)
                sse_try = _sse(new_adj, node_label, D, labels)
                if sse_try < best_sse_try - 1e-6:
                    best_sse_try = sse_try
                    best_target = (p, u, w)
            if best_target is not None:
                p, u, w = best_target
                adj[p].discard(u)
                adj[u].discard(p)
                adj[w].add(u)
                adj[u].add(w)
                best_sse = best_sse_try
                improved = True
        if not improved:
            break
    # Rebuild newick from adj
    best_newick = _adj_to_newick(adj, node_label, labels)
    return best_newick


def _adj_to_newick(adj, node_label, labels):
    """Convert adjacency + labels to a Newick string over the original labels."""
    if not adj:
        return ""
    # pick a root leaf
    leaves = [node for node in node_label if node_label[node] in labels]
    if not leaves:
        leaves = list(adj.keys())
    root = leaves[0]
    # recursive newick
    seen = set()

    def rec(u, parent):
        nonlocal seen
        seen.add(u)
        children = [v for v in adj[u] if v != parent]
        if not children:
            return node_label.get(u, "x")
        parts = [rec(v, u) for v in children]
        return "(" + ",".join(parts) + ")"
    return rec(root, None) + ";"


def reconstruct(sequences):
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return "(" + labels[0] + "," + labels[1] + ");"
    D = _build_distance_matrix(sequences, labels)
    newick = _nj_newick(D, labels)
    # Parse and refine
    adj, node_label = _newick_to_adj(newick, labels)
    if time.time() < _DEADLINE:
        refine_budget = min(40.0, max(2.0, _time_left() * 0.5))
        newick = _spr_refine(newick, D, labels, adj, node_label, refine_budget)
    # ensure all labels present exactly once
    for lab in labels:
        if lab not in newick:
            # fallback: append as star
            newick = "(" + newick.rstrip(';') + "," + lab + ");"
    return newick
# EVOLVE-BLOCK-END
