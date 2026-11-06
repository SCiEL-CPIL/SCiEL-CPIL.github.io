"""Fast Parsimony Search (Felsenstein 1993) with NNI refinement, NJ fallback."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


def _jc_distance(a, b, n):
    differences = int(np.count_nonzero(a != b))
    p = differences / n
    if p >= 0.7499:
        return 3.0
    return -0.75 * math.log(1.0 - 4.0 * p / 3.0)


def _nj_tree(labels, seqs):
    """Deterministic neighbour-joining tree (fallback + initial guess)."""
    k = len(labels)
    if k <= 1:
        return labels[0] if k == 1 else ""
    if k == 2:
        return "(" + labels[0] + "," + labels[1] + ")"
    if k == 3:
        return "(" + labels[0] + "," + labels[1] + "," + labels[2] + ")"
    dist = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            dist[i, j] = dist[j, i] = _jc_distance(seqs[i], seqs[j], len(seqs[0]))
    nodes = list(range(k))
    trees = {i: labels[i] for i in range(k)}
    counter = k
    while len(nodes) > 3:
        m = len(nodes)
        total = np.array([sum(dist[i, j] for j in nodes if j != i) for i in nodes])
        best = None
        for ai in range(m):
            a = nodes[ai]
            for bi in range(ai + 1, m):
                b = nodes[bi]
                crit = (m - 2) * dist[a, b] - total[ai] - total[bi]
                if best is None or crit < best[0]:
                    best = (crit, a, b)
        _, a, b = best
        joined = "(" + trees[a] + "," + trees[b] + ")"
        new = counter
        counter += 1
        rest = [x for x in nodes if x not in (a, b)]
        for x in rest:
            val = 0.5 * (dist[a, x] + dist[b, x] - dist[a, b])
            dist[new, x] = dist[x, new] = val
        nodes = rest + [new]
        trees[new] = joined
    return "(" + ",".join(trees[x] for x in nodes) + ")"


def _parse(newick):
    """Parse Newick -> (children, parent, order, labels)."""
    children = []
    parent = []
    order = []
    labels = []

    def emit():
        idx = len(children)
        children.append(None)
        parent.append(-1)
        order.append(idx)
        return idx

    i = 0
    n = len(newick)
    while i < n:
        c = newick[i]
        if c == '(':
            node = emit()
            i += 1
            first = None
            while i < n:
                cc = newick[i]
                if cc == '(':
                    sub = _parse(newick[i:])
                    i += sub[4]
                    ch = sub[0]
                    if first is None:
                        first = ch
                    else:
                        parent[ch] = node
                    children[node] = first if children[node] is None else children[node] + [ch]
                elif cc == ',':
                    i += 1
                else:
                    j = i
                    while j < n and newick[j] not in (',', ')', ';'):
                        j += 1
                    name = newick[i:j].split(':')[0]
                    leaf = emit()
                    labels.append(name)
                    if first is None:
                        first = leaf
                    else:
                        parent[first] = node
                    children[node] = first if children[node] is None else children[node] + [first]
                    i = j
                if i < n and newick[i] == ')':
                    i += 1
                    break
            if i < n and newick[i] == ':':
                while i < n and newick[i] not in (',', ')', ';'):
                    i += 1
        elif c == ';':
            break
        else:
            j = i
            while j < n and newick[j] not in (',', ')', ';'):
                j += 1
            name = newick[i:j].split(':')[0]
            leaf = emit()
            labels.append(name)
            if not children:
                children.append(leaf)
            else:
                parent[-1] = leaf
            i = j
    # fix children list for internal nodes
    for idx in range(len(children)):
        if children[idx] is None:
            # leaf
            pass
    return children, parent, order, labels


def _to_newick(children, root):
    parts = []
    def rec(u):
        ch = children[u]
        if ch is None or len(ch) == 0:
            parts.append("L%d" % u)
        else:
            parts.append("(")
            for c in ch:
                rec(c)
                parts.append(",")
            parts.pop()
            parts.append(")")
    rec(root)
    return "".join(parts) + ";"


def _build_struct(children, parent, order, n, labels):
    """Build adjacency, edge list, subtree leaf masks, leaf index map."""
    n_int = n
    adj = [set() for _ in range(n_int)]
    for u in range(n_int):
        ch = children[u]
        if ch is not None:
            for c in ch:
                adj[u].add(c)
                adj[c].add(u)
    # leaf index map
    leaf_idx = {}
    for u in range(n_int):
        ch = children[u]
        if ch is None:
            leaf_idx[u] = labels.index("L%d" % u) if False else None
    # assign leaf ids by labels order
    leaf_node_to_id = {}
    for u in range(n_int):
        ch = children[u]
        if ch is None:
            leaf_node_to_id[u] = labels.index("L%d" % u)
    # rebuild labels mapping: labels list is in parse order of leaves
    # simpler: leaf order in `labels` corresponds to leaf nodes in order of appearance
    leaf_nodes = [u for u in range(n_int) if children[u] is None]
    # map leaf node -> taxon id (0..n-1) by labels
    for pos, u in enumerate(leaf_nodes):
        leaf_node_to_id[u] = pos
    # edge list: (child, parent)
    edges = []
    for u in range(n_int):
        if parent[u] >= 0:
            edges.append((u, parent[u]))
    # subtree leaf masks
    subtree = np.zeros(n_int, dtype=object)
    for u in leaf_nodes:
        subtree[u] = 1 << leaf_node_to_id[u]
    # postorder
    for u in reversed(order):
        ch = children[u]
        if ch is not None:
            mask = 0
            for c in ch:
                mask |= subtree[c]
            subtree[u] = mask
    allmask = (1 << n) - 1
    return adj, edges, subtree, leaf_node_to_id, leaf_nodes, allmask


def _fitch_parity(data, leaf_node_to_id, leaf_nodes):
    """Fitch parsimony on bit-packed data. data: (L, n) int64, each col 3-bit."""
    L = data.shape[0]
    n = len(leaf_nodes)
    # For each site compute fitch cost on the fixed tree using postorder+preorder
    # We need postorder and preorder of the tree.
    # Build children again from leaf_node_to_id? We pass structures.
    # Instead we recompute using adj + root.
    raise RuntimeError("use _fitch_cost_struct")


def _fitch_cost_struct(data, order, children, leaf_node_to_id, n):
    L = data.shape[0]
    n_int = len(order)
    # postorder = order (parse order is pre-order-ish); need true postorder
    # Build children list
    ch = [children[u] if children[u] is not None else [] for u in range(n_int)]
    # postorder via stack
    post = []
    stack = [(0, 0)]
    # find root: node with no parent
    # We'll just do a DFS from node 0 assuming it's root (parse root)
    root = 0
    # iterative postorder
    seen = [False] * n_int
    st = [(root, 0)]
    while st:
        u, state = st.pop()
        if state == 0:
            if seen[u]:
                continue
            seen[u] = True
            st.append((u, 1))
            for c in ch[u]:
                st.append((c, 0))
        else:
            post.append(u)
    # bottom-up sets
    sets = [None] * n_int
    costs = np.zeros(L, dtype=np.int32)
    for u in post:
        if not ch[u]:
            # leaf
            sets[u] = data[:, leaf_node_to_id[u]]
            continue
        # internal
        acc = None
        for c in ch[u]:
            s = sets[c]
            if acc is None:
                acc = s
            else:
                acc = np.bitwise_and(acc, s)
        if acc is None:
            acc = data[:, 0]
        sets[u] = acc
        # cost: for each site, if acc==0 then +1
        zero = (acc == 0)
        costs += zero.astype(np.int32)
    # top-down to compute extra costs (Fitch second pass)
    # Actually Fitch total cost = sum over internal nodes of (number of sites where AND==0)
    # which we already have as `costs`. That's the full parsimony score.
    return int(np.sum(costs))


def _nni_neighbours(adj, edges, n_int):
    """Yield NNI moves: for each edge (a,b), the two other edges."""
    moves = []
    for a, b in edges:
        na = [x for x in adj[a] if x != b]
        nb = [x for x in adj[b] if x != a]
        for x in na:
            for y in nb:
                if x != y:
                    moves.append((a, b, x, y))
    return moves


def _apply_nni(adj, edges, a, b, x, y):
    """Apply NNI: remove edges (a,b),(x,y); add (a,x),(b,y) or (a,y),(b,x).
    Returns new adj, edges. Caller decides which of the two."""
    new_adj = [set(s) for s in adj]
    new_adj[a].discard(b); new_adj[b].discard(a)
    new_adj[x].discard(y); new_adj[y].discard(x)
    # option 1: (a,x),(b,y)
    new_adj[a].add(x); new_adj[x].add(a)
    new_adj[b].add(y); new_adj[y].add(b)
    new_edges = [(u, v) for (u, v) in edges if not ((u == a and v == b) or (u == b and v == a) or (u == x and v == y) or (u == y and v == x))]
    new_edges.append((a, x))
    new_edges.append((b, y))
    return new_adj, new_edges


def _rebuild_struct_from_adj(adj, n_int, n, leaf_node_to_id, leaf_nodes):
    """Given adjacency, build children/parent/order/subtree for cost eval.
    Root at node 0."""
    # BFS from root=0
    root = 0
    parent = [-1] * n_int
    children = [None] * n_int
    order = []
    stack = [root]
    parent[root] = -2
    while stack:
        u = stack.pop()
        order.append(u)
        for v in adj[u]:
            if v == parent[u]:
                continue
            parent[v] = u
            if children[u] is None:
                children[u] = []
            children[u].append(v)
            stack.append(v)
    # fix root parent
    parent[root] = -1
    # ensure children None for leaves
    for u in range(n_int):
        if children[u] is not None and len(children[u]) == 0:
            children[u] = None
    # subtree masks
    subtree = [None] * n_int
    for u in leaf_nodes:
        subtree[u] = 1 << leaf_node_to_id[u]
    for u in reversed(order):
        if children[u] is not None:
            mask = 0
            for c in children[u]:
                mask |= subtree[c]
            subtree[u] = mask
    return children, parent, order, subtree


def _parsimony_search(newick, data, n, deadline):
    """Run NNI parsimony refinement. Returns best newick."""
    children, parent, order, labels = _parse(newick)
    n_int = len(order)
    leaf_nodes = [u for u in range(n_int) if children[u] is None]
    leaf_node_to_id = {}
    for pos, u in enumerate(leaf_nodes):
        leaf_node_to_id[u] = pos
    adj, edges, subtree, _, _, allmask = _build_struct(children, parent, order, n, labels)
    best_newick = newick
    best_cost = _fitch_cost_struct(data, order, children, leaf_node_to_id, n)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        moves = _nni_neighbours(adj, edges, n_int)
        # shuffle-ish deterministic order
        for (a, b, x, y) in moves:
            if time.time() >= deadline:
                break
            # try both reconnects
            for opt in range(2):
                new_adj = [set(s) for s in adj]
                new_adj[a].discard(b); new_adj[b].discard(a)
                new_adj[x].discard(y); new_adj[y].discard(x)
                if opt == 0:
                    new_adj[a].add(x); new_adj[x].add(a)
                    new_adj[b].add(y); new_adj[y].add(b)
                    na, nb = (a, x), (b, y)
                else:
                    new_adj[a].add(y); new_adj[y].add(a)
                    new_adj[b].add(x); new_adj[x].add(b)
                    na, nb = (a, y), (b, x)
                new_edges = [(u, v) for (u, v) in edges if not ((u == a and v == b) or (u == b and v == a) or (u == x and v == y) or (u == y and v == x))]
                new_edges.append(na)
                new_edges.append(nb)
                # rebuild structure for cost
                nch, npar, norder, nsub = _rebuild_struct_from_adj(new_adj, n_int, n, leaf_node_to_id, leaf_nodes)
                cost = _fitch_cost_struct(data, norder, nch, leaf_node_to_id, n)
                if cost < best_cost:
                    best_cost = cost
                    best_newick = _to_newick(nch, 0)
                    # commit
                    adj = new_adj
                    edges = new_edges
                    children = nch
                    parent = npar
                    order = norder
                    subtree = nsub
                    improved = True
    return best_newick


def _to_labels(newick, labels):
    """Replace L0..L(n-1) with actual taxon labels in order."""
    # leaves appear in order L0,L1,...; substitute
    import re
    def repl(m):
        idx = int(m.group(1))
        return labels[idx]
    return re.sub(r'L(\d+)', repl, newick)


def reconstruct(sequences):
    start = time.time()
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    seqs = [np.frombuffer(sequences[l].encode('ascii'), dtype=np.uint8) for l in labels]
    L = len(seqs[0])

    # bit-pack: A=0,C=1,G=2,T=3
    code = np.zeros((L, n), dtype=np.int64)
    for j, s in enumerate(seqs):
        c = np.empty(L, dtype=np.int64)
        c[s == 65] = 0  # A
        c[s == 67] = 1  # C
        c[s == 71] = 2  # G
        c[s == 84] = 3  # T
        code[:, j] = c

    # initial NJ tree
    deadline = start + 250.0
    nj = _nj_tree(labels, [s.tobytes() for s in seqs])
    # map taxon labels to L-ids for parsimony
    # build a newick with L-ids
    import re
    nj_l = re.sub(r't\d+', lambda m: "L" + str(int(m.group(1))), nj)

    # parsimony search
    if n >= 4:
        best = _parsimony_search(nj_l, code, n, deadline)
    else:
        best = nj_l
    # convert back to taxon labels
    result = _to_labels(best, labels)
    # ensure trailing semicolon
    if not result.endswith(";"):
        result += ";"
    return result
# EVOLVE-BLOCK-END
