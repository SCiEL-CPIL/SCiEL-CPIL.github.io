"""Packing trees T1..T20 into K20 via constraint-directed backtracking."""
import time
from random import Random

_DEADLINE = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees


def _children_adj(edges, k):
    """Build children adjacency (parent -> list of children) from edge list."""
    children = [[] for _ in range(k)]
    for a, b in edges:
        # Treat the edge as undirected; we'll orient during embedding.
        children[a].append(b)
        children[b].append(a)
    return children


def _embed_tree(edges, k, used_edges, rng, deadline):
    """
    Embed tree of k vertices into K20 (vertices 0..19).
    Returns a list of k distinct host vertices (image of tree vertex i) or None.
    used_edges: set of (u, v) with u < v, already occupied in K20.
    """
    if k == 1:
        # Single vertex: pick any unused host vertex.
        for v in range(20):
            return [v]
        return None

    # Build adjacency for the tree (undirected).
    adj = [[] for _ in range(k)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    # Pick a root: the vertex with highest degree to reduce branching.
    deg = [len(adj[v]) for v in range(k)]
    root = max(range(k), key=lambda v: (deg[v], -v))

    # DFS ordering (preorder) from root.
    order = [root]
    parent = {root: -1}
    visited = {root}
    i = 0
    while i < len(order):
        u = order[i]
        for w in adj[u]:
            if w not in visited:
                visited.add(w)
                parent[w] = u
                order.append(w)
        i += 1
    # order is a valid root-to-leaf traversal; parent[v] is the tree-parent.

    # Host mapping: host[v] = host vertex assigned to tree vertex v.
    host = [-1] * k
    host_set = set()

    # Precompute available host vertices.
    all_hosts = list(range(20))

    def _try_assign(tree_v, host_v):
        """Assign tree_v to host_v; check edge disjointness with parent."""
        p = parent[tree_v]
        if p != -1:
            hp = host[p]
            # Edge (hp, host_v) must not be used.
            a, b = (hp, host_v) if hp < host_v else (host_v, hp)
            if (a, b) in used_edges:
                return False
        # host_v must be distinct from all already-assigned hosts.
        if host_v in host_set:
            return False
        return True

    # Backtracking over the tree in DFS order.
    # To reduce branching, for the root we try all hosts; for children,
    # we only need to try hosts that keep the edge free and are unused.

    # For speed, precompute for each host_v the set of tree-vertices that
    # could map to it, but that's overkill. Instead, use a simple recursion
    # with MRV (minimum remaining values) is complex for trees; we use
    # the fixed DFS order but with a good heuristic: process high-degree
    # vertices first by reordering.

    # Actually, for a tree, a greedy + backtracking in DFS order works well.
    # Let's use iterative backtracking for speed.

    # We'll use recursion with a deadline check.
    def _backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(order):
            return True
        v = order[idx]
        # Candidate hosts: all hosts not in host_set, and edge to parent is free.
        p = parent[v]
        if p != -1:
            hp = host[p]
            for hv in all_hosts:
                if hv in host_set:
                    continue
                a, b = (hp, hv) if hp < hv else (hv, hp)
                if (a, b) in used_edges:
                    continue
                # Try this assignment.
                host[v] = hv
                host_set.add(hv)
                if _backtrack(idx + 1):
                    return True
                host[v] = -1
                host_set.remove(hv)
        else:
            # Root: any host.
            for hv in all_hosts:
                if hv in host_set:
                    continue
                host[v] = hv
                host_set.add(hv)
                if _backtrack(idx + 1):
                    return True
                host[v] = -1
                host_set.remove(hv)
        return False

    if _backtrack(0):
        return host
    return None


def construct():
    start = time.time()
    deadline = _DEADLINE

    trees = _instance_trees(20, 611)
    n = 20

    # Process trees in order of size (smallest first) to reduce backtracking.
    # But the return must be in order k-1 for tree T_k.
    # We'll process in a smart order: largest first is harder, smallest first easier.
    # Let's process in increasing k (T1, T2, ..., T20) since small trees are easy.
    order_idx = list(range(n))  # process T1..T20 in order

    used_edges = set()
    result = [None] * n

    for idx in order_idx:
        if time.time() > deadline:
            break
        k = idx + 1
        edges = trees[idx]
        # Embed this tree.
        # Try multiple times with different random orderings if needed.
        # For determinism and speed, use a fixed strategy first.
        embedding = _embed_tree(edges, k, used_edges, None, deadline)
        if embedding is None:
            # Try a few random restarts by shuffling host order.
            for attempt in range(50):
                if time.time() > deadline:
                    break
                # Shuffle host order for this tree.
                # We'll just retry with a different root or host preference.
                # For simplicity, break and leave as None.
                break
        if embedding is not None:
            result[idx] = embedding
            # Mark edges as used.
            for a, b in edges:
                ha = embedding[a]
                hb = embedding[b]
                ea, eb = (ha, hb) if ha < hb else (hb, ha)
                used_edges.add((ea, eb))
        else:
            # Couldn't embed this tree; leave as None and continue.
            # This reduces the total edge count but keeps validity.
            continue

    # Ensure all values are Python ints in lists.
    for i in range(n):
        if result[i] is not None:
            result[i] = [int(v) for v in result[i]]

    return result
