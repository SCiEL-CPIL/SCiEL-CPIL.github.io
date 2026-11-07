"""Rewritten program: pack trees T1..T20 into K20 with edge-disjoint embeddings."""
from random import Random

_N = 20
_DEADLINE = 125.0  # safety margin under the 150 s kill
_EPOCH = 0.0       # set at construct() start


def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
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
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees


def _embed_tree(edges, deg, used, order):
    """Backtracking injection of a tree into the residual graph.

    Returns a list of host vertices (image of each tree vertex) or None.
    """
    k = len(deg)
    if k == 1:
        return [0]
    used_set = set(used)
    host = [-1] * k
    stack = [(order[0], -1)]
    host[order[0]] = 0

    while stack:
        tv, pv = stack.pop()
        if tv == pv:
            continue
        placed = False
        for h in range(_N):
            if host[tv] == h:
                continue
            if h in used_set:
                continue
            if deg[h] < 1:
                continue
            host[tv] = h
            deg[h] -= 1
            used_set.add(h)
            stack.append((tv, h))
            placed = True
            break
        if not placed:
            # undo
            if host[tv] != -1:
                used_set.discard(host[tv])
                deg[host[tv]] += 1
            host[tv] = -1
            if not stack:
                return None
            stack.pop()
            # restore parent's bookkeeping (it was recorded before we descended)
            # (parent's host entry already set; we just need to retry its other children)
            # continue loop
        # after a successful placement, loop continues
    # all vertices placed
    return host


def construct():
    global _EPOCH
    import time
    t0 = time.time()
    _EPOCH = t0
    trees = instance_trees()

    # order: embed small (easy, few edges) first, large last
    order = list(range(len(trees)))
    order.sort(key=lambda i: len(trees[i]))

    best_result = [None] * len(trees)
    best_edges = 0
    # baseline: only T20 by identity
    best_result[len(trees) - 1] = list(range(_N))
    best_edges = len(trees[len(trees) - 1])

    def score(result):
        return sum(len(trees[i]) for i in range(len(trees)) if result[i] is not None)

    for attempt in range(200):
        if time.time() - t0 > _DEADLINE - 5.0:
            break
        rng = Random(100000 + attempt)
        deg = [_N - 1] * _N
        used = set()
        result = [None] * len(trees)
        # try each tree in the chosen order; skip (leave None) on failure
        for i in order:
            if time.time() - t0 > _DEADLINE - 5.0:
                break
            edges = trees[i]
            k = len(edges) + 1
            if k == 1:
                result[i] = [0]
                continue
            if k == 2:
                # place edge
                placed = False
                for a in range(_N):
                    if a in used or deg[a] < 1:
                        continue
                    for b in range(_N):
                        if b in used or b == a or deg[b] < 1:
                            continue
                        result[i] = [a, b]
                        used.add(a); used.add(b)
                        deg[a] -= 1; deg[b] -= 1
                        placed = True
                        break
                    if placed:
                        break
                continue
            # build degree + parent/children for the tree
            tdeg = [0] * k
            for (a, b) in edges:
                tdeg[a] += 1
                tdeg[b] += 1
            # pick a root and order vertices so parents come before children (BFS/DFS)
            root = 0
            # adjacency
            adj = [[] for _ in range(k)]
            for (a, b) in edges:
                adj[a].append(b)
                adj[b].append(a)
            # order by distance from root (so we place root, then children)
            order_v = [root]
            seen = {root}
            for v in order_v:
                for w in adj[v]:
                    if w not in seen:
                        seen.add(w)
                        order_v.append(w)
            # reverse so that when we pop, we place leaves? We place in order_v order
            # (root first). For backtracking we need parent placed before child.
            # We'll place in order_v order using a stack of (vertex, parent).
            # Simpler: iterative DFS that places root then recurses.
            # Use recursion via explicit stack of (tv, pv) where pv is tree parent.
            # We need to know for each vertex its parent in the rooted tree.
            parent = [-1] * k
            stack = [root]
            while stack:
                v = stack.pop()
                for w in adj[v]:
                    if w == parent[v]:
                        continue
                    parent[w] = v
                    stack.append(w)
            # children list
            children = [[] for _ in range(k)]
            for w in range(1, k):
                children[parent[w]].append(w)
            # place root then depth-first
            host = [-1] * k
            used_set = set(used)
            # choose root host
            root_placed = False
            for h in range(_N):
                if h in used_set or deg[h] < 1:
                    continue
                host[root] = h
                used_set.add(h)
                deg[h] -= 1
                root_placed = True
                break
            if not root_placed:
                continue
            # DFS with backtracking
            # stack entries: (vertex, parent_vertex, child_index)
            dfs = [(root, -1, 0)]
            ok = True
            while dfs:
                tv, pv, ci = dfs[-1]
                if ci < len(children[tv]):
                    ch = children[tv][ci]
                    # try to place ch
                    placed = False
                    for h in range(_N):
                        if h in used_set or deg[h] < 1:
                            continue
                        host[ch] = h
                        used_set.add(h)
                        deg[h] -= 1
                        dfs[-1] = (tv, pv, ci + 1)
                        dfs.append((ch, tv, 0))
                        placed = True
                        break
                    if not placed:
                        # no host for this child: backtrack
                        dfs.pop()
                        if not dfs:
                            ok = False
                            break
                        # undo parent's child index increment? parent already advanced
                        # we just retry next child of parent (loop continues)
                else:
                    # done with this vertex, pop
                    dfs.pop()
                    if not dfs:
                        break
            if ok and all(h != -1 for h in host):
                # success
                for h in host:
                    used.add(h)
                    deg[h] -= 1
                result[i] = list(host)
            else:
                # failed; leave None (do not consume any vertices)
                # deg/used were modified during the attempt; need to revert.
                # Easiest: we mutated deg/used in place during DFS. Revert by
                # re-adding all hosts that were placed.
                for h in host:
                    if h != -1:
                        used.discard(h)
                        deg[h] += 1
        s = score(result)
        if s > best_edges:
            best_edges = s
            best_result = list(result)

    # ensure types: list of int or None
    out = []
    for r in best_result:
        if r is None:
            out.append(None)
        else:
            out.append([int(x) for x in r])
    return out
