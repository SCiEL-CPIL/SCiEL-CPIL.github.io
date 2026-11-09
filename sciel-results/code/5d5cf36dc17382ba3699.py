import time
import sys
from random import Random

def _generate_trees(n=20, seed=611):
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

def construct():
    deadline = time.time() + 145
    trees = _generate_trees(20, 611)
    n = 20
    all_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            all_edges.add((i, j))

    order = list(range(19, -1, -1))
    result = [None] * 20
    available = set(all_edges)

    def can_embed(tree_idx, available_edges, rng):
        k = tree_idx + 1
        if k == 1:
            for v in range(n):
                return [v]
            return None
        if k == 2:
            for (a, b) in available_edges:
                return [a, b]
            return None

        tree_edges = trees[tree_idx]
        adj = [[] for _ in range(k)]
        for (u, v) in tree_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Build available adjacency
        avail_adj = [set() for _ in range(n)]
        for (a, b) in available_edges:
            avail_adj[a].add(b)
            avail_adj[b].add(a)

        # Find all leaves
        leaves = [v for v in range(k) if len(adj[v]) == 1]
        if not leaves:
            leaves = [0]

        def try_root(leaf):
            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            visited[leaf] = True
            stack = [leaf]
            order_list = [leaf]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        parent[v] = u
                        children[u].append(v)
                        stack.append(v)
                        order_list.append(v)

            host = [-1] * k
            used = set()

            def backtrack(idx):
                if time.time() > deadline:
                    return None
                if idx == len(order_list):
                    for (u, v) in tree_edges:
                        hu, hv = host[u], host[v]
                        e = (min(hu, hv), max(hu, hv))
                        if e not in available_edges:
                            return None
                    return list(host)

                v = order_list[idx]
                if parent[v] == -1:
                    candidates = list(range(n))
                    rng.shuffle(candidates)
                else:
                    pv = parent[v]
                    if host[pv] == -1:
                        return None
                    candidates = list(avail_adj[host[pv]])
                    rng.shuffle(candidates)

                for h in candidates:
                    if h in used:
                        continue
                    host[v] = h
                    used.add(h)
                    res = backtrack(idx + 1)
                    if res is not None:
                        return res
                    used.discard(h)
                    host[v] = -1
                return None

            return backtrack(0)

        # Try different roots and random seeds
        rng.shuffle(leaves)
        for leaf in leaves:
            emb = try_root(leaf)
            if emb is not None:
                return emb
        return None

    success = True
    for attempt in range(50):
        if time.time() > deadline:
            break
        rng = Random(611 + attempt * 1000)
        result = [None] * 20
        available = set(all_edges)
        success = True

        for t_idx in order:
            if time.time() > deadline:
                success = False
                break
            k = t_idx + 1
            if k == 1:
                for v in range(n):
                    result[t_idx] = [v]
                    break
                continue
            elif k == 2:
                found = False
                for (a, b) in available:
                    result[t_idx] = [a, b]
                    available.discard((a, b))
                    found = True
                    break
                if not found:
                    success = False
                    break
                continue

            emb = can_embed(t_idx, available, rng)
            if emb is None:
                success = False
                break
            result[t_idx] = emb
            for (u, v) in trees[t_idx]:
                hu, hv = emb[u], emb[v]
                e = (min(hu, hv), max(hu, hv))
                if e in available:
                    available.discard(e)
                else:
                    success = False
                    break
            if not success:
                break

        if success:
            break

    for i in range(20):
        if result[i] is not None:
            result[i] = [int(x) for x in result[i]]

    return result
