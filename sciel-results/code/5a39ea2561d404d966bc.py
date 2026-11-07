import time
from random import Random
from collections import deque

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


def _embed(tree_edges, k, used_edges, rng, deadline):
    """
    Embed tree with k vertices into K20 using 2D boolean for used edges.
    Returns list of host vertices (length k) or None.
    """
    if k == 1:
        return [0]

    if k == 2:
        for u in range(20):
            for v in range(u + 1, 20):
                if not used_edges[u][v]:
                    return [u, v]
        return None

    # Build tree adjacency
    adj = [[] for _ in range(k)]
    for u, v in tree_edges:
        adj[u].append(v)
        adj[v].append(u)

    # Compute degrees
    deg = [len(adj[v]) for v in range(k)]

    # Try different roots: prioritize high-degree vertices first
    roots = sorted(range(k), key=lambda v: -deg[v])
    rng.shuffle(roots)

    for root in roots:
        if time.time() > deadline:
            return None

        # BFS ordering from root
        visited = [False] * k
        order = []
        parent = [-1] * k
        queue = deque([root])
        visited[root] = True
        while queue:
            cur = queue.popleft()
            order.append(cur)
            for nb in adj[cur]:
                if not visited[nb]:
                    visited[nb] = True
                    parent[nb] = cur
                    queue.append(nb)

        # assignment[tv] = host vertex, or -1 if unassigned
        assign = [-1] * k
        used_host = [False] * 20

        # Precompute available neighbor counts for each host vertex
        # avail[h] = number of unused edges from h to other hosts
        avail = [0] * 20
        for h in range(20):
            for nh in range(20):
                if nh != h and not used_edges[h][nh]:
                    avail[h] += 1

        def dfs(idx):
            nonlocal avail
            if time.time() > deadline:
                return False

            if idx == len(order):
                return True

            tv = order[idx]

            if tv == root:
                # Try all available host vertices, prefer those with more unused edges
                candidates = []
                for h in range(20):
                    if not used_host[h]:
                        candidates.append((avail[h], h))
                candidates.sort(reverse=True)
                for _, h in candidates:
                    assign[tv] = h
                    used_host[h] = True
                    if dfs(idx + 1):
                        return True
                    used_host[h] = False
                    assign[tv] = -1
                return False
            else:
                p = parent[tv]
                ph = assign[p]
                if ph == -1:
                    return False

                # Try all available host vertices adjacent to ph with unused edge
                # Prefer vertices with more unused edges (more flexibility)
                candidates = []
                for h in range(20):
                    if h == ph or used_host[h]:
                        continue
                    if used_edges[ph][h]:
                        continue
                    candidates.append((avail[h], h))
                candidates.sort(reverse=True)
                for _, h in candidates:
                    assign[tv] = h
                    used_host[h] = True
                    # Update avail: for each neighbor of h, decrement
                    for nh in range(20):
                        if nh != h and not used_edges[h][nh]:
                            avail[nh] -= 1
                    avail[h] = 0  # h is now used
                    if dfs(idx + 1):
                        return True
                    # Undo
                    for nh in range(20):
                        if nh != h and not used_edges[h][nh]:
                            avail[nh] += 1
                    avail[h] = sum(1 for nh in range(20) if nh != h and not used_edges[h][nh])
                    used_host[h] = False
                    assign[tv] = -1
                return False

        if dfs(0):
            return [assign[i] for i in range(k)]

    return None


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin (150s limit)

    trees = _instance_trees(20, 611)

    best_result = [None] * 20
    best_value = 0

    for trial in range(2000):
        if time.time() > deadline:
            break

        rng = Random(trial * 7919 + 31337)
        # 2D boolean array for used edges
        used_edges = [[False] * 20 for _ in range(20)]
        result = [None] * 20
        total_edges = 0
        success = True

        # Embed in descending order of tree size
        for k in range(20, 0, -1):
            if time.time() > deadline:
                success = False
                break

            tree_edges = trees[k - 1]

            if not tree_edges:
                # T1: single vertex, pick any
                result[k - 1] = [rng.randrange(20)]
                continue

            embedding = _embed(tree_edges, k, used_edges, rng, deadline)

            if embedding is None:
                success = False
                break

            result[k - 1] = embedding
            # Mark edges as used
            for u, v in tree_edges:
                hu, hv = embedding[u], embedding[v]
                used_edges[hu][hv] = True
                used_edges[hv][hu] = True
            total_edges += len(tree_edges)

        if success and total_edges > best_value:
            best_value = total_edges
            best_result = result[:]
            if best_value == 190:
                break

    # Ensure correct types
    final = []
    for entry in best_result:
        if entry is None:
            final.append(None)
        else:
            final.append([int(x) for x in entry])

    return final
