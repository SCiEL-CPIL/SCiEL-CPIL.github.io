import time
from random import Random
from collections import deque

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2: trees.append([] if k == 1 else [(0, 1)]); continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code: degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x)); degree[leaf] -= 1; degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w)); trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110

    trees = instance_trees(20, 611)
    n = 20

    tree_sizes = [len(t) + 1 if t else 1 for t in trees]

    tree_adj = []
    for i, edges in enumerate(trees):
        k = tree_sizes[i]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        tree_adj.append(adj)

    tree_deg = []
    for i, adj in enumerate(tree_adj):
        tree_deg.append([len(adj[v]) for v in range(len(adj))])

    order = list(range(20))
    order.sort(key=lambda i: -tree_sizes[i])

    def try_solve(ord, time_limit):
        used_edges = set()
        assignments = {}

        def bt(idx):
            if time.time() > time_limit:
                return False
            if idx == len(ord):
                return True

            ti = ord[idx]
            k = tree_sizes[ti]
            adj = tree_adj[ti]
            tedges = trees[ti]
            deg = tree_deg[ti]

            if k == 1:
                assignments[ti] = [0]
                if bt(idx + 1):
                    return True
                del assignments[ti]
                return False

            if k == 2:
                u, v = tedges[0]
                for a in range(n):
                    for b in range(a + 1, n):
                        ek = (a, b)
                        if ek in used_edges:
                            continue
                        assignments[ti] = [a, b]
                        used_edges.add(ek)
                        if bt(idx + 1):
                            return True
                        used_edges.discard(ek)
                        del assignments[ti]
                return False

            parent = [-1] * k
            children = [[] for _ in range(k)]
            visited = [False] * k
            stack = [0]
            visited[0] = True
            while stack:
                node = stack.pop()
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        parent[nb] = node
                        children[node].append(nb)
                        stack.append(nb)

            topo = [0]
            queue = deque([0])
            while queue:
                node = queue.popleft()
                sc = sorted(children[node], key=lambda x: -deg[x])
                for c in sc:
                    topo.append(c)
                    queue.append(c)

            def btv(pos, hmap, lused):
                if time.time() > time_limit:
                    return False
                if pos == len(topo):
                    for u, v in tedges:
                        hu, hv = hmap[u], hmap[v]
                        ek = (min(hu, hv), max(hu, hv))
                        if ek in used_edges:
                            return False
                    assignments[ti] = [hmap[i] for i in range(k)]
                    for u, v in tedges:
                        hu, hv = hmap[u], hmap[v]
                        ek = (min(hu, hv), max(hu, hv))
                        used_edges.add(ek)
                    if bt(idx + 1):
                        return True
                    for u, v in tedges:
                        hu, hv = hmap[u], hmap[v]
                        ek = (min(hu, hv), max(hu, hv))
                        used_edges.discard(ek)
                    del assignments[ti]
                    return False

                tv = topo[pos]
                if pos == 0:
                    candidates = [0]
                else:
                    candidates = list(range(n))

                for hv in candidates:
                    ok = True
                    for nb in adj[tv]:
                        if nb in hmap:
                            ek = (min(hv, hmap[nb]), max(hv, hmap[nb]))
                            if ek in used_edges or ek in lused:
                                ok = False
                                break
                    if not ok:
                        continue
                    hmap[tv] = hv
                    new_eks = []
                    for nb in adj[tv]:
                        if nb in hmap:
                            ek = (min(hv, hmap[nb]), max(hv, hmap[nb]))
                            new_eks.append(ek)
                            lused.add(ek)
                    if btv(pos + 1, hmap, lused):
                        return True
                    for ek in new_eks:
                        lused.discard(ek)
                    del hmap[tv]
                return False

            if btv(0, {}, set()):
                return True
            return False

        if bt(0):
            return assignments
        return None

    result = try_solve(order, deadline - 5)

    if result:
        out = [None] * 20
        for i, hv in result.items():
            out[i] = [int(x) for x in hv]
        return out

    return [None] * 19 + [list(range(20))]
