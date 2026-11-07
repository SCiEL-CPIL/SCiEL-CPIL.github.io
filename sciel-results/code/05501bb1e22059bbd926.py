import time
from random import Random
from itertools import permutations

def instance_trees(n=20, seed=611):
    rng = Random(seed); trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)]); continue
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
    deadline = time.time() + 110.0
    trees = instance_trees(20, 611)
    N = 20
    # All K20 edges
    all_edges = set()
    for a in range(N):
        for b in range(a + 1, N):
            all_edges.add((a, b))
    used = set()
    result = [None] * 20

    def edge_key(u, v):
        return (u, v) if u < v else (v, u)

    def try_place(k_idx, edges, used_set, rng):
        """Find injective map of k vertices to 0..19 so all edges land on unused K20 edges."""
        k = len(edges) + (0 if k == 0 else 0)
        if k == 1:
            # T1: single vertex, any host
            return [0]
        if k == 2:
            # T2: single edge (0,1)
            for a in range(N):
                for b in range(a + 1, N):
                    if (a, b) not in used_set:
                        return [a, b]
            return None
        # Build adjacency of the abstract tree
        adj = [[] for _ in range(k)]
        for (u, v) in edges:
            adj[u].append(v); adj[v].append(u)
        # Root the tree at 0, order vertices by BFS for backtracking
        from collections import deque
        parent = [-1] * k
        order = [0]
        parent[0] = -2
        q = deque([0])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if parent[y] == -1:
                    parent[y] = x
                    q.append(y)
                    order.append(y)
        # order[0]=root, rest in BFS order; each vertex (except root) has a parent in the tree
        # Backtracking: assign host vertex to each abstract vertex in order
        assignment = [-1] * k
        used_hosts = [False] * N
        # For each abstract vertex, the constraint is: for each child c, (host[v], host[c]) unused
        # and distinct hosts.
        # We also must ensure edge (v, parent[v]) unused when v assigned.

        def dfs(idx):
            if time.time() > deadline:
                return False
            if idx == k:
                return True
            v = order[idx]
            # candidate hosts: unused, and if v has a parent, edge to parent's host must be free
            pv = parent[v]
            for h in range(N):
                if used_hosts[h]:
                    continue
                if pv >= 0:
                    ph = assignment[pv]
                    e = edge_key(h, ph)
                    if e in used_set:
                        continue
                # tentatively assign
                assignment[v] = h
                used_hosts[h] = True
                if dfs(idx + 1):
                    return True
                used_hosts[h] = False
            assignment[v] = -1
            return False

        if dfs(0):
            return [assignment[i] for i in range(k)]
        return None

    # Process from largest to smallest
    order_idx = list(range(20))
    # sort by size descending (k = idx+1)
    order_idx.sort(key=lambda i: -(i + 1))

    # Use randomized restarts with different host shuffles to escape bad local choices
    best = None
    rng = Random(12345)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        used = set()
        result = [None] * 20
        ok = True
        for i in order_idx:
            k = i + 1
            edges = trees[i]
            # randomize host order for this attempt
            hosts = list(range(N))
            rng.shuffle(hosts)
            # We need a mapping; do a randomized backtracking with host order = hosts
            # Re-implement try_place with shuffled host order
            if k == 1:
                result[i] = [hosts[0]]
                continue
            if k == 2:
                placed = False
                for a in hosts:
                    for b in hosts:
                        if a == b:
                            continue
                        e = edge_key(a, b)
                        if e not in used:
                            result[i] = [a, b]
                            used.add(e)
                            placed = True
                            break
                    if placed:
                        break
                if not placed:
                    ok = False
                    break
                continue
            # general k via backtracking with host priority = hosts
            kk = k
            adj = [[] for _ in range(kk)]
            for (u, v) in edges:
                adj[u].append(v); adj[v].append(u)
            from collections import deque
            parent = [-1] * kk
            order = [0]
            parent[0] = -2
            q = deque([0])
            while q:
                x = q.popleft()
                for y in adj[x]:
                    if parent[y] == -1:
                        parent[y] = x
                        q.append(y)
                        order.append(y)
            assignment = [-1] * kk
            used_hosts = [False] * N
            host_rank = {h: r for r, h in enumerate(hosts)}
            # try hosts in shuffled order
            host_list = hosts[:]

            def dfs(idx):
                if time.time() > deadline:
                    return False
                if idx == kk:
                    return True
                v = order[idx]
                pv = parent[v]
                for h in host_list:
                    if used_hosts[h]:
                        continue
                    if pv >= 0:
                        ph = assignment[pv]
                        e = edge_key(h, ph)
                        if e in used:
                            continue
                    assignment[v] = h
                    used_hosts[h] = True
                    if dfs(idx + 1):
                        return True
                    used_hosts[h] = False
                assignment[v] = -1
                return False

            if dfs(0):
                result[i] = [assignment[j] for j in range(kk)]
                for (u, v) in edges:
                    used.add(edge_key(assignment[u], assignment[v]))
            else:
                ok = False
                break
        if ok:
            # verify edge-disjointness and distinctness
            seen = set()
            for i in range(20):
                r = result[i]
                if r is None:
                    if i == 0:
                        continue
                    ok = False
                    break
                if len(set(r)) != len(r):
                    ok = False
                    break
                for x in r:
                    if not (0 <= x <= 19):
                        ok = False
                        break
                for (u, v) in trees[i]:
                    e = edge_key(r[u], r[v])
                    if e in seen:
                        ok = False
                        break
                    seen.add(e)
                if not ok:
                    break
            if ok:
                # count edges
                total = sum(len(trees[i]) for i in range(20))
                return [ (None if r is None else list(r)) for r in result ]
        if attempt % 50 == 0 and best is None:
            pass
    # If we ran out of time, return the seed (T20 identity) as fallback
    return [None] * 19 + [list(range(20))]
