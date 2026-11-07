import time
import random

def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edges, n):
    """Check if no simple 3-edge path has endpoints with equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edges:
        du = adj[u]
        dv = adj[v]
        for s in du:
            if s == v:
                continue
            ds = deg[s]
            for t in dv:
                if t == u or t == s:
                    continue
                if ds == deg[t]:
                    return False
    return True

def _edge_ok(adj, deg, u, v, n):
    """Check if adding edge (u,v) keeps constraint. Assumes (u,v) not present."""
    # new paths created by adding (u,v)
    # middle: s-u-v-t
    for s in adj[u]:
        if s == v:
            continue
        ds = deg[s]
        for t in adj[v]:
            if t == u or t == s:
                continue
            if ds == deg[t]:
                return False
    # first: u-v-x-y  (endpoints u, y)
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[u] == deg[y]:
                return False
    # last: x-y-v-u (endpoints x, u)
    for x in adj[u]:
        if x == v:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[x] == deg[u]:
                return False
    return True

def _remove_edge_ok(adj, deg, u, v, n):
    """Check if removing edge (u,v) keeps constraint. Assumes (u,v) present."""
    # Removing (u,v) can only create new violations if there was a path
    # s-u-v-t where s,t had equal degree AND s-t is an edge (making it a triangle
    # that becomes a 3-path after removal? No, removing an edge doesn't create paths).
    # Actually, removing an edge never creates new simple paths. It only removes
    # existing paths. So the constraint is automatically preserved.
    return True

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit
    n = 15
    rng = random.Random(12345)

    best_edges = None
    best_count = -1

    def evaluate_and_track(edges):
        nonlocal best_edges, best_count
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = [list(e) for e in edges]

    # ---------- Initial construction via greedy add with distinct-degree bias ----------
    best_seed_edges = []
    best_seed_count = -1

    for trial in range(60):
        if time.time() > deadline - 25:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edges = []
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edges.append((u, v))
        if len(edges) > best_seed_count:
            best_seed_count = len(edges)
            best_seed_edges = [list(e) for e in edges]

    # Start local search from best seed
    edges = [tuple(e) for e in best_seed_edges]
    adj = _build_adj(edges, n)
    deg = [len(adj[i]) for i in range(n)]
    evaluate_and_track(edges)

    # ---------- Local search: add, remove, swap ----------
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    present = set()
    for u, v in edges:
        present.add((u, v))

    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
        deg[u] += 1; deg[v] += 1
        edges.append((u, v))
        present.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v); adj[v].discard(u)
        deg[u] -= 1; deg[v] -= 1
        edges.remove((u, v))
        present.discard((u, v))

    improved = True
    max_iter = 20
    iters = 0
    while improved and time.time() < deadline and iters < max_iter:
        improved = False
        iters += 1
        # Try adding missing edges
        for u, v in all_pairs:
            if time.time() > deadline:
                break
            if (u, v) in present:
                continue
            if _edge_ok(adj, deg, u, v, n):
                add_edge(u, v)
                evaluate_and_track(edges)
                improved = True
        # Try swaps: remove e1, add e2
        if time.time() < deadline:
            for (a, b) in list(edges):
                if time.time() > deadline:
                    break
                for (c, d) in all_pairs:
                    if time.time() > deadline:
                        break
                    if (c, d) in present:
                        continue
                    if (a, b) == (c, d):
                        continue
                    # temp remove (a,b)
                    adj[a].discard(b); adj[b].discard(a)
                    deg[a] -= 1; deg[b] -= 1
                    edges.remove((a, b))
                    present.discard((a, b))
                    if _edge_ok(adj, deg, c, d, n):
                        add_edge(c, d)
                        evaluate_and_track(edges)
                        improved = True
                    else:
                        # restore
                        adj[a].add(b); adj[b].add(a)
                        deg[a] += 1; deg[b] += 1
                        edges.append((a, b))
                        present.add((a, b))

    # ---------- Second phase: more aggressive random restarts with local search ----------
    if time.time() < deadline - 10:
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            # Create a perturbation of the current best
            adj = [set() for _ in range(n)]
            deg = [0] * n
            edges = []
            
            # Start from best and randomly remove some edges
            base_edges = [tuple(e) for e in best_edges] if best_edges else []
            rng.shuffle(base_edges)
            num_remove = min(len(base_edges), rng.randint(3, 8))
            for i in range(num_remove):
                e = base_edges[i]
                adj[e[0]].add(e[1]); adj[e[1]].add(e[0])
                deg[e[0]] += 1; deg[e[1]] += 1
                edges.append(e)
            
            # Now greedily add edges
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            rng.shuffle(pairs)
            for u, v in pairs:
                if time.time() > deadline - 2:
                    break
                if (u, v) in adj[u]:
                    continue
                if _edge_ok(adj, deg, u, v, n):
                    adj[u].add(v)
                    adj[v].add(u)
                    deg[u] += 1
                    deg[v] += 1
                    edges.append((u, v))
            
            evaluate_and_track(edges)
            
            # Quick local search on this new solution
            if len(edges) >= best_count:
                present = set()
                for u, v in edges:
                    present.add((u, v))
                
                improved2 = True
                while improved2 and time.time() < deadline - 2:
                    improved2 = False
                    for u, v in all_pairs:
                        if time.time() > deadline - 2:
                            break
                        if (u, v) in present:
                            continue
                        if _edge_ok(adj, deg, u, v, n):
                            adj[u].add(v); adj[v].add(u)
                            deg[u] += 1; deg[v] += 1
                            edges.append((u, v))
                            present.add((u, v))
                            evaluate_and_track(edges)
                            improved2 = True

    result = []
    seen = set()
    for u, v in best_edges:
        u = int(u); v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return result
