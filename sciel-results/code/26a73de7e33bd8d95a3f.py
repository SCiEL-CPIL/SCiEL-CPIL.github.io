import time
import random

def _build_adj(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def check_constraint(edge_set, n):
    """Check if no simple 3-edge path has endpoints with equal degree."""
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    deg = [len(adj[i]) for i in range(n)]
    for u, v in edge_set:
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
    # Precompute degree sets for faster lookup
    deg_sets = [set() for _ in range(n)]
    for i in range(n):
        deg_sets[deg[i]].add(i)
    
    # new paths created by adding (u,v)
    # middle: s-u-v-t
    for s in adj[u]:
        if s == v:
            continue
        ds = deg[s]
        for t in adj[v]:
            if t == u or t == s:
                continue
            if deg[t] == ds:
                return False
    # first: u-v-x-y  (endpoints u, y)
    du = deg[u]
    for x in adj[v]:
        if x == u:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[y] == du:
                return False
    # last: x-y-v-u (endpoints x, u)
    for x in adj[u]:
        if x == v:
            continue
        for y in adj[x]:
            if y == v or y == u:
                continue
            if deg[x] == du:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 85  # safety margin
    n = 15
    rng = random.Random(12345)

    best_edges_set = None
    best_count = -1

    def evaluate_and_track(edge_set):
        nonlocal best_edges_set, best_count
        if len(edge_set) > best_count:
            best_count = len(edge_set)
            best_edges_set = set(edge_set)

    # ---------- Generate Structured Seeds ----------
    seeds = []
    
    # 1. Complete Bipartite Graphs K_{a, b}
    # Valid if a != b (degrees are b and a, distinct for cross pairs, 
    # and same-partition pairs are at dist 2)
    for a in range(1, n):
        b = n - a
        if a == b:
            continue
        seed_set = set()
        for i in range(a):
            for j in range(a, n):
                seed_set.add((i, j))
        if check_constraint(seed_set, n):
            seeds.append(seed_set)
            
    # 2. Random Greedy Seeds
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        adj = [set() for _ in range(n)]
        deg = [0] * n
        edge_set = set()
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(pairs)
        for u, v in pairs:
            if _edge_ok(adj, deg, u, v, n):
                adj[u].add(v)
                adj[v].add(u)
                deg[u] += 1
                deg[v] += 1
                edge_set.add((u, v))
        seeds.append(edge_set)

    # Select best seed
    best_seed_set = max(seeds, key=len)
    
    # Start local search from best seed
    edge_set = set(best_seed_set)
    adj = _build_adj(edge_set, n)
    deg = [len(adj[i]) for i in range(n)]
    evaluate_and_track(edge_set)

    # ---------- Local Search ----------
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def add_edge(u, v):
        adj[u].add(v); adj[v].add(u)
        deg[u] += 1; deg[v] += 1
        edge_set.add((u, v))

    def remove_edge(u, v):
        adj[u].discard(v); adj[v].discard(u)
        deg[u] -= 1; deg[v] -= 1
        edge_set.discard((u, v))

    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Try adding missing edges
        for u, v in all_pairs:
            if time.time() > deadline - 5:
                break
            if (u, v) in edge_set:
                continue
            if _edge_ok(adj, deg, u, v, n):
                add_edge(u, v)
                evaluate_and_track(edge_set)
                improved = True
        # Try swaps: remove e1, add e2
        if time.time() < deadline - 5:
            for (a, b) in list(edge_set):
                if time.time() > deadline - 5:
                    break
                for (c, d) in all_pairs:
                    if (c, d) in edge_set:
                        continue
                    # temp remove (a,b)
                    remove_edge(a, b)
                    if _edge_ok(adj, deg, c, d, n):
                        add_edge(c, d)
                        evaluate_and_track(edge_set)
                        improved = True
                    else:
                        # restore
                        add_edge(a, b)

    # Second pass: try 2-swap (remove 2, add 1) to escape local optima
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            improved2 = False
            for (a, b) in list(edge_set):
                if time.time() > deadline - 5:
                    break
                for (c, d) in list(edge_set):
                    if time.time() > deadline - 5:
                        break
                    if (a, b) == (c, d):
                        continue
                    # Check if removing both is valid
                    remove_edge(a, b)
                    remove_edge(c, d)
                    for (e, f) in all_pairs:
                        if (e, f) in edge_set:
                            continue
                        if _edge_ok(adj, deg, e, f, n):
                            add_edge(e, f)
                            evaluate_and_track(edge_set)
                            improved2 = True
                            break
                    if not improved2:
                        add_edge(c, d)
                        add_edge(a, b)
                    else:
                        break
                if improved2:
                    break
            if not improved2:
                break

    result = []
    seen = set()
    for u, v in best_edges_set:
        u = int(u); v = int(v)
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            result.append([key[0], key[1]])
    return result
