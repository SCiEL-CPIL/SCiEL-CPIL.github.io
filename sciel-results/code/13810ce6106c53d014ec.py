import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n = 40
    total_pairs = n * (n - 1) // 2  # 780

    # Helper: create adjacency set from edge list
    def make_adj(edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    # Check if adding edge (u,v) creates a C6
    # C6 containing (u,v): need path of length 4 from u to v not using edge (u,v)
    # i.e., u - a - b - c - d - v where a!=v, d!=u, a!=c, etc.
    def would_create_c6(adj, u, v):
        # BFS from u up to depth 2, BFS from v up to depth 2, check if they meet at distance 2 from both
        # Actually need path u-a-b-c-d-v of length 5 (6 vertices)
        # u connects to a, a to b, b to c, c to d, d to v
        # So: for each a in adj[u], for each b in adj[a] (b != u), for each c in adj[b] (c != a, c != u),
        #     check if any d in adj[c] (d != b, d != u, d != v) is in adj[v]
        # But also need the path to be simple: u,a,b,c,d,v all distinct
        neighbors_u = adj[u]
        neighbors_v = adj[v]
        for a in neighbors_u:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v:
                    continue
                for c in adj[b]:
                    if c == u or c == v or c == a:
                        continue
                    # Now check if any neighbor d of c (d != b, d != u, d != v) is in neighbors_v
                    for d in adj[c]:
                        if d == b or d == u or d == v:
                            continue
                        if d in neighbors_v:
                            return True
        return False

    # Verify no C6 in the entire graph (for final validation)
    def has_c6(adj):
        # For each pair (u,v) NOT adjacent, check if there's a path of length 5
        # Actually easier: for each edge (u,v), check if removing it still leaves a path of length 5
        # Or: for each pair u,v, check if shortest path >= 6 or no path... 
        # Simpler: for every edge (u,v), check would_create_c6 on graph without that edge
        # But that's expensive. Let's do: for each non-edge (u,v), check if there's a path of length exactly 5
        # Actually the cleanest: iterate all edges, temporarily remove, check would_create_c6
        # Too slow. Instead: for each pair (u,v) that IS an edge, check if there exists a path u-a-b-c-d-v
        # where all intermediate are distinct from u,v and from each other.
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                # Check path u-a-b-c-d-v with a,b,c,d distinct, not u or v
                neighbors_u = adj[u]
                neighbors_v = adj[v]
                for a in neighbors_u:
                    if a == v:
                        continue
                    for b in adj[a]:
                        if b == u or b == v or b == a:
                            continue
                        for c in adj[b]:
                            if c == u or c == v or c == a or c == b:
                                continue
                            for d in adj[c]:
                                if d == u or d == v or d == a or d == b or d == c:
                                    continue
                                if d in neighbors_v:
                                    return True
        return False

    # Start with 8 disjoint K5s
    edges = []
    for k in range(8):
        for a in range(5):
            for b in range(a + 1, 5):
                edges.append([5 * k + a, 5 * k + b])

    adj = make_adj(edges)
    edge_set = set((min(u, v), max(u, v)) for u, v in edges)
    best_m = len(edges)
    best_edges = list(edges)

    # Random initialization with varying densities to get diverse starts
    # Also try to build up from sparse random graphs
    def random_init(p):
        e = []
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    e.append([i, j])
        # Remove C6s by edge removal
        adj_r = make_adj(e)
        changed = True
        while changed:
            changed = False
            for idx, (u, v) in enumerate(list(e)):
                # Temporarily remove and check
                adj_r[u].discard(v)
                adj_r[v].discard(u)
                if not has_c6(adj_r):
                    e.pop(idx)
                    changed = True
                else:
                    adj_r[u].add(v)
                    adj_r[v].add(u)
        return e, make_adj(e)

    # Try multiple random starts and the K5 baseline
    candidates = []

    # K5 baseline
    candidates.append(list(edges))

    # Random inits with various densities
    for attempt in range(6):
        if time.time() > deadline:
            break
        p = random.uniform(0.05, 0.18)
        re, radj = random_init(p)
        if len(re) > 0:
            candidates.append(re)

    # Also try a "random + greedy add" approach
    for attempt in range(4):
        if time.time() > deadline:
            break
        e = []
        radj = [set() for _ in range(n)]
        # Random sparse start
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.06:
                    e.append([i, j])
                    radj[i].add(j)
                    radj[j].add(i)
        # Greedy add: try adding random non-edges
        random.shuffle(list(range(n)))
        for _ in range(200):
            added = False
            for u in range(n):
                for v in range(u + 1, n):
                    if v in radj[u]:
                        continue
                    if not would_create_c6(radj, u, v):
                        e.append([u, v])
                        radj[u].add(v)
                        radj[v].add(u)
                        added = True
                        break
                if added:
                    break
            if not added:
                break
        candidates.append(e)

    # Local search on each candidate
    def local_search(edge_list, adj_init, max_time):
        m = len(edge_list)
        adj_l = [set(s) for s in adj_init]
        es = set((min(u, v), max(u, v)) for u, v in edge_list)
        best_local = m
        best_local_edges = list(edge_list)
        elapsed0 = time.time()

        # 1-for-0: add edges
        improved = True
        while improved and time.time() - elapsed0 < max_time:
            improved = False
            random.shuffle(list(range(n)))
            for u in range(n):
                random.shuffle(list(adj_l[u]))  # not needed, but shuffle neighbors
                for v in range(n):
                    if u == v or (u, v) in es or (v, u) in es:
                        continue
                    if not would_create_c6(adj_l, u, v):
                        edge_list.append([u, v])
                        adj_l[u].add(v)
                        adj_l[v].add(u)
                        es.add((min(u, v), max(u, v)))
                        m += 1
                        improved = True
                        if m > best_local:
                            best_local = m
                            best_local_edges = list(edge_list)
                        break

        # 1-for-1 swaps: remove one edge, add another (same count) to escape local optima
        swap_count = 0
        while time.time() - elapsed0 < max_time and swap_count < 50:
            swap_count += 1
            # Pick a random edge to remove
            if not edge_list:
                break
            idx = random.randrange(len(edge_list))
            u0, v0 = edge_list[idx]
            # Remove it
            adj_l[u0].discard(v0)
            adj_l[v0].discard(u0)
            es.discard((min(u0, v0), max(u0, v0)))
            edge_list.pop(idx)
            # Try to add a random non-edge
            added = False
            for _ in range(30):
                u = random.randrange(n)
                v = random.randrange(n)
                if u == v:
                    continue
                key = (min(u, v), max(u, v))
                if key in es:
                    continue
                if not would_create_c6(adj_l, u, v):
                    edge_list.append([u, v])
                    adj_l[u].add(v)
                    adj_l[v].add(u)
                    es.add(key)
                    added = True
                    break
            if not added:
                # Revert: add back the removed edge
                edge_list.append([u0, v0])
                adj_l[u0].add(v0)
                adj_l[v0].add(u0)
                es.add((min(u0, v0), max(u0, v0)))

        # 1-for-2: remove one edge, add two edges (net +1)
        improved2 = True
        while improved2 and time.time() - elapsed0 < max_time:
            improved2 = False
            for idx in range(len(edge_list)):
                u0, v0 = edge_list[idx]
                # Remove edge (u0,v0)
                adj_l[u0].discard(v0)
                adj_l[v0].discard(u0)
                es.discard((min(u0, v0), max(u0, v0)))
                # Try to add two edges
                added1 = -1
                added2 = -1
                # First add
                for _ in range(20):
                    u = random.randrange(n)
                    v = random.randrange(n)
                    if u == v:
                        continue
                    key = (min(u, v), max(u, v))
                    if key in es:
                        continue
                    if not would_create_c6(adj_l, u, v):
                        added1 = (u, v)
                        adj_l[u].add(v)
                        adj_l[v].add(u)
                        es.add(key)
                        break
                if added1 < 0:
                    # Revert removal
                    edge_list.insert(idx, [u0, v0])
                    adj_l[u0].add(v0)
                    adj_l[v0].add(u0)
                    es.add((min(u0, v0), max(u0, v0)))
                    continue
                # Second add
                for _ in range(20):
                    u = random.randrange(n)
                    v = random.randrange(n)
                    if u == v:
                        continue
                    key = (min(u, v), max(u, v))
                    if key in es:
                        continue
                    if not would_create_c6(adj_l, u, v):
                        added2 = (u, v)
                        adj_l[u].add(v)
                        adj_l[v].add(u)
                        es.add(key)
                        break
                if added2 < 0:
                    # Revert: remove first added and the original
                    if added1 >= 0:
                        u1, v1 = added1
                        adj_l[u1].discard(v1)
                        adj_l[v1].discard(u1)
                        es.discard((min(u1, v1), max(u1, v1)))
                    # Re-add original
                    edge_list.insert(idx, [u0, v0])
                    adj_l[u0].add(v0)
                    adj_l[v0].add(u0)
                    es.add((min(u0, v0), max(u0, v0)))
                    continue
                # Success: replace [u0,v0] with [added1, added2]
                edge_list[idx] = [added1[0], added1[1]]
                edge_list.append([added2[0], added2[1]])
                m += 1
                improved2 = True
                if m > best_local:
                    best_local = m
                    best_local_edges = list(edge_list)
                break

        return best_local_edges, best_local

    # Run local search on each candidate
    global_best = best_edges
    global_best_m = best_m

    for cand in candidates:
        if time.time() > deadline:
            break
        cadj = make_adj(cand)
        result_edges, result_m = local_search(cand, cadj, max_time=15)
        if result_m > global_best_m:
            global_best_m = result_m
            global_best = result_edges

    # Final validation
    final_adj = make_adj(global_best)
    if has_c6(final_adj):
        # Fall back to safe K5 baseline
        safe = []
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    safe.append([5 * k + a, 5 * k + b])
        return safe

    return global_best
