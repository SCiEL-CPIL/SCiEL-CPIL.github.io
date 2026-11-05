import time
import random
from collections import deque

N = 40

def construct():
    deadline = time.time() + 115.0  # safety margin under 120s
    best = None
    best_m = 0

    def make_adj(edges):
        adj = [set() for _ in range(N)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def c6_conflict(adj, u, v):
        # returns True if adding u-v creates a C6; else False
        if v in adj[u]:
            return True
        A = list(adj[u])
        B = list(adj[v])
        for a in A:
            for b in B:
                if a == b:
                    continue
                C = adj[a]
                D = adj[b]
                # C \ {a} and D \ {b}
                # check intersection
                smaller = C if len(C) < len(D) else D
                larger = D if smaller is C else C
                for x in smaller:
                    if x == a or x == b:
                        continue
                    if x in larger:
                        return True
        return False

    def c6_conflict2(adj, u, v, skip_u, skip_v):
        # conflict check ignoring edges incident to skip_u/skip_v (for swaps)
        if v in adj[u]:
            return True
        A = list(adj[u])
        B = list(adj[v])
        for a in A:
            if a == skip_u or a == skip_v:
                continue
            for b in B:
                if b == skip_u or b == skip_v or a == b:
                    continue
                C = adj[a]
                D = adj[b]
                smaller = C if len(C) < len(D) else D
                larger = D if smaller is C else C
                for x in smaller:
                    if x == a or x == b or x == skip_u or x == skip_v:
                        continue
                    if x in larger:
                        return True
        return False

    def greedy_add(adj):
        # add any possible non-edge
        changed = True
        while changed:
            changed = False
            for u in range(N):
                for v in range(u + 1, N):
                    if v not in adj[u] and not c6_conflict(adj, u, v):
                        adj[u].add(v)
                        adj[v].add(u)
                        changed = True
        return adj

    def local_search(adj):
        # swap-based climb: add an edge, remove one that resolves conflicts
        improved = True
        while improved:
            improved = False
            # try all pairs of non-edge (u,v) and edge (x,y)
            for u in range(N):
                for v in range(u + 1, N):
                    if v in adj[u]:
                        continue
                    if c6_conflict(adj, u, v):
                        continue
                    # need to remove some edge to allow u-v
                    # find edges whose removal allows u-v
                    # conflict edges: for each C6 path u-a-...-b-v, one of those edges
                    # simpler: try removing each edge adjacent to u or v
                    cand_edges = []
                    for a in list(adj[u]):
                        cand_edges.append((a, u))
                    for b in list(adj[v]):
                        cand_edges.append((v, b))
                    # dedupe
                    seen = set()
                    for e in cand_edges:
                        key = (min(e), max(e))
                        if key not in seen:
                            seen.add(key)
                    for (x, y) in seen:
                        # simulate remove x-y, add u-v
                        adj[x].discard(y)
                        adj[y].discard(x)
                        if not c6_conflict(adj, u, v):
                            adj[u].add(v)
                            adj[v].add(u)
                            improved = True
                        else:
                            adj[x].add(y)
                            adj[y].add(x)
                    if improved:
                        break
                if improved:
                    break
        return adj

    def edge_count(adj):
        return sum(len(s) for s in adj) // 2

    def adj_to_edges(adj):
        edges = []
        for u in range(N):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges

    def is_c6_free(adj):
        # verify no C6 via BFS
        for s in range(N):
            # BFS from s, track parents to detect simple 6-cycles
            # distance and parent
            dist = [-1] * N
            parent = [-1] * N
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= 5:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        q.append(w)
                    elif w != parent[u] and dist[w] == dist[u] + 1:
                        # potential cycle; check length
                        # walk back from w
                        # we want a cycle of length 6: s..u..w..back to s
                        # Actually check: does there exist a 6-cycle containing s?
                        pass
        # Use a direct approach: for each pair (u,v) non-adjacent, count C4 paths
        for u in range(N):
            for v in range(u + 1, N):
                if v in adj[u]:
                    continue
                A = list(adj[u])
                B = list(adj[v])
                for a in A:
                    for b in B:
                        if a == b:
                            continue
                        C = adj[a]
                        D = adj[b]
                        smaller = C if len(C) < len(D) else D
                        larger = D if smaller is C else C
                        for x in smaller:
                            if x == a or x == b:
                                continue
                            if x in larger:
                                return False
        return True

    # Strategy: multiple random starts
    for trial in range(200):
        if time.time() > deadline:
            break
        # random start: random graph with moderate density
        p = 0.25 + 0.1 * (trial % 5) / 5.0
        adj = [set() for _ in range(N)]
        for u in range(N):
            for v in range(u + 1, N):
                if random.random() < p:
                    adj[u].add(v)
                    adj[v].add(u)
        # fix: remove C6s by deleting edges
        # simple: while not c6 free, remove a random edge in a C6
        # Instead, just do greedy add from sparse, and local search
        adj = greedy_add(adj)
        adj = local_search(adj)
        m = edge_count(adj)
        if is_c6_free(adj) and m > best_m:
            best_m = m
            best = adj_to_edges(adj)
        if best_m >= 105:
            break

    # fallback: seed construction
    if best is None or best_m < 40:
        edges = []
        for k in range(8):
            edges += [[5 * k + a, 5 * k + b] for a in range(5) for b in range(a + 1, 5)]
        return edges

    return best
