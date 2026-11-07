import time
from itertools import combinations
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = 24
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((min(u, v), max(u, v)))

    # Candidate edges: only between even and odd to guarantee triangle-free
    A = list(range(0, n, 2))  # evens
    B = list(range(1, n, 2))  # odds
    candidates = []
    for u in A:
        for v in B:
            if (min(u, v), max(u, v)) not in base_edges:
                candidates.append((u, v))

    deadline = time.time() + 85.0

    def _build_adj(added):
        adj = [set() for _ in range(n)]
        for u, v in P['edges']:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in added:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def _valid(added):
        # Triangle-free is guaranteed by bipartite structure (all added edges cross even/odd,
        # and path edges also cross even/odd, so graph stays bipartite).
        # Just check diameter <= 4 and connectivity.
        adj = _build_adj(added)
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= 4:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        q.append(v)
            for d in dist:
                if d == -1 or d > 4:
                    return False
        return True

    # --- Phase 1: Exhaustive search for small k (1..4) ---
    for k in range(1, 5):
        if time.time() > deadline:
            break
        if k <= 3:
            # C(119,3) ~ 275k, feasible
            for combo in combinations(candidates, k):
                if time.time() > deadline:
                    break
                if _valid(combo):
                    return [[int(u), int(v)] for u, v in combo]
        elif k == 4:
            # C(119,4) ~ 8M -- too many for exhaustive; use sampling
            import random
            rng = random.Random(12345)
            max_iters = 200000
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                combo = rng.sample(candidates, k)
                if _valid(combo):
                    return [[int(u), int(v)] for u, v in combo]

    # --- Phase 2: Structured double-star initialization + greedy reduction ---
    # 11 is odd, connect to evens; 12 is even, connect to odds
    init_edges = []
    for v in A:
        if (min(11, v), max(11, v)) not in base_edges:
            init_edges.append((11, v))
    for v in B:
        if (min(12, v), max(12, v)) not in base_edges:
            init_edges.append((12, v))

    if _valid(init_edges):
        current = list(init_edges)
    else:
        current = []
        # Greedy addition fallback
        while time.time() < deadline:
            if _valid(current):
                break
            adj = _build_adj(current)
            # Find farthest pair
            far_dist = -1
            far_pair = None
            for s in range(n):
                dist = [-1] * n
                dist[s] = 0
                q = deque([s])
                while q:
                    u = q.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            q.append(v)
                for t in range(n):
                    if dist[t] > far_dist:
                        far_dist = dist[t]
                        far_pair = (s, t)
            if far_dist <= 4:
                break
            a, b = far_pair
            # Try to add edge that helps
            added_set = set((min(u,v), max(u,v)) for u,v in current)
            best_edge = None
            best_d = far_dist
            for e in candidates:
                if e in added_set:
                    continue
                trial = current + [e]
                adj2 = _build_adj(trial)
                # Quick diameter check
                max_d = 0
                ok = True
                for s in range(n):
                    dist = [-1] * n
                    dist[s] = 0
                    q = deque([s])
                    while q:
                        u = q.popleft()
                        if dist[u] >= 4:
                            continue
                        for v in adj2[u]:
                            if dist[v] == -1:
                                dist[v] = dist[u] + 1
                                q.append(v)
                    for d in dist:
                        if d == -1:
                            ok = False
                            break
                        if d > max_d:
                            max_d = d
                    if not ok:
                        break
                if ok and max_d < best_d:
                    best_d = max_d
                    best_edge = e
            if best_edge is None:
                # Add edge between far pair if valid
                e = (min(a, b), max(a, b))
                if e not in added_set and e in set(candidates):
                    current.append(e)
                else:
                    # Find any candidate edge incident to a or b
                    found = False
                    for e in candidates:
                        if e in added_set:
                            continue
                        if a in e or b in e:
                            current.append(e)
                            found = True
                            break
                    if not found:
                        break
            else:
                current.append(best_edge)

    # --- Phase 3: Greedy reduction ---
    if _valid(current):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                trial = current[:i] + current[i+1:]
                if _valid(trial):
                    current = trial
                    improved = True
                    break

    # --- Phase 4: Local search (1-swap) for further reduction ---
    if time.time() < deadline and _valid(current):
        cur_set = set((min(u,v), max(u,v)) for u,v in current)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                e_out = current[i]
                for e_in in candidates:
                    if e_in in cur_set:
                        continue
                    trial = [current[j] for j in range(len(current)) if j != i] + [e_in]
                    if _valid(trial):
                        current = trial
                        cur_set = set((min(u,v), max(u,v)) for u,v in current)
                        improved = True
                        break
                if improved:
                    break

    if not _valid(current):
        return []

    return [[int(u), int(v)] for u, v in current]
