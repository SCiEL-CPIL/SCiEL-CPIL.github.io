import time
import random

def construct():
    """
    Search for a C5-free graph with alpha < 9, maximizing n (target 35).
    Strategy:
      1. Build a C5-free base graph via random G(n, p) with small p.
      2. Repair alpha: find large independent sets and add edges within them
         (edge additions that do not create a C5).
      3. Exact verification of C5-freeness and alpha < 9.
    Falls back to 8 disjoint K4s (n=32) if no valid graph is found in time.
    """
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s wall-clock kill

    def time_left():
        return DEADLINE - (time.time() - start_time)

    # ---------- Exact C5 detection ----------
    def is_c5_free(n, edge_set):
        """Return True iff the simple graph has no 5-cycle (as a subgraph).
        A C5 exists iff some edge (u,v) has a simple path of length 4 between
        u and v not using that edge (dist(u,v)==4 in the graph)."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        for u, v in edge_set:
            # BFS from u, do not traverse the direct edge u-v (so the found
            # path of length 4 is a genuine C5 when combined with (u,v)).
            dist = [-1] * n
            dist[u] = 0
            q = [u]
            head = 0
            found = False
            while head < len(q) and not found:
                cur = q[head]; head += 1
                d = dist[cur]
                if d >= 4:
                    break
                for nb in adj[cur]:
                    if dist[nb] != -1:
                        continue
                    if cur == u and nb == v:
                        continue  # skip the direct edge on the first step
                    dist[nb] = d + 1
                    if nb == v:
                        found = True
                        break
                    q.append(nb)
            if found and dist[v] == 4:
                return False
        return True

    # ---------- Exact alpha check (find independent set of size 9) ----------
    def has_independent_set_9(n, edge_set):
        """Return True iff there is an independent set of size >= 9."""
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)

        import sys
        sys.setrecursionlimit(10000)

        def branch(cand, need):
            if need == 0:
                return True
            if len(cand) < need:
                return False
            # Branch on the first candidate vertex.
            v = cand[0]
            rest = cand[1:]
            # Include v: next candidates = rest minus neighbors of v
            nxt = [x for x in rest if x not in adj[v]]
            if branch(nxt, need - 1):
                return True
            # Exclude v
            if branch(rest, need):
                return True
            return False

        return branch(list(range(n)), 9)

    # ---------- A valid C5-free graph with alpha < 9? ----------
    def is_valid(n, edge_set):
        if not is_c5_free(n, edge_set):
            return False
        if has_independent_set_9(n, edge_set):
            return False
        return True

    def to_edges(edge_set):
        return [[u, v] for (u, v) in edge_set if u < v]

    # ---------- Fallback: 8 disjoint K4s ----------
    def fallback():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            vs = [base, base + 1, base + 2, base + 3]
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([vs[i], vs[j]])
        return {'n': n, 'edges': edges}

    # ---------- Generic search for a given n ----------
    def try_n(n, max_attempts):
        for attempt in range(max_attempts):
            if time_left() < 2.0:
                break
            p = 0.10 + 0.02 * (attempt % 10)  # vary edge probability
            # Build random graph
            edge_set = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        edge_set.add((u, v))
            # Drop edges until C5-free (greedy)
            es = list(edge_set)
            random.shuffle(es)
            for e in es:
                if time_left() < 2.0:
                    break
                if is_c5_free(n, edge_set):
                    break
                edge_set.discard(e)
            if not is_c5_free(n, edge_set):
                continue
            # Repair alpha: iteratively find large independent sets and add
            # edges within them that do not create a C5.
            for repair_round in range(60):
                if time_left() < 2.0:
                    break
                if not has_independent_set_9(n, edge_set):
                    break  # alpha < 9 already
                # Greedy independent set
                adj = [set() for _ in range(n)]
                for u, v in edge_set:
                    adj[u].add(v)
                    adj[v].add(u)
                remaining = set(range(n))
                indep = []
                while remaining:
                    v = min(remaining, key=lambda x: len(adj[x] & remaining))
                    indep.append(v)
                    remaining.discard(v)
                    remaining -= adj[v]
                # Add edges within indep that keep C5-free
                for i in range(len(indep)):
                    for j in range(i + 1, len(indep)):
                        u, v = indep[i], indep[j]
                        if (u, v) in edge_set or (v, u) in edge_set:
                            continue
                        new_set = edge_set | {(u, v)}
                        if is_c5_free(n, new_set):
                            edge_set = new_set
                            adj[u].add(v); adj[v].add(u)
            # Final exact check
            if is_c5_free(n, edge_set) and not has_independent_set_9(n, edge_set):
                return {'n': n, 'edges': to_edges(edge_set)}
        return None

    # ---------- Main ----------
    # Try n=35 first, then 34, then 33. Fall back to 32.
    for n_try, att in ((35, 150), (34, 150), (33, 250)):
        if time_left() < 3.0:
            break
        res = try_n(n_try, att)
        if res is not None:
            return res

    return fallback()
