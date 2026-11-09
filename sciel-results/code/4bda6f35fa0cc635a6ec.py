import time
import random
from collections import defaultdict

def construct():
    def build_instance(n, d, seed):
        rng = random.Random(seed)
        deg = [0] * n
        adj = [set() for _ in range(n)]
        edges = []
        edge_set = set()
        attempts = 0
        max_attempts = 50000
        while attempts < max_attempts:
            attempts += 1
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e in edge_set:
                continue
            if deg[u] < d and deg[v] < d:
                edges.append(e)
                edge_set.add(e)
                deg[u] += 1
                deg[v] += 1
                adj[u].add(v)
                adj[v].add(u)
        return sorted(edges), adj, deg

    def solve(n, d, seed, time_budget):
        t0 = time.time()
        edges, adj, deg = build_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []

        edge_idx = {e: i for i, e in enumerate(edges)}
        E = len(edges)

        # Conflict: share vertex or joined by edge (distance-2)
        conflict = [0] * m  # bit i has bit j if conflict
        for (u, v) in edges:
            i = edge_idx[(u, v)]
            for nb_u in adj[u]:
                for e2 in adj[u]:
                    if e2 == v:
                        continue
                    # edge (u, e2)
                    key = (min(u, e2), max(u, e2))
                    j = edge_idx.get(key)
                    if j is not None and j != i:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)
            for nb_v in adj[v]:
                for e2 in adj[v]:
                    if e2 == u:
                        continue
                    key = (min(v, e2), max(v, e2))
                    j = edge_idx.get(key)
                    if j is not None and j != i:
                        conflict[i] |= (1 << j)
                        conflict[j] |= (1 << i)

        # Also direct shared vertices: edges sharing u
        vert_edges = defaultdict(list)
        for (u, v) in edges:
            vert_edges[u].append(edge_idx[(u, v)])
            vert_edges[v].append(edge_idx[(u, v)])
        for verts in vert_edges.values():
            for a in range(len(verts)):
                for b in range(a + 1, len(verts)):
                    conflict[verts[a]] |= (1 << verts[b])
                    conflict[verts[b]] |= (1 << verts[a])

        # Precompute neighbor lists for speed
        neigh = []
        for i in range(m):
            mask = conflict[i]
            lst = []
            j = 0
            while mask:
                if mask & 1:
                    lst.append(j)
                mask >>= 1
                j += 1
            neigh.append(lst)

        # DSATUR
        def dsatur():
            assign = [-1] * m
            color_sets = []
            degs = [len(neigh[i]) for i in range(m)]
            for _ in range(m):
                best = -1
                best_sat = -1
                best_deg = -1
                for i in range(m):
                    if assign[i] != -1:
                        continue
                    sat = set()
                    for j in neigh[i]:
                        if assign[j] != -1:
                            sat.add(assign[j])
                    s = len(sat)
                    if s > best_sat or (s == best_sat and degs[i] > best_deg):
                        best_sat = s
                        best_deg = degs[i]
                        best = i
                # assign best
                used = set()
                for j in neigh[best]:
                    if assign[j] != -1:
                        used.add(assign[j])
                c = 0
                while c in used:
                    c += 1
                assign[best] = c
                if c >= len(color_sets):
                    color_sets.append({best})
                else:
                    color_sets[c].add(best)
            return assign

        assign = dsatur()
        k = max(assign) + 1 if m else 0

        def count_conflicts(a):
            c = 0
            for i in range(m):
                for j in neigh[i]:
                    if j > i and a[i] == a[j]:
                        c += 1
            return c

        def min_conflicts(kc, iters, rng):
            a = [rng.randrange(kc) for _ in range(m)]
            # color counts per color
            from collections import Counter
            cc = Counter(a)
            # per-node conflicts
            node_conf = [0] * m
            for i in range(m):
                for j in neigh[i]:
                    if i < j and a[i] == a[j]:
                        node_conf[i] += 1
                        node_conf[j] += 1
            for _ in range(iters):
                # pick conflicted node
                cand = [i for i in range(m) if node_conf[i] > 0]
                if not cand:
                    break
                i = rng.choice(cand)
                old = a[i]
                # try all colors, pick best
                best_c = -1
                best_cost = m + 1
                for c in range(kc):
                    if c == old:
                        continue
                    # cost = neighbors with color c
                    cost = 0
                    for j in neigh[i]:
                        if a[j] == c:
                            cost += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_c = c
                if best_c == -1:
                    best_c = rng.randrange(kc)
                # update
                cc[old] -= 1
                cc[best_c] += 1
                for j in neigh[i]:
                    if a[j] == old:
                        node_conf[i] -= 1
                        node_conf[j] -= 1
                    if a[j] == best_c:
                        node_conf[i] += 1
                        node_conf[j] += 1
                a[i] = best_c
            return a, count_conflicts(a)

        deadline = t0 + time_budget
        rng = random.Random(seed * 13 + 7)

        # Try to reduce k
        # lower bound: max degree
        lb = max(deg) if deg else 0
        # try from k-1 down to lb, with time slice
        for kc in range(k - 1, max(lb - 1, 0), -1):
            if time.time() > deadline:
                break
            # time slice
            slice_t = 2.0
            iters = 30000
            a2, cf = min_conflicts(kc, iters, rng)
            if cf == 0:
                k = kc
                assign = a2
                # verify
                if count_conflicts(assign) == 0:
                    continue
                else:
                    break
            # else keep going
        return assign

    out = []
    global_deadline = time.time() + 145.0
    specs = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    for i, (n, d, seed) in enumerate(specs):
        remaining = global_deadline - time.time()
        if remaining < 5:
            # fallback: greedy
            edges, adj, deg = build_instance(n, d, seed)
            m = len(edges)
            assign = list(range(m))
            out.append(assign)
            continue
        t_budget = min(remaining - 3, 45.0)
        out.append(solve(n, d, seed, t_budget))
    return out

if __name__ == "__main__":
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(set(r)) if r else 0} colors")
