"""Rewritten: constructive C4-free graph with no independent 11-set, maximizing n."""
import time
import random

_DEADLINE_MARGIN = 15.0
_LIMIT = 150.0

def construct():
    random.seed(20240517)
    t0 = time.time()
    deadline = _LIMIT - _DEADLINE_MARGIN

    def time_ok():
        return time.time() - t0 < deadline

    # State
    adj = []  # list of sets
    edge_set = set()  # frozenset pairs

    def n():
        return len(adj)

    def add_vertex():
        adj.append(set())

    def can_add(u, v):
        if u == v:
            return False
        if (u, v) in edge_set or (v, u) in edge_set:
            return False
        # C4: at most 1 common neighbor
        common = adj[u] & adj[v]
        return len(common) <= 1

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))

    def remove_edge(u, v):
        adj[u].discard(v)
        adj[v].discard(u)
        if u < v:
            edge_set.discard((u, v))
        else:
            edge_set.discard((v, u))

    def find_independent_11():
        """Find an independent set of size 11, or None."""
        rem = list(range(n()))
        # Greedy: pick vertex with fewest neighbors in rem
        chosen = []
        rem_set = set(rem)
        while rem_set and len(chosen) < 11:
            # find vertex in rem_set with min degree within rem_set
            best = None
            best_d = None
            for v in rem_set:
                d = len(adj[v] & rem_set)
                if best is None or d < best_d:
                    best = v
                    best_d = d
            chosen.append(best)
            rem_set.discard(best)
            rem_set -= adj[best]
        if len(chosen) >= 11:
            return chosen[:11]
        return None

    def repair(max_iters=10000):
        """Try to eliminate independent 11-sets by adding edges."""
        for _ in range(max_iters):
            if not time_ok():
                return False
            iso = find_independent_11()
            if iso is None:
                return True
            # try to add a random edge within iso
            pairs = [(iso[i], iso[j]) for i in range(len(iso)) for j in range(i+1, len(iso))]
            random.shuffle(pairs)
            placed = False
            for (a, b) in pairs:
                if can_add(a, b):
                    add_edge(a, b)
                    placed = True
                    break
            if not placed:
                return False
        return find_independent_11() is None

    # ---- Initialize with 10 disjoint triangles (n=30, alpha=10) ----
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        add_vertex(); add_vertex(); add_vertex()
        add_edge(a, b); add_edge(a, c); add_edge(b, c)

    best_n = n()
    best_edges = sorted([(u, v) for (u, v) in edge_set])

    def snapshot():
        nonlocal best_n, best_edges
        best_n = n()
        best_edges = sorted([(u, v) for (u, v) in edge_set])

    # ---- Grow: add vertices with random edges, repair independence ----
    attempts = 0
    while time_ok() and attempts < 300:
        attempts += 1
        old_n = n()
        add_vertex()
        nv = n() - 1
        # random edges to existing vertices
        k = random.randint(1, 4)
        targets = random.sample(range(old_n), min(k, old_n))
        added = []
        for t in targets:
            if can_add(nv, t):
                add_edge(nv, t)
                added.append(t)
        # repair
        if find_independent_11() is not None:
            if not repair(max_iters=5000):
                # revert
                for t in added:
                    remove_edge(nv, t)
                adj.pop()
                continue
        if find_independent_11() is None:
            snapshot()
        else:
            for t in added:
                remove_edge(nv, t)
            adj.pop()

    # ---- Further polish: try to add more vertices with denser connections ----
    while time_ok():
        old_n = n()
        add_vertex()
        nv = n() - 1
        k = random.randint(2, 5)
        targets = random.sample(range(old_n), min(k, old_n))
        added = []
        for t in targets:
            if can_add(nv, t):
                add_edge(nv, t)
                added.append(t)
        if find_independent_11() is not None:
            if not repair(max_iters=3000):
                for t in added:
                    remove_edge(nv, t)
                adj.pop()
                continue
        if find_independent_11() is None:
            snapshot()
        else:
            for t in added:
                remove_edge(nv, t)
            adj.pop()

    return {'n': best_n, 'edges': [[u, v] for (u, v) in best_edges]}
