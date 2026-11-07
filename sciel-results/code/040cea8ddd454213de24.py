"""Rewritten program: explicit search over vertex sets for max edge-disjoint Hamiltonian cycles."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adj(n, edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_ham_cycle(vertices, adj, used_edges, timeout_per_try=None, start=None):
    """Find a Hamiltonian cycle on vertex set using only unused edges. Fast DFS with Warnsdorff + edge-pruning."""
    k = len(vertices)
    if k < 3:
        return None
    vset = set(vertices)
    if start is None:
        start = min(vset)
    if start not in vset:
        return None
    # quick degree feasibility
    for v in vertices:
        deg = 0
        for nb in adj[v]:
            if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                deg += 1
        if deg < 2:
            return None

    path = [start]
    visited = set([start])
    deadline_local = time.time() + timeout_per_try if timeout_per_try else None
    deadline_check_interval = 100
    call_count = [0]

    sys.setrecursionlimit(10000)

    # Precompute available edges within vset for faster lookup
    avail = set()
    for u in vset:
        for v in adj[u]:
            if v in vset:
                e = (min(u, v), max(u, v))
                if e not in used_edges:
                    avail.add(e)

    def dfs(cur):
        call_count[0] += 1
        if deadline_local and call_count[0] % deadline_check_interval == 0 and time.time() > deadline_local:
            return None
        if len(path) == k:
            last = path[-1]
            e_close = (min(last, start), max(last, start))
            if e_close in avail:
                return list(path)
            return None
        remaining = k - len(path)
        # candidates: unvisited vertices in vset adjacent to cur via unused edge
        cands = []
        for nxt in adj[cur]:
            if nxt not in visited and nxt in vset:
                e = (min(cur, nxt), max(cur, nxt))
                if e not in avail:
                    continue
                # Warnsdorff: prefer nxt with fewer onward options
                if remaining == 1:
                    # must be able to close to start
                    e_close = (min(nxt, start), max(nxt, start))
                    if e_close not in avail:
                        continue
                    onward = 0
                else:
                    onward = 0
                    for nn in adj[nxt]:
                        if nn not in visited and nn in vset and nn != cur:
                            if (min(nxt, nn), max(nxt, nn)) in avail:
                                onward += 1
                    # Additional pruning: if onward == 0 and remaining > 1, dead end
                    if onward == 0:
                        continue
                cands.append((onward, nxt))
        if not cands:
            return None
        cands.sort()
        for _, nxt in cands:
            e = (min(cur, nxt), max(cur, nxt))
            visited.add(nxt)
            path.append(nxt)
            res = dfs(nxt)
            if res is not None:
                return res
            path.pop()
            visited.discard(nxt)
        return None

    return dfs(start)

def _find_max_disjoint_cycles(vertices, adj, max_cycles=10, time_budget=5.0):
    """Greedily find max edge-disjoint Hamiltonian cycles on vertices."""
    vlist = list(vertices)
    vset = set(vlist)
    used_edges = set()
    cycles = []
    t0 = time.time()
    rng = random.Random(12345)
    for _ in range(max_cycles):
        if time.time() - t0 > time_budget:
            break
        found = None
        # heuristic: start at vertex with lowest remaining degree
        rem_degs = []
        for v in vlist:
            d = 0
            for nb in adj[v]:
                if nb in vset and (min(v, nb), max(v, nb)) not in used_edges:
                    d += 1
            rem_degs.append((d, v))
        rem_degs.sort()
        # Shuffle to add diversity, but keep low-degree vertices prioritized
        shuffled_starts = list(rem_degs)
        rng.shuffle(shuffled_starts)
        # Interleave: first few in degree order, rest shuffled
        ordered = rem_degs[:3] + [x for x in shuffled_starts if x not in rem_degs[:3]]
        for _, st in ordered:
            if time.time() - t0 > time_budget:
                break
            remaining_time = time_budget - (time.time() - t0)
            if remaining_time < 0.3:
                break
            per_try = min(1.5, remaining_time / 3.0)
            found = _find_ham_cycle(vlist, adj, used_edges, timeout_per_try=per_try, start=st)
            if found is not None:
                break
        if found is None:
            break
        # verify all edges unused and mark them
        ok = True
        for i in range(len(found)):
            u, v = found[i], found[(i + 1) % len(found)]
            e = (min(u, v), max(u, v))
            if e in used_edges:
                ok = False
                break
            used_edges.add(e)
        if not ok:
            break
        cycles.append(found)
    return cycles

def _induced_min_deg(subset_set, adj):
    m = 10**9
    for v in subset_set:
        d = sum(1 for nb in adj[v] if nb in subset_set)
        if d < m:
            m = d
    return m

def _induced_edge_count(subset_set, edges):
    c = 0
    for u, v in edges:
        if u in subset_set and v in subset_set:
            c += 1
    return c

def construct():
    t0 = time.time()
    DEADLINE = 85  # safety margin before 90s kill
    
    n = P['n']
    edges = P['edges']
    adj = _build_adj(n, edges)
    
    best_cycles = []
    best_count = 0
    best_vertices = None
    
    def try_set(vlist, budget):
        nonlocal best_cycles, best_count, best_vertices
        if len(vlist) < 3:
            return
        sset = set(vlist)
        md = _induced_min_deg(sset, adj)
        if md < 2:
            return
        ec = _induced_edge_count(sset, edges)
        max_possible = min(md // 2, ec // len(vlist))
        if max_possible < best_count:
            return
        if time.time() - t0 > DEADLINE:
            return
        b = min(budget, max(0.5, DEADLINE - (time.time() - t0)))
        if b < 0.3:
            return
        cycles = _find_max_disjoint_cycles(vlist, adj, max_cycles=min(10, max_possible + 1), time_budget=b)
        if len(cycles) > best_count:
            best_count = len(cycles)
            best_cycles = cycles
            best_vertices = list(vlist)
    
    # Strategy A: full vertex set with generous budget
    try_set(list(range(n)), 20.0)
    
    # Strategy B: dense subsets
    remaining = DEADLINE - (time.time() - t0)
    if remaining > 2:
        deg = {v: len(adj[v]) for v in range(n)}
        verts_by_deg = sorted(range(n), key=lambda v: -deg[v])
        
        random.seed(42)
        tried = set()
        full = frozenset(range(n))
        tried.add(full)
        
        # 1) Try high-degree core subsets
        for size in range(14, 5, -1):
            if time.time() - t0 > DEADLINE - 1:
                break
            core = verts_by_deg[:size]
            fs = frozenset(core)
            if fs not in tried:
                tried.add(fs)
                try_set(list(core), min(3.0, max(0.5, DEADLINE - (time.time() - t0))))
        
        # 2) Consecutive windows
        for start in range(n):
            if time.time() - t0 > DEADLINE - 1:
                break
            for size in range(6, min(15, n - start + 1)):
                window = list(range(start, start + size))
                fs = frozenset(window)
                if fs not in tried:
                    tried.add(fs)
                    try_set(window, min(2.0, max(0.5, DEADLINE - (time.time() - t0))))
        
        # 3) Random subsets biased to high degree and connectivity
        if time.time() - t0 > DEADLINE - 3:
            return _finalize(best_cycles, best_count, n)
        
        budget_each = max(0.5, (DEADLINE - 2 - (time.time() - t0)) / 60.0)
        trials = 0
        while time.time() - t0 < DEADLINE - 2 and trials < 400:
            trials += 1
            size = random.randint(6, 14)
            subset = set()
            # Start with a random seed vertex, grow by adding neighbors
            seed_v = random.choice(verts_by_deg[:8])
            subset.add(seed_v)
            frontier = list(adj[seed_v])
            random.shuffle(frontier)
            while len(subset) < size and frontier:
                v = frontier.pop()
                if v not in subset:
                    subset.add(v)
                    for nb in adj[v]:
                        if nb not in subset:
                            frontier.append(nb)
            # If still not enough, fill randomly
            if len(subset) < size:
                for v in range(n):
                    if v not in subset:
                        subset.add(v)
                    if len(subset) == size:
                        break
            if len(subset) < 3:
                continue
            fs = frozenset(subset)
            if fs in tried:
                continue
            tried.add(fs)
            sset = set(subset)
            md = _induced_min_deg(sset, adj)
            if md < 2:
                continue
            ec = _induced_edge_count(sset, edges)
            if ec < len(subset):
                continue
            try_set(list(subset), budget_each)
    
    return _finalize(best_cycles, best_count, n)

def _finalize(cycles, count, n):
    if not cycles:
        edges = P['edges']
        adj = _build_adj(n, edges)
        used = set()
        c = _find_ham_cycle(list(range(n)), adj, used, timeout_per_try=5.0)
        if c:
            cycles = [c]
        else:
            cycles = [list(range(n))]
    out = []
    seen_edges = set()
    vset = None
    for cyc in cycles[:10]:
        cyc = list(cyc)
        if len(cyc) < 3:
            continue
        if len(set(cyc)) != len(cyc):
            continue
        if vset is None:
            vset = set(cyc)
        elif set(cyc) != vset:
            continue
        ok = True
        for i in range(len(cyc)):
            u, v = cyc[i], cyc[(i + 1) % len(cyc)]
            e = (min(u, v), max(u, v))
            if e in seen_edges:
                ok = False
                break
            seen_edges.add(e)
        if not ok:
            continue
        out.append(tuple(cyc))
    if not out:
        out = [tuple(range(n))]
    return out

# EVOLVE-BLOCK-END
