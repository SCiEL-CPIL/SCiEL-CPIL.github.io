import time
import random
from collections import deque

_T0 = time.time()
_DEADLINE = 145.0  # wall-clock limit with margin

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _adj_from_edges(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _find_long_cycle(adj, n, budget=2.0):
    """Find a long cycle in the graph represented by adj (list of sets)."""
    start_time = time.time()
    best_cycle = None
    best_len = 0

    # Try from several starting vertices
    starts = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    for start in starts:
        if time.time() - start_time > budget:
            break
        if len(adj[start]) < 2:
            continue
        # DFS with backtracking, depth-limited
        found = _dfs_cycle(adj, n, start, 30)
        if found and len(found) > best_len:
            best_cycle = found
            best_len = len(found)
            if best_len >= n:  # Hamiltonian
                break
    return best_cycle


def _dfs_cycle(adj, n, start, max_depth):
    """Backtracking DFS to find a long cycle through start."""
    best = [None]
    best_len = [0]

    def dfs(v, path, used):
        if time.time() - start > 5:  # safety
            return
        if len(path) > max_depth:
            return
        if v == start and len(path) >= 3:
            if len(path) > best_len[0]:
                best_len[0] = len(path)
                best[0] = path[:]
            return
        for w in adj[v]:
            if w in used:
                if w == start and len(path) >= 3:
                    if len(path) > best_len[0]:
                        best_len[0] = len(path)
                        best[0] = path[:]
                continue
            used.add(w)
            path.append(w)
            dfs(w, path, used)
            path.pop()
            used.discard(w)
            if best_len[0] >= max_depth:
                return

    dfs(start, [start], {start})
    return best[0]


def _find_any_cycle(adj, n):
    """Quick BFS-based cycle finding. Returns cycle or None."""
    for start in range(n):
        if len(adj[start]) < 2:
            continue
        # BFS
        parent = {start: -1}
        depth = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for w in adj[v]:
                if w not in depth:
                    depth[w] = depth[v] + 1
                    parent[w] = v
                    q.append(w)
                elif w != parent.get(v, -1) and depth[w] < depth[v] - 1:
                    # Found a back edge
                    cycle = []
                    cur = v
                    while cur != w:
                        cycle.append(cur)
                        cur = parent[cur]
                    cycle.append(w)
                    return cycle
    return None


def _decompose_even(adj, n, time_budget=5.0):
    """Decompose even-degree graph into cycles. Returns list of cycles."""
    cycles = []
    start_time = time.time()
    while True:
        if time.time() - start_time > time_budget:
            break
        # Check if any edges remain
        has_edge = False
        for v in range(n):
            if adj[v]:
                has_edge = True
                break
        if not has_edge:
            break
        # Find a cycle
        cycle = _find_long_cycle(adj, n, budget=1.5)
        if cycle is None:
            cycle = _find_any_cycle(adj, n)
        if cycle is None:
            # Shouldn't happen for even graph, but break to avoid infinite loop
            break
        # Remove cycle edges
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            adj[u].discard(v)
            adj[v].discard(u)
        cycles.append(cycle)
    return cycles


def _max_even_subgraph(adj, n, edges, time_budget=8.0):
    """Find a large even-degree subgraph using greedy + local search."""
    start_time = time.time()
    # Start with all edges
    sel = [set() for _ in range(n)]
    for (u, v) in edges:
        sel[u].add(v)
        sel[v].add(u)

    def odd_count():
        return sum(1 for v in range(n) if len(sel[v]) % 2 == 1)

    # Greedy: repeatedly find and remove cycles
    improved = True
    while improved and time.time() - start_time < time_budget - 1:
        improved = False
        for _ in range(500):
            if time.time() - start_time > time_budget - 1:
                break
            # Find a cycle in current subgraph
            cycle = _find_any_cycle(sel, n)
            if cycle is None:
                break
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                sel[u].discard(v)
                sel[v].discard(u)
            improved = True

    # Local search: try adding edges that reduce odd count
    for _ in range(3):
        if time.time() - start_time > time_budget:
            break
        improved = False
        for (u, v) in edges:
            if u in sel[v]:
                continue
            # Adding this edge toggles parity of u and v
            if len(sel[u]) % 2 == 1 and len(sel[v]) % 2 == 1:
                sel[u].add(v)
                sel[v].add(u)
                improved = True
            elif len(sel[u]) % 2 == 0 and len(sel[v]) % 2 == 0:
                # Adding keeps them even - good for coverage
                sel[u].add(v)
                sel[v].add(u)
                improved = True
        if not improved:
            break

    # Final cleanup: remove edges to ensure all even
    for v in range(n):
        while len(sel[v]) % 2 == 1:
            if not sel[v]:
                break
            w = next(iter(sel[v]))
            sel[v].discard(w)
            sel[w].discard(v)

    return sel


def _validate(adj, n, pieces):
    """Check pieces are edge-disjoint and cover all edges exactly."""
    seen = set()
    for p in pieces:
        k = len(p)
        if k == 2:
            e = tuple(sorted(p))
        elif k >= 3:
            for i in range(k):
                e = tuple(sorted((p[i], p[(i + 1) % k])))
                if e in seen:
                    return False
                seen.add(e)
            continue
        else:
            return False
        if e in seen:
            return False
        seen.add(e)
    all_edges = set()
    for v in range(n):
        for w in adj[v]:
            if v < w:
                all_edges.add((v, w))
    return seen == all_edges


def _solve_one(n, p, seed, time_budget=35.0):
    edges = _instance(n, p, seed)
    adj = _adj_from_edges(edges, n)
    m = len(edges)

    # Baseline: all single edges
    best_pieces = [[u, v] for (u, v) in edges]
    best_count = m

    # Try to find a large even subgraph and decompose
    start_time = time.time()
    try:
        even_sel = _max_even_subgraph(adj, n, edges, time_budget=min(time_budget, 10.0))
        # Build pieces from even subgraph
        pieces = []
        even_edges = set()
        for v in range(n):
            for w in even_sel[v]:
                if v < w:
                    even_edges.add((v, w))

        # Decompose even subgraph into cycles
        even_adj = [set() for _ in range(n)]
        for (u, v) in even_edges:
            even_adj[u].add(v)
            even_adj[v].add(u)

        cycles = _decompose_even(even_adj, n, time_budget=min(time_budget - 10, 15.0))
        for c in cycles:
            pieces.append(c)

        # Add remaining edges as single edges
        for (u, v) in edges:
            if (u, v) not in even_edges:
                pieces.append([u, v])

        if _validate(adj, n, pieces) and len(pieces) < best_count:
            best_count = len(pieces)
            best_pieces = pieces

    except Exception:
        pass

    # Try direct cycle extraction on full graph (greedy long cycles)
    if time.time() - start_time < time_budget - 2:
        try:
            work_adj = [set(s) for s in adj]
            pieces2 = []
            start2 = time.time()
            while True:
                if time.time() - start2 > time_budget - 2:
                    break
                # Check if edges remain
                has_edge = False
                for v in range(n):
                    if work_adj[v]:
                        has_edge = True
                        break
                if not has_edge:
                    break
                # Find a long cycle
                cycle = _find_long_cycle(work_adj, n, budget=1.0)
                if cycle is None:
                    cycle = _find_any_cycle(work_adj, n)
                if cycle is None:
                    # No more cycles; add remaining edges as singles
                    for v in range(n):
                        for w in list(work_adj[v]):
                            if v < w:
                                pieces2.append([v, w])
                                work_adj[v].discard(w)
                                work_adj[w].discard(v)
                    break
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    work_adj[u].discard(v)
                    work_adj[v].discard(u)
                pieces2.append(cycle)

            if _validate(adj, n, pieces2) and len(pieces2) < best_count:
                best_count = len(pieces2)
                best_pieces = pieces2
        except Exception:
            pass

    return best_pieces


def construct():
    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    for spec in specs:
        if time.time() - _T0 > _DEADLINE - 2:
            # Fallback: all single edges
            n, p, seed = spec
            edges = _instance(n, p, seed)
            results.append([[u, v] for (u, v) in edges])
            continue
        try:
            pieces = _solve_one(spec[0], spec[1], spec[2], time_budget=35.0)
        except Exception:
            edges = _instance(spec[0], spec[1], spec[2])
            pieces = [[u, v] for (u, v) in edges]
        results.append(pieces)
    return results
