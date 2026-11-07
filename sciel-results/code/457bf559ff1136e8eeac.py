"""Packing edge-disjoint cycles on one vertex set."""
# EVOLVE-BLOCK-START
import time
import sys
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

DEADLINE_MARGIN = 15  # seconds to spare before 90s kill


def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _cycle_edges(cycle):
    m = len(cycle)
    out = []
    for i in range(m):
        out.append(_edge_key(cycle[i], cycle[(i + 1) % m]))
    return out


def _is_valid_cycle(cycle, adj, vertex_set):
    if not isinstance(cycle, (list, tuple)):
        return False
    m = len(cycle)
    if m < 3:
        return False
    if len(cycle) != len(set(cycle)):
        return False
    for v in cycle:
        if not isinstance(v, int) or isinstance(v, bool):
            return False
        if v not in vertex_set:
            return False
    for i in range(m):
        a = cycle[i]
        b = cycle[(i + 1) % m]
        if b not in adj[a]:
            return False
    if set(cycle) != vertex_set:
        return False
    return True


def _is_valid_solution(cycles, adj, vertex_set):
    for c in cycles:
        if not _is_valid_cycle(c, adj, vertex_set):
            return False
    seen = set()
    for c in cycles:
        for e in _cycle_edges(c):
            if e in seen:
                return False
            seen.add(e)
    return True


def _complete_from_residual(adj, edge_set, used):
    """If residual (host edges minus used) is a single 2-regular Hamiltonian cycle,
    walk it and return the vertex list. Returns None otherwise."""
    n = P['n']
    residual = set()
    for (u, v) in P['edges']:
        e = _edge_key(u, v)
        if e not in used:
            residual.add(e)
    rem = [[] for _ in range(n)]
    for (u, v) in residual:
        rem[u].append(v)
        rem[v].append(u)
    for u in range(n):
        if len(rem[u]) != 2:
            return None
    # walk
    cyc = [0]
    prev = -1
    cur = 0
    for _ in range(n):
        a, b = rem[cur]
        nxt = a if a != prev else b
        if nxt == 0 and len(cyc) < n:
            return None
        cyc.append(nxt)
        prev, cur = cur, nxt
        if cur == 0:
            break
    if len(cyc) != n + 1 or cyc[-1] != 0:
        return None
    return cyc[1:]


def construct():
    start_time = time.time()
    deadline = start_time + 90 - DEADLINE_MARGIN

    n = P['n']
    edges_list = P['edges']
    num_edges = len(edges_list)

    adj = {u: set() for u in range(n)}
    edge_set = set()
    edge_index = {}
    for i, (u, v) in enumerate(edges_list):
        key = _edge_key(u, v)
        edge_index[key] = i
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add(key)

    vertex_set = set(range(n))

    def time_left():
        return deadline - time.time()

    def find_hamiltonian(used, start_vertex):
        """
        Find a Hamiltonian cycle on the full vertex set avoiding used edges,
        using DFS with MRV ordering and connectivity pruning.
        Returns list of vertices or None.
        """
        if time_left() < 2:
            return None

        nbrs = [sorted(adj[v]) for v in range(n)]
        path = [start_vertex]
        pos = [False] * n
        pos[start_vertex] = True
        count = 1
        remaining = n - 1
        local_used = set(used)

        # connectivity prune: from 'cur', can we reach every unvisited vertex
        # via unused edges staying within unvisited vertices (cur as entry)?
        def connected_from(cur):
            seen = {cur}
            stack = [cur]
            target = 0
            while stack:
                x = stack.pop()
                for y in nbrs[x]:
                    if y in seen:
                        continue
                    if y == start_vertex:
                        continue
                    if not pos[y]:
                        target += 1
                        seen.add(y)
                        stack.append(y)
            return target == remaining

        def dfs():
            nonlocal count, remaining
            if time.time() > deadline:
                return None
            if count == n:
                last = path[-1]
                e = _edge_key(last, start_vertex)
                if e not in local_used and last in adj[start_vertex]:
                    return path[:]
                return None
            last = path[-1]
            cands = []
            for y in nbrs[last]:
                if y == start_vertex:
                    continue
                if pos[y]:
                    continue
                if _edge_key(last, y) in local_used:
                    continue
                cands.append(y)

            if not cands:
                return None

            # connectivity prune only when it is cheap-ish
            if remaining > 1:
                if not connected_from(last):
                    return None

            # MRV: order by number of forward options
            def fwd(y):
                k = 0
                for z in nbrs[y]:
                    if z == start_vertex:
                        continue
                    if pos[z]:
                        continue
                    if _edge_key(y, z) in local_used:
                        continue
                    k += 1
                return k
            cands.sort(key=fwd)

            for y in cands:
                path.append(y)
                pos[y] = True
                count += 1
                remaining -= 1
                local_used.add(_edge_key(last, y))
                res = dfs()
                if res is not None:
                    return res
                remaining += 1
                count -= 1
                pos[y] = False
                path.pop()
                local_used.discard(_edge_key(last, y))
            return None

        return dfs()

    def add_cycle(cycles, used, cyc):
        for e in _cycle_edges(cyc):
            used.add(e)
        cycles.append(cyc)

    used = set()
    cycles = []

    # Cycle 1: try the natural path 0..20 first (instant), else DFS.
    base = list(range(n))
    if _is_valid_cycle(base, adj, vertex_set):
        c1 = base
    else:
        c1 = None
        for sv in range(n):
            if time_left() < 3:
                break
            c1 = find_hamiltonian(used, sv)
            if c1 is not None:
                break
    if c1 is not None and _is_valid_cycle(c1, adj, vertex_set):
        add_cycle(cycles, used, c1)

    # Cycle 2: residual completion, else DFS.
    if time_left() > 3:
        c2 = _complete_from_residual(adj, edge_set, used)
        if c2 is None or not _is_valid_cycle(c2, adj, vertex_set):
            c2 = None
            for sv in range(n):
                if time_left() < 3:
                    break
                c2 = find_hamiltonian(used, sv)
                if c2 is not None:
                    break
        if c2 is not None and _is_valid_cycle(c2, adj, vertex_set):
            add_cycle(cycles, used, c2)

    # Cycle 3: residual completion, else DFS.
    if time_left() > 3:
        c3 = _complete_from_residual(adj, edge_set, used)
        if c3 is None or not _is_valid_cycle(c3, adj, vertex_set):
            c3 = None
            for sv in range(n):
                if time_left() < 3:
                    break
                c3 = find_hamiltonian(used, sv)
                if c3 is not None:
                    break
        if c3 is not None and _is_valid_cycle(c3, adj, vertex_set):
            add_cycle(cycles, used, c3)

    # Validate the full set; drop invalid trailing cycles until valid.
    while cycles and not _is_valid_solution(cycles, adj, vertex_set):
        cycles.pop()

    if not cycles:
        fb = find_hamiltonian(set(), 0)
        if fb is None:
            fb = list(range(n))
        return [[int(v) for v in fb]]

    return [[int(v) for v in c] for c in cycles]

# EVOLVE-BLOCK-END
