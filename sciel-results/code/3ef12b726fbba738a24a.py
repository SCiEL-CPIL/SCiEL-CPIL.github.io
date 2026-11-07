"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import permutations

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}


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


def _find_hamiltonian(adj, used, deadline, start=0, timeout_check_interval=100000):
    """DFS for a Hamiltonian cycle on the pinned vertex set, avoiding used edges."""
    n = P['n']
    nbrs = [sorted(adj[v]) for v in range(n)]
    calls = [0]

    def dfs(path, pos, count):
        calls[0] += 1
        if calls[0] % timeout_check_interval == 0:
            if time.time() > deadline:
                return None
        if count == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and last in adj[start]:
                return path[:]
            return None
        last = path[-1]
        cands = []
        for y in nbrs[last]:
            if y == start:
                continue
            if pos[y]:
                continue
            if _edge_key(last, y) in used:
                continue
            cands.append(y)
        # MRV heuristic: prefer vertices with fewer forward options
        def fwd(y):
            k = 0
            for z in nbrs[y]:
                if z == start:
                    continue
                if pos[z]:
                    continue
                if _edge_key(y, z) in used:
                    continue
                k += 1
            return k
        cands.sort(key=fwd)
        for y in cands:
            path.append(y)
            pos[y] = True
            res = dfs(path, pos, count + 1)
            if res is not None:
                return res
            pos[y] = False
            path.pop()
        return None

    path = [start]
    pos = [False] * n
    pos[start] = True
    return dfs(path, pos, 1)


def _find_hamiltonian_with_connectivity(adj, used, deadline, start=0):
    """DFS with connectivity pruning for stronger search."""
    n = P['n']
    nbrs = [sorted(adj[v]) for v in range(n)]
    calls = [0]
    check_interval = 50000

    def check_connectivity(cur, pos):
        """Check all unvisited vertices are reachable from cur via unused edges."""
        unvisited = [v for v in range(n) if not pos[v] and v != start]
        if not unvisited:
            return True
        seen = set()
        stack = [cur]
        seen.add(cur)
        while stack:
            x = stack.pop()
            for y in nbrs[x]:
                if y in seen:
                    continue
                if y == start:
                    continue
                if pos[y]:
                    continue
                if _edge_key(x, y) in used:
                    continue
                seen.add(y)
                stack.append(y)
        for v in unvisited:
            if v not in seen:
                return False
        return True

    def dfs(path, pos, count):
        calls[0] += 1
        if calls[0] % check_interval == 0:
            if time.time() > deadline:
                return None
        if count == n:
            last = path[-1]
            e = _edge_key(last, start)
            if e not in used and last in adj[start]:
                return path[:]
            return None
        last = path[-1]
        cands = []
        for y in nbrs[last]:
            if y == start:
                continue
            if pos[y]:
                continue
            if _edge_key(last, y) in used:
                continue
            cands.append(y)
        
        # Connectivity prune: only proceed if all unvisited reachable
        if len(cands) > 1:
            # Try each candidate and check connectivity
            valid_cands = []
            for y in cands:
                pos[y] = True
                if check_connectivity(y, pos):
                    valid_cands.append(y)
                pos[y] = False
            if not valid_cands:
                return None
            cands = valid_cands
        
        # MRV
        def fwd(y):
            k = 0
            for z in nbrs[y]:
                if z == start:
                    continue
                if pos[z]:
                    continue
                if _edge_key(y, z) in used:
                    continue
                k += 1
            return k
        cands.sort(key=fwd)
        
        for y in cands:
            path.append(y)
            pos[y] = True
            res = dfs(path, pos, count + 1)
            if res is not None:
                return res
            pos[y] = False
            path.pop()
        return None

    path = [start]
    pos = [False] * n
    pos[start] = True
    return dfs(path, pos, 1)


def construct():
    deadline = time.time() + 75  # safety margin under 90s kill

    n = P['n']
    adj = {u: set() for u in range(n)}
    for (u, v) in P['edges']:
        adj[u].add(v)
        adj[v].add(u)

    vertex_set = set(range(n))

    # Try known good cycles first
    # Cycle 1: natural path 0,1,2,...,20
    base = list(range(n))
    if not _is_valid_cycle(base, adj, vertex_set):
        base = None

    used = set()
    cycles = []

    # Cycle 1
    if base is not None:
        c1 = base
    else:
        c1 = _find_hamiltonian(adj, used, deadline)
    if c1 is not None and _is_valid_cycle(c1, adj, vertex_set):
        cycles.append(c1)
        for e in _cycle_edges(c1):
            used.add(e)

    # Cycle 2: try some known patterns, then DFS
    if time.time() < deadline:
        # Try a different ordering
        c2 = None
        # Try pattern: 0,2,4,6,8,10,12,14,16,18,20,19,17,15,13,11,9,7,5,3,1
        pattern2 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
        if _is_valid_cycle(pattern2, adj, vertex_set):
            if all(e not in used for e in _cycle_edges(pattern2)):
                c2 = pattern2
        
        if c2 is None:
            c2 = _find_hamiltonian_with_connectivity(adj, used, deadline)
        
        if c2 is None:
            c2 = _find_hamiltonian(adj, used, deadline)
        
        if c2 is not None and _is_valid_cycle(c2, adj, vertex_set):
            cycles.append(c2)
            for e in _cycle_edges(c2):
                used.add(e)

    # Cycle 3: DFS on residual
    if time.time() < deadline:
        c3 = None
        # Try pattern: 0,5,10,15,20,19,18,17,16,14,13,11,9,7,6,4,3,2,1,17,18 - no, must be valid
        # Let's just use DFS
        c3 = _find_hamiltonian_with_connectivity(adj, used, deadline)
        
        if c3 is None:
            c3 = _find_hamiltonian(adj, used, deadline)
        
        if c3 is not None and _is_valid_cycle(c3, adj, vertex_set):
            cycles.append(c3)
            for e in _cycle_edges(c3):
                used.add(e)

    # Validate
    while cycles and not _is_valid_solution(cycles, adj, vertex_set):
        cycles.pop()

    if not cycles:
        fb = _find_hamiltonian(adj, set(), deadline)
        if fb is None:
            fb = list(range(n))
        return [[int(v) for v in fb]]

    return [[int(v) for v in c] for c in cycles]
# EVOLVE-BLOCK-END
