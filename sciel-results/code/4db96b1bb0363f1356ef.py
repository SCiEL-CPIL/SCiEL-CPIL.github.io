"""Packing edge-disjoint Hamiltonian cycles on the 21-vertex host graph."""
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _build_adjacency(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_key(u, v):
    return (u, v) if u < v else (v, u)

def construct():
    n = P['n']
    edges = P['edges']
    adj = _build_adjacency(n, edges)
    all_edges = set()
    for u, v in edges:
        all_edges.add(_edge_key(u, v))

    deadline = time.time() + 55.0  # safety margin before 60s kill
    found = []
    used_edges = set()
    remaining = set(all_edges)

    # Precompute: for each vertex, its neighbors
    nbrs = [sorted(adj[v]) for v in range(n)]

    def find_one_hamiltonian(deadline_t):
        """Find one Hamiltonian cycle using only edges in `remaining`."""
        # Symmetry breaking: fix vertex 0 as start, and require first step
        # to go to the smallest available neighbor to cut mirror cycles.
        start = 0
        # Choose first neighbor: pick smallest neighbor in remaining to break reversal symmetry
        first_choices = [v for v in nbrs[start] if _edge_key(start, v) in remaining]
        if not first_choices:
            return None
        first_choices.sort()
        # We'll try each first choice; use ordering by degree in remaining graph
        def rem_degree(x):
            return sum(1 for y in nbrs[x] if _edge_key(x, y) in remaining)
        first_choices.sort(key=lambda v: rem_degree(v))

        for first in first_choices:
            if time.time() > deadline_t:
                return None
            path = [start, first]
            visited = {start, first}
            if _backtrack(path, visited, start, deadline_t, remaining):
                return path
        return None

    def _backtrack(path, visited, start, deadline_t, rem):
        # path is list of vertices, visited is set
        if len(path) == n:
            last = path[-1]
            if _edge_key(last, start) in rem:
                return True
            return False
        if time.time() > deadline_t:
            return False
        cur = path[-1]
        # Candidate next vertices: unvisited neighbors of cur with edge in rem
        cands = [v for v in nbrs[cur] if v not in visited and _edge_key(cur, v) in rem]
        if not cands:
            return False
        # Heuristic: order by (degree in rem ascending) to prune
        def rem_degree(x):
            return sum(1 for y in nbrs[x] if _edge_key(x, y) in rem)
        cands.sort(key=lambda v: (rem_degree(v), v))
        for v in cands:
            path.append(v)
            visited.add(v)
            if _backtrack(path, visited, start, deadline_t, rem):
                return True
            path.pop()
            visited.remove(v)
        return False

    # Find up to 3 cycles
    max_cycles = 3
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        cyc = find_one_hamiltonian(deadline)
        if cyc is None:
            break
        # Record edges of this cycle
        cyc_edges = set()
        for i in range(len(cyc)):
            u = cyc[i]
            v = cyc[(i + 1) % len(cyc)]
            cyc_edges.add(_edge_key(u, v))
        # Ensure no overlap (should be guaranteed since we only use remaining)
        found.append(cyc)
        used_edges |= cyc_edges
        remaining -= cyc_edges

    # Validate and convert to plain int lists
    result = []
    for cyc in found:
        result.append([int(x) for x in cyc])

    # Fallback: if we got nothing, return a single valid Hamiltonian cycle if one is known/quick
    if not result:
        # Try a deterministic quick attempt with a fresh deadline slice
        if time.time() < deadline:
            used_edges = set()
            remaining = set(all_edges)
            cyc = find_one_hamiltonian(deadline)
            if cyc is not None:
                result = [[int(x) for x in cyc]]

    return result
