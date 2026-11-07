"""Combined approach: constraint propagation with forward checking,
smart variable ordering, and incremental bicolored cycle detection via BFS."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _check_coloring(coloring, edges):
    """Verify coloring: adjacent differ, no bicolored cycle (each pair subgraph is forest)."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    for pair, elist in pair_edges.items():
        parent = {}
        def find(x):
            if x not in parent:
                parent[x] = x
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True


def _has_bicolored_cycle(v, c, coloring, adj, edges):
    """Check if assigning color c to vertex v creates a bicolored cycle.
    Only check edges from v to already-colored neighbors.
    For each such edge (v, nb), check if v and nb are connected in the
    bicolored subgraph of colors {c, coloring[nb]} without using edge (v, nb).
    """
    for nb in adj[v]:
        if coloring[nb] == -1:
            continue
        d = coloring[nb]
        if d == c:
            return True  # adjacent same color
        # Check if there's a path from v to nb in subgraph of colors {c, d}
        # excluding the direct edge (v, nb)
        c1, c2 = (c, d) if c < d else (d, c)
        # BFS from v to nb in subgraph induced by colors c1, c2
        # Exclude the edge (v, nb) by not traversing from v to nb directly
        visited = set()
        queue = deque()
        queue.append(v)
        visited.add(v)
        found = False
        while queue:
            curr = queue.popleft()
            if curr == nb:
                found = True
                break
            for nb2 in adj[curr]:
                if nb2 in visited:
                    continue
                if nb2 == nb and curr == v:
                    continue  # Skip the direct edge we're checking
                cc = coloring[nb2]
                if cc == -1:
                    continue
                if cc == c1 or cc == c2:
                    visited.add(nb2)
                    queue.append(nb2)
        if found:
            return True  # Cycle detected
    return False


def _find_coloring_with_k_colors(k, edges, n, deadline):
    """Find valid coloring with k colors using backtracking with forward checking."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices: higher degree first
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    pos_to_vertex = order
    vertex_to_pos = {v: i for i, v in enumerate(order)}
    
    coloring = [-1] * n
    domains = [set(range(k)) for _ in range(n)]
    
    # Symmetry breaking: first vertex in order must be color 0
    domains[order[0]] = {0}
    
    best_coloring = [None]
    nodes_visited = [0]
    
    def get_unassigned():
        """Get unassigned vertices, ordered by domain size (MRV heuristic)."""
        unassigned = []
        for v in range(n):
            if coloring[v] == -1:
                unassigned.append(v)
        # Sort by domain size, then by degree
        unassigned.sort(key=lambda v: (len(domains[v]), -len(adj[v]), v))
        return unassigned
    
    def propagate(v, c):
        """Assign color c to v, propagate constraints. Returns (success, assignments_made)."""
        assignments = [(v, c)]
        coloring[v] = c
        domains[v] = {c}
        
        # Check bicolored cycle
        if _has_bicolored_cycle(v, c, coloring, adj, edges):
            return False, assignments
        
        # Propagate to neighbors
        for nb in adj[v]:
            if coloring[nb] == -1:
                if c in domains[nb]:
                    domains[nb].discard(c)
                    if not domains[nb]:
                        return False, assignments
                    # If only one option left, assign it
                    if len(domains[nb]) == 1:
                        nc = next(iter(domains[nb]))
                        if nc == c:
                            return False, assignments
                        # Recursively propagate
                        ok, sub_assignments = propagate(nb, nc)
                        if not ok:
                            return False, assignments
                        assignments.extend(sub_assignments)
        
        return True, assignments
    
    def unpropagate(assignments):
        """Undo assignments."""
        for v, c in reversed(assignments):
            coloring[v] = -1
            # Rebuild domain
            domains[v] = set(range(k))
            for nb in adj[v]:
                if coloring[nb] != -1:
                    domains[v].discard(coloring[nb])
    
    def backtrack():
        if time.time() > deadline:
            return False
        nodes_visited[0] += 1
        
        unassigned = get_unassigned()
        if not unassigned:
            best_coloring[0] = coloring[:]
            return True
        
        v = unassigned[0]
        
        # Try colors in order of least constraining value
        colors = sorted(domains[v], key=lambda c: -sum(1 for nb in adj[v] if coloring[nb] == -1 and c in domains[nb]))
        
        for c in colors:
            ok, assignments = propagate(v, c)
            if ok:
                if backtrack():
                    return True
                unpropagate(assignments)
        
        return False
    
    if backtrack():
        return best_coloring[0]
    return None


def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Try k = 2, 3, 4, ... until we find a valid coloring
    for k in range(2, 12):
        if time.time() > deadline - 10:
            break
        result = _find_coloring_with_k_colors(k, edges, n, deadline)
        if result is not None:
            # Verify
            if _check_coloring(result, edges):
                return [int(c) for c in result]
    
    # Fallback: return a valid coloring with more colors
    coloring = [0] * n
    for i in range(n):
        used = set()
        for u, v in P['edges']:
            if u == i:
                used.add(coloring[v])
            elif v == i:
                used.add(coloring[u])
        for c in range(50):
            if c not in used:
                coloring[i] = c
                break
    
    # Verify and return
    if _check_coloring(coloring, edges):
        return [int(c) for c in coloring]
    
    # Ultimate fallback
    return list(range(n))
# EVOLVE-BLOCK-END
