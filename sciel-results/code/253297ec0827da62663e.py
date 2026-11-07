"""Solver for few-color graph coloring with no bicolored cycles."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles(coloring, adj):
    """Check if any pair of colors induces a cycle. Returns True if valid (no bicolored cycles)."""
    n = len(coloring)
    # Group edges by color pair
    from collections import defaultdict
    # For each pair of colors, check if the subgraph is a forest
    # A subgraph is a forest iff for every connected component, edges = vertices - 1
    # More simply: use union-find per color pair
    color_pairs = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                cu, cv = coloring[u], coloring[v]
                if cu == cv:
                    return False  # adjacent same color
                color_pairs.add((min(cu, cv), max(cu, cv)))
    
    for c1, c2 in color_pairs:
        # Union-find for vertices with color c1 or c2, edges between them
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False  # cycle detected
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u in range(n):
            if coloring[u] not in (c1, c2):
                continue
            for v in adj[u]:
                if u < v and coloring[v] in (c1, c2):
                    if not union(u, v):
                        return False  # bicolored cycle
    return True

def _backtrack_coloring(adj, num_colors, deadline):
    """Backtracking to find valid coloring with exactly num_colors colors."""
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices by degree descending for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    # Map position in order to vertex
    order_pos = [0] * n
    for i, v in enumerate(order):
        order_pos[v] = i
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        v = order[idx]
        # Try colors in order
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            coloring[v] = c
            # Check bicolored cycles involving v
            # Only need to check pairs (c, c') where c' is a neighbor's color
            valid = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1:
                    continue
                # Check if adding edge (v, nb) creates a cycle in subgraph of colors {c, cn}
                # Use union-find on the subgraph induced by colors c and cn
                # Quick check: find if v and nb are already connected in the {c,cn} subgraph
                # We do a BFS/DFS from v in the {c,cn} subgraph to see if we reach nb
                # But this is expensive; instead use incremental union-find per pair
                
                # For efficiency, we'll do a full check only when needed
                # Actually, let's do a targeted check: in the subgraph of vertices colored c or cn,
                # is there already a path from v to nb not using edge (v,nb)?
                # If yes, then adding (v,nb) creates a cycle.
                
                # BFS from v in {c,cn} subgraph, avoiding edge to nb directly
                visited = set()
                stack = [v]
                visited.add(v)
                found = False
                while stack:
                    cur = stack.pop()
                    for nb2 in adj[cur]:
                        if nb2 == nb and cur == v:
                            continue  # skip the edge we're adding
                        if nb2 == nb and cur != v:
                            # We reached nb through another path
                            found = True
                            break
                        if coloring[nb2] in (c, cn) and nb2 not in visited:
                            visited.add(nb2)
                            stack.append(nb2)
                    if found:
                        break
                if found:
                    valid = False
                    break
            if valid:
                result = dfs(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        return None
    
    return dfs(0)

def construct():
    deadline = time.time() + 80  # safety margin
    adj = _build_adj()
    
    # Try 3 colors first (minimum possible since graph has cycles)
    for nc in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            # Verify
            assert _check_bicolored_cycles(result, adj)
            return [int(x) for x in result]
    
    # Fallback: use all 50 colors (always valid since no two adjacent share a color,
    # and with all distinct colors, no color pair has more than one edge)
    return list(range(P['n']))
