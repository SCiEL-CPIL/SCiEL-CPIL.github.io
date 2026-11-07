"""Solver for few-color graph coloring with no bicolored cycles."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_valid(coloring, adj):
    """Full validation: no adjacent same color, no bicolored cycles."""
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    
    color_pairs = {}
    for u in range(n):
        for v in adj[u]:
            if u < v:
                cu, cv = coloring[u], coloring[v]
                key = (min(cu, cv), max(cu, cv))
                if key not in color_pairs:
                    color_pairs[key] = []
                color_pairs[key].append((u, v))
    
    for (c1, c2), edges in color_pairs.items():
        vertices = set()
        for u, v in edges:
            vertices.add(u)
            vertices.add(v)
        
        parent = {v: v for v in vertices}
        rank = {v: 0 for v in vertices}
        
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        for u, v in edges:
            if not union(u, v):
                return False
    return True


def _solve_backtrack(adj, num_colors, deadline, order=None):
    """Backtracking with incremental union-find for bicolored cycle detection."""
    n = P['n']
    coloring = [-1] * n
    
    if order is None:
        order = list(range(n))
    
    # Precompute neighbor lists for fast access
    adj_set = [set(adj[i]) for i in range(n)]
    
    # Union-find state per color pair
    # We'll maintain a dictionary: (c1, c2) -> {parent, rank}
    uf_state = {}
    
    def uf_init(c1, c2):
        key = (min(c1, c2), max(c1, c2))
        if key not in uf_state:
            uf_state[key] = {'parent': {}, 'rank': {}, 'edges': 0}
        return uf_state[key]
    
    def uf_find(state, x):
        parent = state['parent']
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def uf_union(state, a, b):
        parent = state['parent']
        rank = state['rank']
        if a not in parent:
            parent[a] = a
            rank[a] = 0
        if b not in parent:
            parent[b] = b
            rank[b] = 0
        ra, rb = uf_find(state, a), uf_find(state, b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
        state['edges'] += 1
        return True
    
    def uf_rollback(state, edge_count_before):
        """Rollback union-find to a previous state. Simple approach: rebuild."""
        pass  # We'll use a different strategy: snapshot/restore
    
    def dfs(idx, deadline):
        if time.time() > deadline:
            return None
        
        if idx == n:
            return coloring[:]
        
        v = order[idx]
        
        # Get colors used by already-colored neighbors
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        # Try each color not used by neighbors
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            
            coloring[v] = c
            
            # Check for bicolored cycles by testing each colored neighbor
            valid = True
            snapshots = []  # (key, parent_snapshot, rank_snapshot, edges_snapshot)
            
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1:
                    continue
                
                key = (min(c, cn), max(c, cn))
                state = uf_init(c, cn)
                
                # Snapshot before union
                parent_snap = dict(state['parent'])
                rank_snap = dict(state['rank'])
                edges_snap = state['edges']
                snapshots.append((key, parent_snap, rank_snap, edges_snap))
                
                if not uf_union(state, v, nb):
                    valid = False
                    break
            
            if valid:
                result = dfs(idx + 1, deadline)
                if result is not None:
                    return result
            
            # Rollback
            for key, parent_snap, rank_snap, edges_snap in snapshots:
                state = uf_state[key]
                state['parent'] = parent_snap
                state['rank'] = rank_snap
                state['edges'] = edges_snap
            
            coloring[v] = -1
        
        return None
    
    return dfs(0, deadline)


def _get_smart_order(adj):
    """Get vertex ordering: most constrained first (highest degree, then by index)."""
    n = P['n']
    deg = [len(adj[i]) for i in range(n)]
    # Sort by degree descending, then by index for determinism
    order = sorted(range(n), key=lambda i: (-deg[i], i))
    return order


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    adj = _build_adj()
    
    # Try with increasing number of colors
    # 3 colors is likely the minimum for this degree-4 graph
    for nc in [3, 4, 5, 6]:
        if time.time() > deadline:
            break
        
        # Try with smart ordering first
        order = _get_smart_order(adj)
        result = _solve_backtrack(adj, nc, deadline, order)
        
        if result is not None:
            if _check_valid(result, adj):
                return [int(x) for x in result]
        
        # If smart order failed, try natural order
        if time.time() < deadline:
            order = list(range(P['n']))
            result = _solve_backtrack(adj, nc, deadline, order)
            if result is not None:
                if _check_valid(result, adj):
                    return [int(x) for x in result]
    
    # Fallback: use all distinct colors (always valid)
    return [int(i) for i in range(P['n'])]
