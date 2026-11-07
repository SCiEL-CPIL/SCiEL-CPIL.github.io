"""Combined approach: backtracking with maintained union-find per color pair,
proper backtracking via save/restore, symmetry breaking, and degree-based ordering."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

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


def _find_coloring_with_k_colors(k, edges, n, deadline):
    """Find valid coloring with k colors using backtracking with maintained UF per pair."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Order vertices: higher degree first, with tie-breaking by index
    order = sorted(range(n), key=lambda x: (-len(adj[x]), x))
    pos_to_vertex = order
    
    coloring = [-1] * n
    
    # For each color pair, maintain a union-find as a list (size n) for speed
    # pair_uf[pair] = (parent_list, rank_list)
    # But we need backtracking, so we save snapshots of affected entries
    
    # Alternative: use a dict-based UF per pair, and save/restore affected vertices
    # pair_uf: dict from (c1,c2) -> dict mapping vertex -> parent
    # pair_rank: dict from (c1,c2) -> dict mapping vertex -> rank
    
    pair_uf = {}
    pair_rank = {}
    
    def get_uf(c1, c2):
        pair = (min(c1, c2), max(c1, c2))
        if pair not in pair_uf:
            pair_uf[pair] = {}
            pair_rank[pair] = {}
        return pair
    
    def uf_find(pair, x):
        uf = pair_uf[pair]
        if x not in uf:
            uf[x] = x
            pair_rank[pair][x] = 0
        root = x
        while uf[root] != root:
            root = uf[root]
        # Path compression
        while uf[x] != root:
            parent = uf[x]
            uf[x] = root
            x = parent
        return root
    
    def uf_union(pair, x, y):
        """Union x and y in the UF for this pair. Returns False if cycle detected.
        Saves state for backtracking."""
        uf = pair_uf[pair]
        rank = pair_rank[pair]
        
        # Ensure x and y are in the UF
        if x not in uf:
            uf[x] = x
            rank[x] = 0
        if y not in uf:
            uf[y] = y
            rank[y] = 0
        
        rx = uf_find(pair, x)
        ry = uf_find(pair, y)
        
        if rx == ry:
            return False  # Cycle!
        
        # Union by rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        uf[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True
    
    # For backtracking, we need to undo UF unions.
    # Strategy: save the state of affected UF entries before each assignment,
    # restore after backtracking.
    
    # Actually, a cleaner approach: since k is small, the number of pairs is small.
    # We can save the entire UF state for affected pairs before assigning,
    # and restore after. But that's expensive.
    
    # Better: use a "log" of changes. Each union modifies at most 2 entries in uf
    # and possibly 1 in rank. We can log these and undo them.
    
    # Let's use a different strategy: maintain the UF, and for backtracking,
    # we save the parent and rank of the specific vertices involved.
    
    # Actually, the simplest correct approach for backtracking:
    # Before assigning vertex v with color c, save the state of all UF pairs
    # that will be affected (pairs involving c). After backtracking, restore.
    
    # For efficiency, let's save only the entries that are modified.
    
    # Let me use a simpler but still efficient approach:
    # At each backtracking step, we save a snapshot of the relevant UF states.
    
    # Given the small size, let's just save/restore the entire pair_uf and pair_rank
    # for the affected pairs.
    
    def save_uf_state():
        """Save current UF state for all pairs."""
        return {pair: (dict(pair_uf[pair]), dict(pair_rank[pair])) for pair in pair_uf}
    
    def restore_uf_state(saved):
        """Restore UF state from saved snapshot."""
        for pair in list(pair_uf.keys()):
            if pair in saved:
                pair_uf[pair] = saved[pair][0]
                pair_rank[pair] = saved[pair][1]
            else:
                del pair_uf[pair]
                if pair in pair_rank:
                    del pair_rank[pair]
        for pair, (uf, rank) in saved.items():
            if pair not in pair_uf:
                pair_uf[pair] = uf
                pair_rank[pair] = rank
    
    best_coloring = [None]
    nodes_visited = [0]
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        nodes_visited[0] += 1
        
        if pos == n:
            best_coloring[0] = coloring[:]
            return True
        
        v = pos_to_vertex[pos]
        
        # Determine which colors are allowed (adjacency constraint)
        # and try them in order
        # Symmetry breaking: for the first vertex in order, only try color 0
        # For subsequent vertices, we can restrict colors based on what's already used
        # Simple symmetry breaking: the first vertex must be color 0
        
        if pos == 0:
            colors_to_try = [0]
        else:
            colors_to_try = list(range(k))
        
        for c in colors_to_try:
            # Check adjacency constraint
            valid = True
            for nb in adj[v]:
                if coloring[nb] == c:
                    valid = False
                    break
            if not valid:
                continue
            
            # Save UF state before assignment
            saved_state = save_uf_state()
            
            coloring[v] = c
            
            # Check bicolored cycles for each colored neighbor
            cycle_found = False
            for nb in adj[v]:
                if coloring[nb] == -1:
                    continue
                d = coloring[nb]
                if d == c:
                    cycle_found = True
                    break
                pair = get_uf(c, d)
                if not uf_union(pair, v, nb):
                    cycle_found = True
                    break
            
            if not cycle_found:
                if backtrack(pos + 1):
                    return True
            
            # Restore UF state
            restore_uf_state(saved_state)
            coloring[v] = -1
        
        return False
    
    if backtrack(0):
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
    # Greedy approach: assign colors to avoid bicolored cycles
    # With enough colors, this is always possible
    coloring = [0] * n
    for i in range(n):
        used = set()
        for nb in P['edges']:
            if nb[0] == i:
                used.add(coloring[nb[1]])
            elif nb[1] == i:
                used.add(coloring[nb[0]])
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
