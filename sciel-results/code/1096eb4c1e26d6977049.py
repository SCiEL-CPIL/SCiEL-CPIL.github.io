"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

def _build_edge_index():
    n = P['n']
    edge_idx = {}
    for i, (u, v) in enumerate(P['edge_order']):
        edge_idx[(min(u,v), max(u,v))] = i
    return edge_idx

def _cycle_score(cycle, edge_idx):
    n = len(cycle)
    max_idx = 0
    for i in range(n):
        u, v = cycle[i], cycle[(i+1) % n]
        key = (min(u,v), max(u,v))
        idx = edge_idx[key]
        if idx > max_idx:
            max_idx = idx
    return max_idx

def _find_hamilton_cycle_k(k, deadline):
    """Find a Hamilton cycle using only edges with 0-based index < k.
    Returns the cycle as a list of vertices, or None if not found."""
    n = P['n']
    
    # Build adjacency as bitmasks
    adj = [0] * n
    for i in range(min(k, len(P['edge_order']))):
        u, v = P['edge_order'][i]
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Check degree constraint: every vertex must have degree >= 2
    for i in range(n):
        if bin(adj[i]).count('1') < 2:
            return None
    
    # Fix vertex 0 as start. Try each neighbor as second vertex.
    # Use iterative deepening DFS with strong pruning.
    
    all_mask = (1 << n) - 1
    
    # Precompute: for each vertex, its neighbors as a list
    adj_lists = []
    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj[i] & (1 << j):
                neighbors.append(j)
        adj_lists.append(neighbors)
    
    # Sort neighbors by degree (ascending) for better pruning
    for i in range(n):
        adj_lists[i].sort(key=lambda x: bin(adj[x]).count('1'))
    
    best_cycle = [None]
    
    def dfs(path, visited_mask, depth):
        if time.time() > deadline:
            return False
        
        if depth == n:
            last = path[-1]
            if adj[last] & (1 << 0):
                best_cycle[0] = path[:]
                return True
            return False
        
        last = path[-1]
        
        # Pruning: check if any unvisited vertex has no unvisited neighbors
        # (excluding the current last vertex, which can still connect)
        unvisited = all_mask & ~visited_mask
        for v in range(n):
            if unvisited & (1 << v):
                # Vertex v is unvisited
                # It needs at least 2 neighbors in (visited ∪ unvisited) 
                # but more importantly, it needs to be reachable
                # Simple check: does it have any unvisited neighbor or is it adjacent to last?
                v_neighbors = adj[v]
                # Count neighbors that are either unvisited or equal to last
                available = v_neighbors & (unvisited | (1 << last))
                if available == 0:
                    return False
        
        # Try each unvisited neighbor of last
        for nxt in adj_lists[last]:
            if visited_mask & (1 << nxt):
                continue
            # Additional pruning: if nxt is the only unvisited neighbor of some vertex, 
            # and that vertex isn't last, we might get stuck
            visited_mask |= (1 << nxt)
            path.append(nxt)
            if dfs(path, visited_mask, depth + 1):
                return True
            path.pop()
            visited_mask &= ~(1 << nxt)
        
        return False
    
    # Try each neighbor of 0 as the second vertex
    for second in adj_lists[0]:
        if time.time() > deadline:
            break
        path = [0, second]
        visited_mask = (1 << 0) | (1 << second)
        if dfs(path, visited_mask, 2):
            return best_cycle[0]
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin
    
    edge_idx = _build_edge_index()
    n = P['n']
    total_edges = n * (n - 1) // 2  # 276
    
    # Binary search on k (1-based max arrival index)
    lo = n  # minimum possible: need at least n edges
    hi = total_edges
    
    best_cycle = None
    best_score = float('inf')
    
    # First, try to find the minimum k via binary search
    while lo < hi and time.time() < deadline:
        mid = (lo + hi) // 2
        cycle = _find_hamilton_cycle_k(mid, deadline)
        if cycle is not None:
            hi = mid
            if best_cycle is None or _cycle_score(cycle, edge_idx) < best_score:
                best_score = _cycle_score(cycle, edge_idx)
                best_cycle = cycle[:]
        else:
            lo = mid + 1
    
    # Now lo == hi is the minimum k. Let's verify and get the cycle.
    if time.time() < deadline - 1.0:
        cycle = _find_hamilton_cycle_k(lo, deadline)
        if cycle is not None:
            sc = _cycle_score(cycle, edge_idx)
            if best_cycle is None or sc < best_score:
                best_score = sc
                best_cycle = cycle[:]
    
    # Fallback if no cycle found
    if best_cycle is None:
        # Try with all edges
        cycle = _find_hamilton_cycle_k(total_edges, deadline)
        if cycle is not None:
            best_cycle = cycle[:]
            best_score = _cycle_score(cycle, edge_idx)
        else:
            # Last resort: return identity (shouldn't happen)
            best_cycle = list(range(n))
            best_score = _cycle_score(best_cycle, edge_idx)
    
    # Try to improve with 2-opt style local search
    if time.time() < deadline - 2.0:
        cycle = best_cycle[:]
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for i in range(n):
                if time.time() > deadline - 2.0:
                    break
                for j in range(i+1, n):
                    if time.time() > deadline - 2.0:
                        break
                    # Reverse segment i..j
                    new_cycle = cycle[:i] + cycle[i:j+1][::-1] + cycle[j+1:]
                    sc = _cycle_score(new_cycle, edge_idx)
                    if sc < best_score:
                        best_score = sc
                        best_cycle = new_cycle[:]
                        cycle = new_cycle[:]
                        improved = True
    
    return best_cycle
# EVOLVE-BLOCK-END
