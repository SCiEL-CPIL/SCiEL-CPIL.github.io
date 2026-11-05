"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
TOTAL_EDGES = len(EDGE_ORDER)

# Build edge index: for each edge (u,v) with u<v, store its 1-based arrival index
edge_index = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    a, b = (u, v) if u < v else (v, u)
    edge_index[(a, b)] = i

def build_adjacency(K):
    """Build adjacency list using edges 1..K."""
    adj = [set() for _ in range(N)]
    for i in range(1, K + 1):
        u, v = EDGE_ORDER[i - 1]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def is_connected(adj):
    """Check if graph is connected using BFS."""
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)
    return count == N

def has_articulation_points(adj):
    """Check if graph has articulation points using DFS."""
    n = len(adj)
    visited = [False] * n
    disc = [0] * n
    low = [0] * n
    parent = [-1] * n
    time = 0
    has_ap = False

    def dfs(u):
        nonlocal time, has_ap
        visited[u] = True
        disc[u] = low[u] = time
        time += 1
        children = 0
        for v in adj[u]:
            if not visited[v]:
                children += 1
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if parent[u] == -1 and children > 1:
                    has_ap = True
                if parent[u] != -1 and low[v] >= disc[u]:
                    has_ap = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for i in range(n):
        if not visited[i]:
            dfs(i)
    return has_ap

def get_degree_lower_bound():
    """Calculate the minimum K such that all vertices have degree >= 2."""
    edge_counts = [0] * N
    min_k = 0
    for i, (u, v) in enumerate(EDGE_ORDER, 1):
        edge_counts[u] += 1
        if edge_counts[u] == 2:
            min_k = max(min_k, i)
        edge_counts[v] += 1
        if edge_counts[v] == 2:
            min_k = max(min_k, i)
    return min_k

def has_hamilton_cycle(K, deadline):
    """Check if subgraph with edges 1..K has a Hamiltonian cycle."""
    adj = build_adjacency(K)
    
    # Check: every vertex must have degree >= 2
    for i in range(N):
        if len(adj[i]) < 2:
            return False
            
    # Check connectivity
    if not is_connected(adj):
        return False
        
    # Check for articulation points (Hamiltonian cycle implies 2-connectivity)
    if has_articulation_points(adj):
        return False
        
    # Use DFS with bitmask for visited vertices
    # Fix starting vertex as 0 to break symmetry
    start = 0
    full_mask = (1 << N) - 1
    
    # Memo: (vertex, mask) -> False if no Hamiltonian path from vertex visiting mask back to start
    failed = set()
    
    def dfs(v, mask):
        if time.time() > deadline:
            return False
        if mask == full_mask:
            return start in adj[v]
        state = (v, mask)
        if state in failed:
            return False
        
        # Sort neighbors by degree (ascending) to prune faster
        neighbors = sorted(adj[v], key=lambda x: len(adj[x]))
        
        for u in neighbors:
            if not (mask & (1 << u)):
                if dfs(u, mask | (1 << u)):
                    return True
        failed.add(state)
        return False
    
    # Start from vertex 0, having visited only 0
    return dfs(start, 1 << start)

def find_min_K():
    """Binary search for the minimal K such that edges 1..K contain a Hamiltonian cycle."""
    deadline = time.time() + 75  # 75 seconds, leaving 15s margin
    
    # Lower bound: at least N edges needed (a cycle has N edges)
    # Also, each vertex needs degree >= 2 in the subgraph
    lo = max(N, get_degree_lower_bound())
    hi = TOTAL_EDGES  # 276
    
    best_K = None
    
    # Binary search
    while lo <= hi:
        if time.time() > deadline:
            break
        mid = (lo + hi) // 2
        if has_hamilton_cycle(mid, deadline):
            best_K = mid
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_K is not None:
        return best_K
    return None

def construct():
    K = find_min_K()
    if K is not None:
        # Now we need to actually construct the Hamiltonian cycle
        # Find the actual cycle using DFS
        adj = build_adjacency(K)
        start = 0
        full_mask = (1 << N) - 1
        
        path = []
        
        def dfs(v, mask, path):
            if time.time() > deadline:
                return False
            path.append(v)
            if mask == full_mask:
                if start in adj[v]:
                    return True
            else:
                for u in sorted(adj[v]):
                    if not (mask & (1 << u)):
                        if dfs(u, mask | (1 << u), path):
                            return True
            path.pop()
            return False
        
        deadline = time.time() + 50
        if dfs(start, 1 << start, path):
            # path is the Hamiltonian cycle (vertices in order)
            return path
        else:
            # Fallback: return identity permutation
            return list(range(N))
    else:
        # Fallback
        return list(range(N))

# EVOLVE-BLOCK-END
