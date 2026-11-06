"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import heapq

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']
FULL_MASK = (1 << N) - 1

# Precompute edge position (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

# Build edge weight matrix: weight[u][v] = position of edge (u,v), or large if not present
# Actually all edges are present in the complete graph, just at different positions
edge_weight = [[0] * N for _ in range(N)]
for i in range(N):
    for j in range(N):
        if i == j:
            edge_weight[i][j] = 0
        else:
            key = (min(i, j), max(i, j))
            edge_weight[i][j] = edge_pos[key]

def dijkstra_bottleneck(start_vertex, deadline, max_states=5000000):
    """
    Dijkstra on state space (mask, last) with bottleneck cost.
    Cost of a state = max edge position used in the path.
    Returns (cycle, bottleneck_cost) or (None, None).
    """
    # State: (mask, last) -> (bottleneck_cost, path)
    # We use a priority queue ordered by bottleneck cost
    # best_cost[mask][last] = minimum bottleneck cost to reach this state
    
    # For memory efficiency with N=24, we use a dict for visited states
    # Key: (mask, last), Value: (cost, path_tuple)
    
    # Initialize
    init_mask = 1 << start_vertex
    init_state = (init_mask, start_vertex)
    
    # Priority queue: (cost, counter, mask, last, path)
    # counter for tie-breaking to avoid comparing paths
    counter = 0
    pq = [(0, counter, init_mask, start_vertex, (start_vertex,))]
    
    # Best known cost for each (mask, last)
    best_cost = {}
    best_cost[init_state] = 0
    
    # Store parent pointers for path reconstruction
    # parent[(mask, last)] = (prev_mask, prev_last)
    parent = {}
    
    states_explored = 0
    t0 = time.time()
    
    while pq:
        cost, _, mask, last, path = heapq.heappop(pq)
        
        # Time check
        if time.time() > deadline:
            return None, None
        
        states_explored += 1
        if states_explored > max_states:
            return None, None
        
        # If we've already found a better cost for this state, skip
        if (mask, last) in best_cost and best_cost[(mask, last)] < cost:
            continue
        
        # Check if we have a complete cycle
        if mask == FULL_MASK:
            # Check if last connects to start
            close_cost = edge_weight[last][start_vertex]
            total_cost = max(cost, close_cost)
            # Reconstruct path
            cycle = list(path)
            return cycle, total_cost
        
        # Expand
        for nxt in range(N):
            if mask & (1 << nxt):
                continue
            w = edge_weight[last][nxt]
            new_cost = max(cost, w)
            new_mask = mask | (1 << nxt)
            new_state = (new_mask, nxt)
            
            if new_state not in best_cost or best_cost[new_state] > new_cost:
                best_cost[new_state] = new_cost
                parent[new_state] = (mask, last)
                counter += 1
                new_path = path + (nxt,)
                heapq.heappush(pq, (new_cost, counter, new_mask, nxt, new_path))
    
    return None, None

def dijkstra_bottleneck_no_path(start_vertex, deadline, max_states=10000000):
    """
    Dijkstra on state space (mask, last) with bottleneck cost.
    Only stores parent pointers, not full paths. More memory efficient.
    Returns (cycle, bottleneck_cost) or (None, None).
    """
    init_mask = 1 << start_vertex
    init_state = (init_mask, start_vertex)
    
    counter = 0
    pq = [(0, counter, init_mask, start_vertex)]
    
    best_cost = {}
    best_cost[init_state] = 0
    parent = {}
    
    states_explored = 0
    
    while pq:
        cost, _, mask, last = heapq.heappop(pq)
        
        if time.time() > deadline:
            return None, None
        
        states_explored += 1
        if states_explored > max_states:
            return None, None
        
        if (mask, last) in best_cost and best_cost[(mask, last)] < cost:
            continue
        
        if mask == FULL_MASK:
            close_cost = edge_weight[last][start_vertex]
            total_cost = max(cost, close_cost)
            # Reconstruct path
            path = []
            state = (mask, last)
            while state is not None:
                path.append(state[1])
                if state in parent:
                    state = parent[state]
                else:
                    break
            path.reverse()
            return path, total_cost
        
        for nxt in range(N):
            if mask & (1 << nxt):
                continue
            w = edge_weight[last][nxt]
            new_cost = max(cost, w)
            new_mask = mask | (1 << nxt)
            new_state = (new_mask, nxt)
            
            if new_state not in best_cost or best_cost[new_state] > new_cost:
                best_cost[new_state] = new_cost
                parent[new_state] = (mask, last)
                counter += 1
                heapq.heappush(pq, (new_cost, counter, new_mask, nxt))
    
    return None, None

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    # Try Dijkstra with different start vertices
    best_cycle = None
    best_cost = float('inf')
    
    # Try start vertices in order of their "centrality" - vertices with more early edges
    # First, compute how many early edges each vertex has
    vertex_early_degree = [0] * N
    for i in range(min(50, len(EDGE_ORDER))):
        u, v = EDGE_ORDER[i]
        vertex_early_degree[u] += 1
        vertex_early_degree[v] += 1
    
    # Sort vertices by early degree (descending) - more connected vertices first
    start_order = sorted(range(N), key=lambda v: -vertex_early_degree[v])
    
    for sv in start_order:
        if time.time() > deadline - 3:
            break
        cycle, cost = dijkstra_bottleneck_no_path(sv, deadline - 3, max_states=3000000)
        if cycle is not None and cost < best_cost:
            best_cost = cost
            best_cycle = cycle
            # If we found a very good solution, we can stop early
            # The theoretical minimum is N (24), since we need at least N edges
            if best_cost <= N:
                break
    
    if best_cycle is None:
        # Fallback: use the old DFS approach
        # Build adjacency with all edges
        adj = [set() for _ in range(N)]
        for u, v in EDGE_ORDER:
            adj[u].add(v)
            adj[v].add(u)
        
        # Simple DFS to find any Hamilton cycle
        path = [0]
        visited = [False] * N
        visited[0] = True
        
        def dfs_fallback():
            if len(path) == N:
                if 0 in adj[path[-1]]:
                    return True
                return False
            last = path[-1]
            # Warnsdorff
            candidates = [(sum(1 for nb in adj[nxt] if not visited[nb]), nxt) for nxt in adj[last] if not visited[nxt]]
            candidates.sort()
            for _, nxt in candidates:
                path.append(nxt)
                visited[nxt] = True
                if dfs_fallback():
                    return True
                path.pop()
                visited[nxt] = False
            return False
        
        if dfs_fallback():
            best_cycle = path[:]
        else:
            best_cycle = list(range(N))
    
    return best_cycle
# EVOLVE-BLOCK-END
