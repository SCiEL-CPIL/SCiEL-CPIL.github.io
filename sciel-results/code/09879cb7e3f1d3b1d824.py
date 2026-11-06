"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

# Precompute edge position (1-based index)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER):
    key = (min(u, v), max(u, v))
    edge_pos[key] = i + 1

def build_adj(K):
    adj = [[] for _ in range(N)]
    for i in range(K):
        u, v = EDGE_ORDER[i]
        adj[u].append(v)
        adj[v].append(u)
    return adj

def check_graph_properties(adj):
    for i in range(N):
        if len(adj[i]) < 2:
            return False
    # Connectivity check
    visited = [False] * N
    stack = [0]
    visited[0] = True
    count = 1
    while stack:
        node = stack.pop()
        for nb in adj[node]:
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    return count == N

def find_hamilton_cycle(K, time_limit=5.0, seed=42):
    start_time = time.time()
    adj = build_adj(K)
    if not check_graph_properties(adj):
        return None
    
    path = [0]
    visited = [False] * N
    visited[0] = True
    rng = random.Random(seed)
    
    def dfs():
        if time.time() - start_time > time_limit:
            return None
        if len(path) == N:
            last = path[-1]
            if 0 in adj[last]:
                return path[:]
            return None
        
        last = path[-1]
        candidates = []
        for nb in adj[last]:
            if not visited[nb]:
                unvisited_count = sum(1 for nnn in adj[nb] if not visited[nnn])
                if len(path) == N - 1:
                    if 0 in adj[nb]:
                        candidates.append((0, 0.0, nb))
                else:
                    # Add small random noise to break ties
                    candidates.append((unvisited_count, rng.random(), nb))
        
        if not candidates:
            return None
            
        candidates.sort(key=lambda x: (x[0], x[1]))
        
        for _, _, nb in candidates:
            path.append(nb)
            visited[nb] = True
            result = dfs()
            if result is not None:
                return result
            path.pop()
            visited[nb] = False
            
        return None

    return dfs()

def get_max_edge_index(cycle):
    max_k = 0
    for i in range(N):
        u = cycle[i]
        v = cycle[(i+1)%N]
        k = edge_pos[(min(u,v), max(u,v))]
        if k > max_k:
            max_k = k
    return max_k

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin
    
    # 1. Find an upper bound
    high = 276 # Max possible edges
    cycle = None
    # Try to find a cycle with a reasonably high K
    for K_test in [50, 80, 100, 150, 200, 276]:
        if time.time() > deadline - 10:
            break
        cycle = find_hamilton_cycle(K_test, time_limit=2.0)
        if cycle:
            high = K_test
            break
            
    if not cycle:
        # Fallback: return identity (invalid but prevents crash)
        return list(range(N))
        
    low = 24
    best_cycle = cycle
    best_K = get_max_edge_index(best_cycle)
    high = best_K
    
    # 2. Binary search for minimum K
    while low <= high and time.time() < deadline - 5:
        mid = (low + high) // 2
        
        # Try to find a cycle for mid
        # Use a tighter time limit for binary search steps
        c = find_hamilton_cycle(mid, time_limit=3.0)
        if c:
            max_k_c = get_max_edge_index(c)
            if max_k_c <= mid:
                best_cycle = c
                best_K = max_k_c
                high = mid - 1
            else:
                # Should not happen if adj is built correctly
                high = mid - 1
        else:
            low = mid + 1
            
    # 3. Final polish: Try to improve best_K by checking best_K - 1 with more time
    if time.time() < deadline - 2:
        target_K = best_K - 1
        if target_K >= 24:
            c = find_hamilton_cycle(target_K, time_limit=5.0)
            if c:
                max_k_c = get_max_edge_index(c)
                if max_k_c <= target_K:
                    best_cycle = c
                    best_K = max_k_c

    return best_cycle
# EVOLVE-BLOCK-END
