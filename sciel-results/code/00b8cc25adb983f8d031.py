"""Find Hamilton cycle with minimal max edge arrival index."""
import time
import sys
import random
from collections import defaultdict

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions (1-indexed)
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

# Build adjacency lists sorted by edge position
adj_with_pos = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    adj_with_pos[u].append((v, i))
    adj_with_pos[v].append((u, i))
for v in range(N):
    adj_with_pos[v].sort(key=lambda x: x[1])

MAX_K = len(EDGE_ORDER)

# Precompute connectivity and minimum degree threshold
def _min_k_for_connectivity():
    """Find the smallest K where all vertices have degree >= 2 and graph is connected."""
    for K in range(N, MAX_K + 1):
        # Check degree >= 2 for all
        deg = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            deg[u] += 1
            deg[v] += 1
        if any(d < 2 for d in deg):
            continue
        # Check connectivity
        masks = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        visited = [False] * N
        stack = [0]
        visited[0] = True
        count = 1
        while stack:
            u = stack.pop()
            neighbors = masks[u]
            while neighbors:
                lsb = neighbors & (-neighbors)
                nb = lsb.bit_length() - 1
                neighbors ^= lsb
                if not visited[nb]:
                    visited[nb] = True
                    count += 1
                    stack.append(nb)
        if count == N:
            return K
    return MAX_K

K_START = _min_k_for_connectivity()

def _connected_in_prefix(K, start_vertex=0):
    """Check if prefix K is connected from start_vertex."""
    masks = [0] * N
    for i in range(K):
        u, v = EDGE_ORDER[i]
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    visited = [False] * N
    stack = [start_vertex]
    visited[start_vertex] = True
    count = 1
    while stack:
        u = stack.pop()
        neighbors = masks[u]
        while neighbors:
            lsb = neighbors & (-neighbors)
            nb = lsb.bit_length() - 1
            neighbors ^= lsb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                stack.append(nb)
    return count == N


def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin out of 90s

    sys.setrecursionlimit(100000)

    best_cycle = None
    best_K = MAX_K + 1

    # Build masks for a given K efficiently
    def build_masks(K):
        masks = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
        return masks

    # DFS with Warnsdorff heuristic and restarts
    def find_cycle_in_prefix(K, max_restarts=50):
        masks = build_masks(K)
        
        # Quick feasibility: degree >= 2
        for v in range(N):
            if bin(masks[v]).count('1') < 2:
                return None
        
        # Quick feasibility: connectivity
        if not _connected_in_prefix(K, 0):
            return None

        def dfs_warnsdorff(v, visited_mask, path, rng):
            nonlocal best_cycle, best_K
            if time.time() > deadline - 3:
                return None
            
            if len(path) == N:
                # Check closing edge
                if masks[v] & (1 << 0):
                    cycle = path[:]
                    score = max(edge_pos[(path[i], path[i+1])] for i in range(N-1))
                    score = max(score, edge_pos[(path[N-1], 0)])
                    if score < best_K:
                        best_K = score
                        best_cycle = cycle[:]
                    return cycle
                return None
            
            available = masks[v] & ~visited_mask
            if available == 0:
                return None
            
            neighbors = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                nb = lsb.bit_length() - 1
                temp ^= lsb
                
                if len(path) == N - 1:
                    # Must go to 0 to close the cycle
                    if masks[nb] & (1 << 0):
                        neighbors.append((0, nb))
                else:
                    onward = masks[nb] & ~visited_mask & ~(1 << v)
                    deg = bin(onward).count('1')
                    neighbors.append((deg, nb))
            
            if not neighbors:
                return None
            
            # Sort by onward degree (Warnsdorff)
            neighbors.sort(key=lambda x: x[0])
            
            # Group by degree and shuffle within groups for diversity
            deg_groups = defaultdict(list)
            for deg, nb in neighbors:
                deg_groups[deg].append(nb)
            
            for deg in sorted(deg_groups.keys()):
                nbs = deg_groups[deg]
                rng.shuffle(nbs)
                for nb in nbs:
                    result = dfs_warnsdorff(nb, visited_mask | (1 << nb), path + [nb], rng)
                    if result is not None:
                        return result
            return None

        for attempt in range(max_restarts):
            if time.time() > deadline - 3:
                break
            rng = random.Random(attempt * 7919 + 13)
            cycle = dfs_warnsdorff(0, 1 << 0, [0], rng)
            if cycle is not None:
                return cycle
        return None

    # Binary search for minimum feasible K
    lo = K_START
    hi = MAX_K
    
    # First, find a feasible upper bound
    while hi > lo and time.time() < deadline - 20:
        mid = (lo + hi) // 2
        if mid < K_START:
            lo = K_START
            continue
        
        if not _connected_in_prefix(mid, 0):
            lo = mid + 1
            continue
        
        # Check degree constraint
        deg = [0] * N
        for i in range(mid):
            u, v = EDGE_ORDER[i]
            deg[u] += 1
            deg[v] += 1
        if any(d < 2 for d in deg):
            lo = mid + 1
            continue
        
        cycle = find_cycle_in_prefix(mid, max_restarts=30)
        if cycle is not None:
            hi = mid
        else:
            lo = mid + 1
    
    # Try the boundary and nearby values with more effort
    for K in range(max(lo, K_START), min(hi + 3, MAX_K + 1)):
        if time.time() > deadline - 10:
            break
        if best_cycle is not None and K >= best_K:
            continue
        
        # Check degree
        deg = [0] * N
        for i in range(K):
            u, v = EDGE_ORDER[i]
            deg[u] += 1
            deg[v] += 1
        if any(d < 2 for d in deg):
            continue
        
        if not _connected_in_prefix(K, 0):
            continue
        
        cycle = find_cycle_in_prefix(K, max_restarts=80)
        if cycle is not None and (best_cycle is None or K < best_K):
            best_cycle = cycle
            best_K = K

    # If still nothing, try with larger restarts around K_START
    if best_cycle is None:
        for K in range(K_START, min(K_START + 10, MAX_K + 1)):
            if time.time() > deadline - 5:
                break
            deg = [0] * N
            for i in range(K):
                u, v = EDGE_ORDER[i]
                deg[u] += 1
                deg[v] += 1
            if any(d < 2 for d in deg):
                continue
            if not _connected_in_prefix(K, 0):
                continue
            cycle = find_cycle_in_prefix(K, max_restarts=150)
            if cycle is not None:
                best_cycle = cycle
                best_K = K
                break

    if best_cycle is not None:
        return best_cycle

    # Final fallback: greedy construction
    return list(range(N))
