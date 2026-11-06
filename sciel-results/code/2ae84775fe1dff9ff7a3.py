"""Find Hamilton cycle with minimal max edge arrival index."""
import time
import sys
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

# Precompute edge positions
edge_pos = {}
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    edge_pos[(u, v)] = i
    edge_pos[(v, u)] = i

# Precompute adjacency with edge positions
adj_with_pos = [[] for _ in range(N)]
for i, (u, v) in enumerate(EDGE_ORDER, 1):
    adj_with_pos[u].append((v, i))
    adj_with_pos[v].append((u, i))
for v in range(N):
    adj_with_pos[v].sort(key=lambda x: x[1])


def cycle_score(cyc):
    """Compute the max edge position in a cycle."""
    max_pos = 0
    for i in range(len(cyc)):
        u = cyc[i]
        v = cyc[(i + 1) % len(cyc)]
        pos = edge_pos[(u, v)]
        if pos > max_pos:
            max_pos = pos
    return max_pos


def is_valid_cycle(cyc):
    """Check if cyc is a valid Hamilton cycle on N vertices."""
    if len(cyc) != N:
        return False
    if len(set(cyc)) != N:
        return False
    for i in range(N):
        u = cyc[i]
        v = cyc[(i + 1) % N]
        if (u, v) not in edge_pos:
            return False
    return True


def check_connectivity(masks):
    """Check if graph with given masks is connected."""
    visited = [False] * N
    queue = [0]
    visited[0] = True
    count = 1
    while queue:
        u = queue.pop(0)
        neighbors = masks[u]
        while neighbors:
            lsb = neighbors & (-neighbors)
            nb = lsb.bit_length() - 1
            neighbors ^= lsb
            if not visited[nb]:
                visited[nb] = True
                count += 1
                queue.append(nb)
    return count == N


def build_masks(K):
    """Build adjacency masks for graph with first K edges."""
    masks = [0] * N
    for i in range(K):
        u, v = EDGE_ORDER[i]
        masks[u] |= (1 << v)
        masks[v] |= (1 << u)
    return masks


def greedy_hamilton_cycle(masks, deadline):
    """Try to find a Hamilton cycle using greedy/Warnsdorff heuristic."""
    sys.setrecursionlimit(10000)
    
    def dfs(v, visited_mask, path, depth=0):
        if time.time() > deadline:
            return None
        if len(path) == N:
            if masks[v] & (1 << 0):
                return path[:]
            return None
        
        available = masks[v] & ~visited_mask
        if available == 0:
            return None
        
        # Warnsdorff: sort by onward degree
        neighbors = []
        temp = available
        while temp:
            lsb = temp & (-temp)
            nb = lsb.bit_length() - 1
            temp ^= lsb
            if len(path) == N - 1:
                onward = 1 if (masks[nb] & (1 << 0)) else 0
            else:
                onward = bin(masks[nb] & ~visited_mask & ~(1 << v)).count('1')
            neighbors.append((onward, nb))
        
        neighbors.sort(key=lambda x: x[0])
        
        for _, nb in neighbors:
            result = dfs(nb, visited_mask | (1 << nb), path + [nb], depth + 1)
            if result is not None:
                return result
        return None
    
    # Try from vertex 0
    result = dfs(0, 1 << 0, [0])
    if result is not None:
        return result
    
    # Try from other vertices if needed
    for start in range(1, N):
        if time.time() > deadline:
            break
        result = dfs(start, 1 << start, [start])
        if result is not None:
            return result
    
    return None


def ilp_hamilton_cycle(K, deadline):
    """Find Hamiltonian cycle in G_K using ILP with subtour elimination."""
    active_edges = [(u, v) for i, (u, v) in enumerate(EDGE_ORDER) if i < K]
    M = len(active_edges)
    if M < N:
        return None
    
    # Build edge index map
    edge_idx = {}
    for j, (u, v) in enumerate(active_edges):
        edge_idx[(min(u, v), max(u, v))] = j
    
    # Degree constraints: each vertex has degree 2
    A_deg = np.zeros((N, M))
    for j, (u, v) in enumerate(active_edges):
        A_deg[u, j] = 1
        A_deg[v, j] = 1
    
    c = np.zeros(M)
    
    # Start with just degree constraints
    constraints_list = [LinearConstraint(A_deg, 2, 2)]
    bounds = Bounds(lb=np.zeros(M), ub=np.ones(M))
    integrality = np.ones(M)
    
    max_iter = 20
    for iteration in range(max_iter):
        if time.time() > deadline:
            return None
        
        try:
            res = milp(c=c, constraints=constraints_list, integrality=integrality,
                       bounds=bounds, options={'time_limit': 5, 'presolve': True})
        except Exception:
            return None
        
        if res.status != 0:
            return None
        
        x = res.x
        selected = [(u, v) for j, (u, v) in enumerate(active_edges) if x[j] > 0.5]
        
        # Extract cycles
        adj = {i: [] for i in range(N)}
        for u, v in selected:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = set()
        cycles = []
        for i in range(N):
            if i not in visited:
                cycle = []
                curr = i
                prev = -1
                while True:
                    visited.add(curr)
                    cycle.append(curr)
                    next_node = -1
                    for nb in adj[curr]:
                        if nb != prev:
                            next_node = nb
                            break
                    if next_node == -1 or next_node == i:
                        break
                    prev, curr = curr, next_node
                if len(cycle) > 1:
                    cycles.append(cycle)
        
        if len(cycles) == 1 and len(cycles[0]) == N:
            return cycles[0]
        
        # Add subtour elimination constraint for the smallest cycle
        if not cycles:
            return None
        
        # Find the smallest cycle (not the full cycle)
        subtour = None
        for cyc in cycles:
            if len(cyc) < N:
                subtour = cyc
                break
        
        if subtour is None:
            # All cycles are full length but there are multiple? That can't happen with degree 2.
            return None
        
        # DJ constraint: for subtour S, sum of edges with both endpoints in S <= |S| - 1
        S_set = set(subtour)
        A_sub = np.zeros(M)
        for j, (u, v) in enumerate(active_edges):
            if u in S_set and v in S_set:
                A_sub[j] = 1
        
        constraints_list.append(LinearConstraint(A_sub.reshape(1, -1), -np.inf, len(subtour) - 1))
    
    return None


def two_opt_polish(cycle, deadline):
    """Apply exhaustive best-improvement 2-opt to minimize max edge position."""
    if cycle is None:
        return None
    
    best_cycle = cycle[:]
    best_score = cycle_score(best_cycle)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(best_cycle)
        best_move = None
        best_move_score = best_score
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                # Reverse segment from i+1 to j
                new_cycle = best_cycle[:i+1] + best_cycle[i+1:j+1][::-1] + best_cycle[j+1:]
                new_score = cycle_score(new_cycle)
                if new_score < best_move_score:
                    best_move_score = new_score
                    best_move = new_cycle
        
        if best_move is not None:
            best_cycle = best_move
            best_score = best_move_score
            improved = True
    
    return best_cycle


def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    best_cycle = None
    best_K = len(EDGE_ORDER)
    
    # Binary search for minimum K
    lo, hi = N, len(EDGE_ORDER)
    
    while lo <= hi and time.time() < deadline - 20:
        mid = (lo + hi) // 2
        masks = build_masks(mid)
        
        # Quick checks
        valid = True
        for v in range(N):
            if bin(masks[v]).count('1') < 2:
                valid = False
                break
        if not valid:
            lo = mid + 1
            continue
        
        if not check_connectivity(masks):
            lo = mid + 1
            continue
        
        # Try greedy first (fast)
        cycle = greedy_hamilton_cycle(masks, time.time() + 2)
        
        # If greedy failed, try ILP
        if cycle is None and time.time() < deadline - 10:
            cycle = ilp_hamilton_cycle(mid, time.time() + 10)
        
        if cycle is not None and is_valid_cycle(cycle):
            # Check if this cycle's max edge is within mid
            sc = cycle_score(cycle)
            if sc <= mid:
                if best_cycle is None or sc < best_K:
                    best_cycle = cycle
                    best_K = sc
                hi = mid - 1
            else:
                # Cycle exists but uses edges beyond mid; still useful as candidate
                if best_cycle is None or sc < best_K:
                    best_cycle = cycle
                    best_K = sc
                # Can't confirm mid is feasible, but cycle might help at higher K
                # Try to see if a better cycle exists at this K
                lo = mid + 1
        else:
            lo = mid + 1
    
    # If we found a cycle, try to polish it
    if best_cycle is not None and time.time() < deadline - 5:
        polished = two_opt_polish(best_cycle, time.time() + 5)
        if polished is not None:
            best_cycle = polished
    
    # Fallback: try higher K values with DFS
    if best_cycle is None:
        for K in range(N, len(EDGE_ORDER) + 1, 3):
            if time.time() > deadline - 5:
                break
            masks = build_masks(K)
            
            valid = True
            for v in range(N):
                if bin(masks[v]).count('1') < 2:
                    valid = False
                    break
            if not valid:
                continue
            
            if not check_connectivity(masks):
                continue
            
            cycle = greedy_hamilton_cycle(masks, time.time() + 2)
            if cycle is not None and is_valid_cycle(cycle):
                best_cycle = cycle
                break
    
    if best_cycle is not None:
        return best_cycle
    
    return list(range(N))
