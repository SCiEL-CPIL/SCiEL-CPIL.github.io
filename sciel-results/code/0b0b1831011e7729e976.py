"""Find Hamilton cycle with minimal max edge arrival index using bitmask DP with pruning."""
import time
import sys

sys.setrecursionlimit(100000)

# EVOLVE-BLOCK-START
N = 24
EDGE_ORDER = [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]

M = len(EDGE_ORDER)

EDGE_IDX = {}
for i, (a, b) in enumerate(EDGE_ORDER):
    EDGE_IDX[(min(a, b), max(a, b))] = i


def _build_adj(K):
    adj = [0] * N
    for i in range(K):
        a, b = EDGE_ORDER[i]
        adj[a] |= 1 << b
        adj[b] |= 1 << a
    return adj


def _hamilton_cycle_exists(adj, start, end):
    """Check if there's a Hamilton path from start to end in adj.
    Uses DFS with memoization and connectivity pruning."""
    start_bit = 1 << start
    end_bit = 1 << end
    full_mask = (1 << N) - 1
    all_except_end = full_mask & ~end_bit
    
    adj_no_end = [adj[v] & ~end_bit for v in range(N)]
    
    # Quick check: is the graph connected?
    visited = start_bit
    frontier = start_bit
    while frontier:
        lsb = frontier & -frontier
        frontier ^= lsb
        v = lsb.bit_length() - 1
        new = adj[v] & ~visited
        visited |= new
        frontier |= new
    if visited != full_mask:
        return None
    
    memo = {}
    
    def dfs(mask, v):
        key = (mask, v)
        if key in memo:
            return memo[key]
        
        rem = all_except_end & ~mask
        if rem == 0:
            result = bool(adj[v] & end_bit)
            memo[key] = result
            return result
        
        # Connectivity prune: check if all vertices in rem are reachable from v
        # via the subgraph induced by (mask | rem) & ~end_bit
        # Quick prune: BFS from v within (mask|rem) excluding end
        reach = 1 << v
        frontier = 1 << v
        allowed = (mask | rem) & ~end_bit
        while frontier:
            lsb = frontier & -frontier
            frontier ^= lsb
            u = lsb.bit_length() - 1
            new = adj_no_end[u] & allowed & ~reach
            reach |= new
            frontier |= new
        if (reach & rem) != rem:
            memo[key] = False
            return False
        
        nb = adj_no_end[v] & rem
        # Order by degree in rem (ascending) - forced moves first
        cands = []
        while nb:
            lsb = nb & -nb
            nb ^= lsb
            u = lsb.bit_length() - 1
            deg = (adj_no_end[u] & rem).bit_count()
            cands.append((deg, u))
        cands.sort()
        
        for deg, u in cands:
            if dfs(mask | (1 << u), u):
                memo[key] = True
                return True
        
        memo[key] = False
        return False
    
    if not dfs(start_bit, start):
        return None
    
    # Reconstruct path
    path = [end]
    mask = full_mask
    v = end
    # Go backwards from end
    # We need to find the predecessor of end
    # The predecessor must be in (full_mask & ~end_bit) and adjacent to end
    # And must be the last vertex in the path
    
    # Reconstruct forward
    path = [start]
    mask = start_bit
    v = start
    while mask != full_mask:
        rem = all_except_end & ~mask
        nb = adj_no_end[v] & rem
        cands = []
        while nb:
            lsb = nb & -nb
            nb ^= lsb
            u = lsb.bit_length() - 1
            deg = (adj_no_end[u] & rem).bit_count()
            cands.append((deg, u))
        cands.sort()
        found = False
        for deg, u in cands:
            new_mask = mask | (1 << u)
            if memo.get((new_mask, u), False):
                path.append(u)
                mask = new_mask
                v = u
                found = True
                break
        if not found:
            return None
    path.append(end)
    return path


def _score_of(cycle):
    n = len(cycle)
    worst = 0
    for i in range(n):
        a = cycle[i]
        b = cycle[(i + 1) % n]
        key = (min(a, b), max(a, b))
        val = EDGE_IDX.get(key, M) + 1
        if val > worst:
            worst = val
    return worst


def construct():
    deadline = time.time() + 85.0
    
    # Binary search on K
    lo, hi = N, M
    best_cycle = None
    
    # Verify hi is feasible
    adj_hi = _build_adj(hi)
    for end in range(N):
        if end == 0 or not (adj_hi[0] >> end) & 1:
            continue
        if time.time() > deadline - 20:
            break
        cyc = _hamilton_cycle_exists(adj_hi, 0, end)
        if cyc is not None:
            best_cycle = cyc
            break
    
    if best_cycle is None:
        return list(range(N))
    
    while lo < hi:
        if time.time() > deadline - 25:
            break
        mid = (lo + hi) // 2
        if mid < N:
            lo = mid + 1
            continue
        
        adj_mid = _build_adj(mid)
        found = False
        for end in range(N):
            if end == 0 or not (adj_mid[0] >> end) & 1:
                continue
            if time.time() > deadline - 25:
                break
            cyc = _hamilton_cycle_exists(adj_mid, 0, end)
            if cyc is not None:
                hi = mid
                best_cycle = cyc
                found = True
                break
        
        if not found:
            lo = mid + 1
    
    # Try to improve with remaining time
    if time.time() < deadline - 15:
        # Try smaller K values near the found optimum
        for k in range(max(N, hi - 5), hi):
            if time.time() > deadline - 15:
                break
            if k >= hi:
                break
            adj_k = _build_adj(k)
            for end in range(N):
                if end == 0 or not (adj_k[0] >> end) & 1:
                    continue
                if time.time() > deadline - 15:
                    break
                cyc = _hamilton_cycle_exists(adj_k, 0, end)
                if cyc is not None:
                    best_cycle = cyc
                    hi = k
                    break
    
    return best_cycle if best_cycle is not None else list(range(N))
# EVOLVE-BLOCK-END
