import time
import random
import numpy as np

P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
EDGE_ORDER = P['edge_order']

_edge_pos = {}
for _i, (_u, _v) in enumerate(EDGE_ORDER, 1):
    _edge_pos[(min(_u, _v), max(_u, _v))] = _i

def _edge_idx(u, v):
    return _edge_pos[(min(u, v), max(u, v))]

def _cycle_score(cycle):
    return max(_edge_idx(cycle[i], cycle[(i+1) % N]) for i in range(N))

def construct():
    t0 = time.time()
    deadline = t0 + 85.0
    
    idx_mat = [[0]*N for _ in range(N)]
    for u in range(N):
        for v in range(N):
            if u != v:
                idx_mat[u][v] = _edge_idx(u, v)
    
    trivial = list(range(N))
    best_score = _cycle_score(trivial)
    best_cycle = trivial[:]
    
    # Precompute reverse adjacency for each K
    # adj_fwd[u] = bitmask of v where idx_mat[u][v] <= K
    # adj_rev[v] = bitmask of u where idx_mat[u][v] <= K
    
    def dp_feasible(K):
        n = N
        adj_fwd = [0] * n
        adj_rev = [0] * n
        for u in range(n):
            for v in range(n):
                if u != v and idx_mat[u][v] <= K:
                    adj_fwd[u] |= (1 << v)
                    adj_rev[v] |= (1 << u)
        
        for u in range(n):
            if (adj_fwd[u] & ~(1 << u)).bit_count() < 2:
                return None
        
        # DP: dp[idx] = bitmask of vertices reachable at state mask = (idx << 1) | 1
        # idx = mask >> 1, so mask = (idx << 1) | 1
        size = 1 << (n - 1)
        dp = np.zeros(size, dtype=np.uint32)
        dp[0] = 1  # mask=1 (vertex 0 only), at vertex 0
        
        full_mask = (1 << n) - 1
        full_idx = full_mask >> 1
        
        # Process masks in order of popcount for better cache locality
        # Group masks by popcount
        masks_by_popcount = [[] for _ in range(n)]
        for idx in range(size):
            mask = (idx << 1) | 1
            pc = mask.bit_count()
            masks_by_popcount[pc].append(idx)
        
        for pc in range(2, n + 1):
            for idx in masks_by_popcount[pc]:
                mask = (idx << 1) | 1
                # For each v in mask where v != 0
                m = mask & ~1
                while m:
                    lsb = m & -m
                    v = lsb.bit_length() - 1
                    prev_mask = mask ^ lsb
                    prev_idx = prev_mask >> 1
                    prev_reachable = dp[prev_idx]
                    
                    # Check if any u in prev_reachable is adjacent to v
                    # i.e., prev_reachable & adj_rev[v] != 0
                    if prev_reachable & adj_rev[v]:
                        dp[idx] |= lsb
                    m ^= lsb
            
            # Early exit check at full popcount
            if pc == n:
                full_reachable = dp[full_idx]
                if full_reachable & adj_rev[0]:
                    # Reconstruct path
                    path = [0] * n
                    cur_mask = full_mask
                    cur_v = None
                    # Find a v such that dp[full_idx] has bit v and adj_rev[0] has bit v
                    candidates = full_reachable & adj_rev[0]
                    if candidates:
                        lsb_v = candidates & -candidates
                        cur_v = lsb_v.bit_length() - 1
                        path[n-1] = cur_v
                    else:
                        return None
                    
                    for pos in range(n-2, -1, -1):
                        prev_mask = cur_mask ^ (1 << cur_v)
                        if prev_mask == 0:
                            path[pos] = 0
                            break
                        prev_idx = prev_mask >> 1
                        prev_reachable = dp[prev_idx]
                        # Find u in prev_reachable such that adj_fwd[u] has bit cur_v
                        # i.e., prev_reachable & adj_rev[cur_v] != 0
                        candidates_u = prev_reachable & adj_rev[cur_v]
                        if not candidates_u:
                            return None
                        lsb_u = candidates_u & -candidates_u
                        u = lsb_u.bit_length() - 1
                        path[pos] = u
                        cur_mask = prev_mask
                        cur_v = u
                    return path
                return None
        
        return None
    
    # Binary search
    lo = N
    hi = best_score
    
    while lo <= hi and time.time() < deadline - 5.0:
        mid = (lo + hi) // 2
        path = dp_feasible(mid)
        if path is not None:
            sc = _cycle_score(path)
            if sc <= mid:
                best_cycle = path
                best_score = sc
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            lo = mid + 1
    
    # Fallback: 2-opt with random restarts
    if time.time() < deadline - 2.0:
        rng = random.Random(42)
        
        def two_opt(cycle, time_limit):
            n = N
            improved = True
            dl = time.time() + time_limit
            best = cycle[:]
            best_sc = _cycle_score(best)
            while improved and time.time() < dl:
                improved = False
                for i in range(n - 1):
                    for j in range(i + 2, n):
                        best[i:j+1] = reversed(best[i:j+1])
                        new_sc = _cycle_score(best)
                        if new_sc < best_sc:
                            best_sc = new_sc
                            improved = True
                        else:
                            best[i:j+1] = reversed(best[i:j+1])
            return best, best_sc
        
        while time.time() < deadline - 1.0:
            perm = list(range(N))
            rng.shuffle(perm)
            c, s = two_opt(perm, 1.0)
            if s < best_score:
                best_cycle = c
                best_score = s
    
    return best_cycle
