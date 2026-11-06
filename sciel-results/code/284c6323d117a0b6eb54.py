"""Find minimum-bottleneck Hamilton cycle via binary search + bitmask DP with time management."""
import time

# EVOLVE-BLOCK-START
P = {'n': 24, 'edge_order': [[3, 11], [9, 18], [6, 7], [8, 18], [5, 15], [19, 22], [0, 23], [13, 19], [1, 3], [21, 23], [0, 2], [15, 23], [4, 12], [16, 21], [3, 8], [10, 23], [15, 22], [9, 13], [10, 13], [18, 19], [3, 7], [9, 12], [0, 17], [12, 17], [3, 14], [6, 23], [0, 12], [2, 5], [8, 20], [11, 19], [21, 22], [10, 16], [4, 11], [0, 19], [13, 21], [15, 16], [9, 11], [5, 8], [5, 12], [7, 8], [9, 14], [2, 23], [17, 20], [1, 23], [4, 7], [14, 22], [18, 22], [10, 15], [11, 22], [16, 17], [6, 16], [1, 5], [1, 20], [6, 10], [0, 21], [0, 10], [15, 21], [12, 23], [7, 19], [8, 23], [8, 21], [4, 14], [0, 22], [3, 22], [13, 18], [20, 23], [17, 19], [14, 17], [3, 17], [0, 4], [1, 8], [2, 22], [2, 20], [8, 22], [1, 16], [3, 19], [14, 20], [7, 18], [4, 15], [10, 12], [7, 13], [18, 20], [2, 19], [1, 19], [13, 17], [1, 2], [9, 21], [12, 15], [8, 16], [19, 20], [1, 11], [5, 17], [13, 14], [11, 17], [13, 16], [7, 17], [2, 15], [16, 22], [2, 7], [9, 16], [7, 23], [9, 23], [8, 9], [5, 23], [7, 20], [1, 17], [0, 7], [9, 22], [2, 17], [3, 18], [17, 22], [0, 5], [9, 15], [0, 8], [1, 9], [6, 11], [5, 7], [0, 15], [12, 19], [7, 21], [15, 18], [11, 15], [7, 10], [3, 10], [0, 13], [12, 13], [6, 20], [9, 10], [2, 18], [1, 14], [4, 22], [14, 15], [3, 9], [16, 20], [16, 23], [10, 20], [4, 9], [6, 18], [0, 14], [0, 6], [17, 23], [5, 10], [20, 22], [10, 19], [18, 21], [16, 19], [11, 21], [7, 12], [1, 10], [6, 8], [0, 11], [9, 19], [0, 9], [4, 6], [3, 16], [3, 13], [1, 6], [6, 21], [2, 14], [12, 18], [17, 21], [12, 16], [14, 21], [13, 20], [11, 20], [5, 16], [3, 23], [1, 18], [7, 9], [6, 12], [8, 19], [11, 12], [10, 11], [7, 22], [8, 11], [20, 21], [2, 12], [8, 17], [8, 10], [4, 23], [9, 17], [14, 18], [8, 13], [9, 20], [1, 15], [5, 9], [3, 6], [2, 11], [3, 15], [3, 20], [4, 19], [4, 21], [6, 15], [10, 22], [7, 14], [2, 6], [13, 23], [0, 18], [15, 19], [4, 5], [19, 23], [12, 21], [5, 14], [4, 20], [5, 21], [8, 14], [12, 14], [12, 22], [0, 3], [7, 11], [5, 13], [13, 15], [7, 16], [1, 13], [4, 8], [14, 23], [15, 17], [6, 9], [2, 8], [1, 7], [4, 10], [2, 21], [7, 15], [13, 22], [2, 4], [16, 18], [6, 17], [10, 21], [5, 22], [2, 9], [1, 12], [0, 1], [12, 20], [4, 16], [22, 23], [19, 21], [5, 11], [10, 18], [5, 20], [1, 4], [17, 18], [18, 23], [14, 16], [11, 16], [8, 12], [2, 13], [3, 5], [5, 18], [6, 22], [6, 14], [4, 13], [2, 3], [6, 19], [0, 20], [1, 21], [5, 6], [10, 17], [11, 14], [15, 20], [11, 23], [10, 14], [8, 15], [11, 13], [3, 21], [14, 19], [11, 18], [3, 4], [4, 17], [3, 12], [0, 16], [2, 10], [2, 16], [1, 22], [4, 18], [5, 19], [6, 13]]}

N = P['n']
FULL = (1 << N) - 1

def construct():
    start_time = time.time()
    deadline = start_time + 82  # safety margin before 90s kill
    
    # Build edge index lookup: edge_index[(u,v)] = 1-based arrival index
    edge_index = {}
    for idx, (u, v) in enumerate(P['edge_order']):
        edge_index[(min(u,v), max(u,v))] = idx + 1
    
    # For a given cycle (permutation), compute its bottleneck score
    def cycle_score(perm):
        """perm is a list of N vertices forming a cycle with implicit closing edge."""
        max_idx = 0
        for i in range(N):
            u = perm[i]
            v = perm[(i + 1) % N]
            key = (min(u, v), max(u, v))
            idx = edge_index[key]
            if idx > max_idx:
                max_idx = idx
        return max_idx
    
    # Get baseline score from identity permutation
    identity = list(range(N))
    best_score = cycle_score(identity)
    best_path = identity
    
    # Binary search on K
    lo = N
    hi = best_score  # we know identity achieves best_score
    
    # For feasibility check, use bitmask DP with vertex 0 fixed as start
    # dp[mask] = set of possible 'last' vertices
    # We'll use a dict: mask -> int (bitmask of possible last vertices)
    
    def feasible(K, time_limit):
        """Check if Hamilton cycle exists using only edges with arrival <= K."""
        # Build adjacency as bitmasks
        adj = [0] * N
        for idx, (u, v) in enumerate(P['edge_order']):
            if idx + 1 > K:
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        # Degree check
        for i in range(N):
            if bin(adj[i]).count('1') < 2:
                return False, None
        
        # DP: fix vertex 0 as start
        # State: (mask, last) where 0 in mask
        # Use dict: mask -> bitmask of last vertices
        # Process by increasing popcount
        
        dp = {}
        parent = {}
        
        # Initialize
        start_mask = 1
        dp[start_mask] = 1  # only last=0
        parent[(start_mask, 0)] = -1
        
        # Use a list of (mask, last) to process, ordered by popcount
        # For efficiency, process level by level
        current = [(start_mask, 0)]
        
        while current:
            if time.time() > time_limit:
                return False, None
            
            next_level = []
            for mask, last in current:
                avail = adj[last] & ~mask
                while avail:
                    bit = avail & (-avail)
                    nxt = bit.bit_length() - 1
                    new_mask = mask | bit
                    
                    if new_mask in dp:
                        if dp[new_mask] & bit:
                            avail ^= bit
                            continue
                        dp[new_mask] |= bit
                    else:
                        dp[new_mask] = bit
                    
                    parent[(new_mask, nxt)] = last
                    next_level.append((new_mask, nxt))
                    avail ^= bit
            
            current = next_level
        
        # Check for completion
        if FULL in dp:
            last_bits = dp[FULL]
            while last_bits:
                bit = last_bits & (-last_bits)
                last = bit.bit_length() - 1
                if adj[last] & 1:
                    # Reconstruct
                    path = [last]
                    mask = FULL
                    cur = last
                    while cur != 0:
                        prev = parent[(mask, cur)]
                        path.append(prev)
                        mask ^= (1 << cur)
                        cur = prev
                    path.reverse()
                    return True, path
                last_bits ^= bit
        
        return False, None
    
    # Binary search
    while lo <= hi and time.time() < deadline:
        mid = (lo + hi) // 2
        remaining = deadline - time.time()
        if remaining < 5:
            break
        ok, path = feasible(mid, time.time() + remaining - 2)
        if ok:
            s = cycle_score(path)
            if s < best_score:
                best_score = s
                best_path = path
            hi = mid - 1
        else:
            lo = mid + 1
    
    return best_path
# EVOLVE-BLOCK-END
