import time
from collections import defaultdict, deque

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    P = 101
    # Precompute inverses
    inv = {x: pow(x, -1, P) for x in range(1, 11)}
    
    # Generate all non-empty subsets of {1..10}, compute their residue and size
    subsets = []
    for mask in range(1, 1 << 10):
        subset = []
        s = 0
        for x in range(1, 11):
            if mask >> (x - 1) & 1:
                subset.append(x)
                s += inv[x]
        r = s % P
        sz = len(subset)
        subsets.append((r, sz, subset))
    
    # Precompute: for each (residue, size) pair, list of subset indices
    subsets_by_residue_size = defaultdict(list)
    for idx, (r, sz, subset) in enumerate(subsets):
        subsets_by_residue_size[(r, sz)].append(idx)
    
    # Hopcroft-Karp for bipartite matching with current-edge optimization
    def hopcroft_karp(adj, n_left, n_right):
        matchL = [-1] * n_left
        matchR = [-1] * n_right
        dist = [0] * n_left
        
        def bfs():
            queue = deque()
            for u in range(n_left):
                if matchL[u] == -1:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            found = False
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if matchR[v] == -1:
                        found = True
                    elif dist[matchR[v]] == float('inf'):
                        dist[matchR[v]] = dist[u] + 1
                        queue.append(matchR[v])
            return found
        
        def dfs(u, cur):
            for i in range(cur[u], len(adj[u])):
                cur[u] = i
                v = adj[u][i]
                if matchR[v] == -1 or (dist[matchR[v]] == dist[u] + 1 and dfs(matchR[v], cur)):
                    matchL[u] = v
                    matchR[v] = u
                    return True
            dist[u] = float('inf')
            return False
        
        matching_size = 0
        while bfs():
            cur = [0] * n_left
            for u in range(n_left):
                if matchL[u] == -1:
                    if dfs(u, cur):
                        matching_size += 1
        
        if matching_size == n_left:
            return matchL
        return None
    
    # Try max_size from 1 to 10
    best_result = None
    for max_size in range(1, 11):
        if time.time() > deadline:
            break
        
        # Build adjacency and right node mapping efficiently
        right_map = {}
        right_idx = 0
        adj = [[] for _ in range(P)]
        
        for r in range(P):
            for sz in range(1, max_size + 1):
                for idx in subsets_by_residue_size.get((r, sz), []):
                    if idx not in right_map:
                        right_map[idx] = right_idx
                        right_idx += 1
                    adj[r].append(right_map[idx])
        
        n_right = right_idx
        
        # Run Hopcroft-Karp
        matchL = hopcroft_karp(adj, P, n_right)
        
        if matchL is not None:
            # Perfect matching found! Build the result
            rev_right_map = {ridx: idx for idx, ridx in right_map.items()}
            result = [None] * P
            for r in range(P):
                ridx = matchL[r]
                subset_idx = rev_right_map[ridx]
                result[r] = subsets[subset_idx][2]
            
            best_result = result
            break
    
    if best_result is None:
        # Fallback: use the greedy minimum-size approach
        best = [None] * P
        for mask in range(1, 1 << 10):
            subset = []
            s = 0
            for x in range(1, 11):
                if mask >> (x - 1) & 1:
                    subset.append(x)
                    s += inv[x]
            r = s % P
            sz = len(subset)
            if best[r] is None or sz < best[r][0]:
                best[r] = (sz, subset)
        for r in range(P):
            if best[r] is None:
                for mask in range(1, 1 << 10):
                    subset = []
                    s = 0
                    for x in range(1, 11):
                        if mask >> (x - 1) & 1:
                            subset.append(x)
                            s += inv[x]
                    if s % P == r:
                        best[r] = (len(subset), subset)
                        break
                if best[r] is None:
                    best[r] = (1, [1])
        best_result = [best[r][1] for r in range(P)]
    
    return best_result
# EVOLVE-BLOCK-END
