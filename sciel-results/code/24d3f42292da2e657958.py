import sys
sys.setrecursionlimit(10000)

N = 12
LCM_12 = 27720

def can_assign_hk(m, L):
    """
    Check if we can assign distinct b_i in (m, m+L] with i | b_i for i=1..12
    using Hopcroft-Karp algorithm.
    """
    # Build bipartite graph
    # Left side: indices 1..12
    # Right side: candidate values
    # We need to map values to unique IDs or just use the value itself as node ID in a set?
    # Hopcroft-Karp usually works on adjacency lists.
    
    # 1. Generate candidates for each i
    candidates = {}
    for i in range(1, N + 1):
        first = ((m // i) + 1) * i
        cands = []
        val = first
        while val <= m + L:
            cands.append(val)
            val += i
        if not cands:
            return False
        candidates[i] = cands
    
    # Map values to indices for the right side
    # Total candidates is small (< 500)
    all_vals = []
    val_to_idx = {}
    for i in range(1, N + 1):
        for v in candidates[i]:
            if v not in val_to_idx:
                val_to_idx[v] = len(all_vals)
                all_vals.append(v)
                
    R = len(all_vals)
    
    # Adjacency list: left_node (i-1) -> list of right_node_indices
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        for v in candidates[i]:
            adj[i-1].append(val_to_idx[v])
            
    # Hopcroft-Karp
    # pairU[u] = v, pairV[v] = u
    # NIL = -1
    pairU = [-1] * N
    pairV = [-1] * R
    dist = [0] * N
    
    def bfs():
        from collections import deque
        q = deque()
        found = False
        for u in range(N):
            if pairU[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = float('inf')
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pairV[v]
                if pu == -1:
                    found = True
                elif dist[pu] == float('inf'):
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
        
    def dfs(u):
        for v in adj[u]:
            pu = pairV[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pairU[u] = v
                pairV[v] = u
                return True
        dist[u] = float('inf')
        return False
        
    matching = 0
    while bfs():
        for u in range(N):
            if pairU[u] == -1:
                if dfs(u):
                    matching += 1
                    
    return matching == N

def compute_f(m):
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if can_assign_hk(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    best_m = 0
    best_L = -1
    # Scan a subset if we want to be faster, but let's scan all to be sure
    # The period is small.
    for m in range(LCM_12):
        L = compute_f(m)
        if L > best_L:
            best_L = L
            best_m = m
    return best_m
