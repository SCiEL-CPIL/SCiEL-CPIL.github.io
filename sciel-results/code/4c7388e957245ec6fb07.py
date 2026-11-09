import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n = 16
    # Adjacency matrix
    adj = [[False] * n for _ in range(n)]
    
    def add_edge(u, v):
        adj[u][v] = True
        adj[v][u] = True
    
    def remove_edge(u, v):
        adj[u][v] = False
        adj[v][u] = False
    
    def has_edge(u, v):
        return adj[u][v]
    
    def edge_count():
        return sum(sum(row) for row in adj) // 2
    
    # Check if adding edge (u,v) creates a Q3
    # Q3 containing edge (u,v): u and v in different bipartition parts
    # Need 4-set A containing u, 4-set B containing v, A∩B=∅
    # Such that non-edges between A and B have a perfect matching
    
    def nonedges_have_perfect_matching(A, B):
        """Check if the non-edge bipartite graph between A and B has a perfect matching."""
        # Build non-edge adjacency: non_edge[i][j] = not adj[A[i]][B[j]]
        # Find perfect matching in this 4x4 bipartite graph
        # Use bitmask DP
        # dp[mask] = whether we can match first popcount(mask) vertices of A to a subset of B
        # Actually: dp[mask] where mask represents which B vertices are used
        # We process A vertices 0,1,2,3
        # dp[mask] = True if we can match A[0..k-1] to B vertices in mask, where k=popcount(mask)
        
        # Precompute non-edge matrix
        ne = [[not adj[A[i]][B[j]] for j in range(4)] for i in range(4)]
        
        # DP: dp[mask] = can we match first popcount(mask) A-vertices to B-vertices in mask
        dp = [False] * 16
        dp[0] = True
        for mask in range(16):
            if not dp[mask]:
                continue
            k = bin(mask).count('1')  # next A vertex to match
            if k >= 4:
                continue
            for j in range(4):
                if mask & (1 << j):
                    continue
                if ne[k][j]:
                    dp[mask | (1 << j)] = True
        return dp[15]
    
    def creates_q3(u, v):
        """Check if adding edge (u,v) creates a Q3 subgraph."""
        # u and v must be in different parts of Q3's bipartition
        # Case 1: u in A, v in B
        # A = {u} + 3 from remaining 14 (excluding v)
        # B = {v} + 3 from remaining (excluding u and A)
        
        others = [x for x in range(n) if x != u and x != v]
        
        for A_rest in combinations(others, 3):
            A = [u] + list(A_rest)
            remaining = [x for x in others if x not in A_rest]
            for B_rest in combinations(remaining, 3):
                B = [v] + list(B_rest)
                if nonedges_have_perfect_matching(A, B):
                    return True
        return False
    
    # Initialize with K_{3,13}
    for a in range(3):
        for b in range(3, 16):
            add_edge(a, b)
    
    best_edges = edge_count()
    
    # Local search: try to add edges
    # Generate list of all missing edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Shuffle for randomization
    random.seed(42)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add each missing edge
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not adj[i][j]]
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() >= deadline:
                break
            if creates_q3(u, v):
                continue
            add_edge(u, v)
            best_edges = edge_count()
            improved = True
    
    # Return edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    return edges
