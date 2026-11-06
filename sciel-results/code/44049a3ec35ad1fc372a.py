import time
import random

def construct():
    """
    Tries to find a C5-free graph with no independent set of 9 vertices
    with as many vertices as possible. Falls back to proven n=32 solution.
    """
    deadline = time.time() + 110  # 110 seconds max
    
    # First, return the proven n=32 solution as fallback
    def make_k4_solution():
        n = 32
        edges = []
        for b in range(8):
            base = b * 4
            for i in range(4):
                for j in range(i + 1, 4):
                    edges.append([base + i, base + j])
        return {'n': n, 'edges': edges}
    
    fallback = make_k4_solution()
    
    # Try to find n=33, 34, or 35 solutions
    for target_n in [33, 34, 35]:
        if time.time() > deadline - 10:
            break
        result = _search_graph(target_n, deadline)
        if result is not None:
            return result
    
    return fallback

def _has_c5(n, adj):
    """Check if graph has a C5 (5-cycle) as a subgraph."""
    # For each pair of non-adjacent vertices, check if they're in a C5
    # A C5: v0-v1-v2-v3-v4-v0
    # We need to find 5 distinct vertices forming a cycle
    
    # Use a more efficient approach: for each vertex, do BFS/DFS limited
    # to depth 4 looking for back edges to start
    
    for v0 in range(n):
        # Find neighbors of v0
        nbs0 = [u for u in range(n) if u != v0 and adj[v0][u]]
        for v1 in nbs0:
            # v1 is adjacent to v0, find neighbors of v1 (excluding v0)
            nbs1 = [u for u in range(n) if u != v0 and u != v1 and adj[v1][u]]
            for v2 in nbs1:
                nbs2 = [u for u in range(n) if u != v0 and u != v1 and u != v2 and adj[v2][u]]
                for v3 in nbs2:
                    nbs3 = [u for u in range(n) if u != v0 and u != v1 and u != v2 and u != v3 and adj[v3][u]]
                    for v4 in nbs3:
                        # Check if v4 is adjacent to v0
                        if adj[v4][v0]:
                            return True
    return False

def _max_independent_set(n, adj):
    """Find maximum independent set size (exact, for small n)."""
    # Use branch and bound
    best = [0]
    def backtrack(remaining, current_size):
        if not remaining:
            return
        if current_size >= best[0]:
            return
        if current_size + len(remaining) <= best[0]:
            return
        # Pick a vertex
        v = remaining[0]
        rest = remaining[1:]
        # Include v
        if current_size + 1 > best[0]:
            best[0] = current_size + 1
        # Vertices that can still be added (non-adjacent to v)
        new_remaining = [u for u in rest if not adj[v][u]]
        backtrack(new_remaining, current_size + 1)
        # Don't include v
        backtrack(rest, current_size)
    
    backtrack(list(range(n)), 0)
    return best[0]

def _is_valid(n, adj):
    """Check if graph is C5-free and has no independent set of size 9."""
    if _has_c5(n, adj):
        return False
    if _max_independent_set(n, adj) >= 9:
        return False
    return True

def _search_graph(target_n, deadline):
    """Randomized search for a valid graph with target_n vertices."""
    if target_n < 2:
        return None
    
    best_result = None
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        # Generate random graph
        adj = [[False] * target_n for _ in range(target_n)]
        # Start with some structure: try to build from K4-like blocks
        # Use a density that might work
        
        # Strategy: start with random edges, then try to remove C5's
        # Or: build up from known good structures
        
        # Try: random graph with appropriate density
        # For C5-free, density should be moderate
        # For alpha < 9, need enough edges
        
        p = 0.4  # edge probability
        for i in range(target_n):
            for j in range(i+1, target_n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Check validity
        if _is_valid(target_n, adj):
            edges = []
            for i in range(target_n):
                for j in range(i+1, target_n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': target_n, 'edges': edges}
    
    return None
