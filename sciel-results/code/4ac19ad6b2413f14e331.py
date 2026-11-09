"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

def _is_triangle_free(adj, n):
    """Check if graph with adjacency matrix adj on n vertices is triangle-free."""
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        return False
    return True

def _is_3_colorable(adj, n):
    """Check if graph is 3-colorable using backtracking."""
    color = [-1] * n
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda v: sum(adj[v]), reverse=True)
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        # Determine available colors
        used = set()
        for u in range(n):
            if adj[v][u] and color[u] != -1:
                used.add(color[u])
        for c in range(3):
            if c not in used:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False
    
    return backtrack(0)

def construct():
    deadline = time.time() + 100  # Safety margin before 120s limit
    
    n = 16
    # Start with Grötzsch graph on vertices 0-10
    adj = [[0]*n for _ in range(n)]
    cycle = [(i, (i + 1) % 5) for i in range(5)]
    for u, v in cycle:
        adj[u][v] = adj[v][u] = 1
    for u, w in cycle:
        adj[u][5+w] = adj[5+w][u] = 1
        adj[w][5+u] = adj[5+u][w] = 1
    for u in range(5):
        adj[5+u][10] = adj[10][5+u] = 1
    
    best_edges = sum(sum(row) for row in adj) // 2
    
    # Try to add edges involving vertices 11-15
    # Candidate edges: all edges from 11-15 to 0-10, and among 11-15
    candidate_edges = []
    for u in range(11, 16):
        for v in range(u):
            if v < 11:
                candidate_edges.append((u, v))
            else:
                candidate_edges.append((u, v))
    
    # Sort candidates by some heuristic (e.g., lower degree endpoints first for more room)
    candidate_edges.sort(key=lambda e: (sum(adj[e[0]]), sum(adj[e[1]])))
    
    # Greedy: try to add each edge if it maintains triangle-freeness and 4-chromaticity
    for u, v in candidate_edges:
        if time.time() > deadline:
            break
        if adj[u][v]:
            continue
        # Check triangle-freeness: no common neighbor
        common = False
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                common = True
                break
        if common:
            continue
        # Tentatively add edge
        adj[u][v] = adj[v][u] = 1
        # Check 4-chromaticity (not 3-colorable)
        if not _is_3_colorable(adj, n):
            best_edges += 1
        else:
            # Revert
            adj[u][v] = adj[v][u] = 0
    
    # Convert to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if adj[u][v]:
                edges.append([u, v])
    
    return edges
# EVOLVE-BLOCK-END
