"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    n = len(adj)
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a][b]:
                continue
            for c in range(b + 1, n):
                if not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        return False
    return True

def _max_triangle_free_size_fast(adj):
    """
    Estimate the largest triangle-free induced subgraph size using a greedy
    approach: repeatedly try to build a large independent set in the triangle
    complement, but also account for edges that don't form triangles.
    
    Actually, a triangle-free induced subgraph means no three vertices form a triangle.
    We use a greedy heuristic: start with all vertices, and iteratively remove
    vertices that participate in triangles, trying to maximize remaining size.
    """
    n = len(adj)
    # Use a greedy approach: try to find a large subset with no triangles
    # This is NP-hard, so we use heuristics.
    
    # Greedy: start with all vertices, remove vertices that are in many triangles
    remaining = set(range(n))
    while remaining:
        # Count triangles for each vertex in remaining
        tri_count = {v: 0 for v in remaining}
        rem_list = list(remaining)
        for i in range(len(rem_list)):
            a = rem_list[i]
            for j in range(i + 1, len(rem_list)):
                b = rem_list[j]
                if not adj[a][b]:
                    continue
                for k in range(j + 1, len(rem_list)):
                    c = rem_list[k]
                    if adj[a][c] and adj[b][c]:
                        tri_count[a] += 1
                        tri_count[b] += 1
                        tri_count[c] += 1
        
        if all(t == 0 for t in tri_count.values()):
            return len(remaining)
        
        # Remove vertex with highest triangle count
        worst = max(remaining, key=lambda v: tri_count[v])
        remaining.remove(worst)
    
    return 0

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    best_edges = None
    best_score = 18  # Worst case
    
    # Start with the known K6,6,6 construction
    n = 18
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    base_adj = [[False] * n for _ in range(n)]
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    base_adj[a][b] = True
                    base_adj[b][a] = True
    
    base_edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if base_adj[a][b]:
                base_edges.append([a, b])
    
    base_score = _max_triangle_free_size_fast(base_adj)
    best_score = base_score
    best_edges = base_edges[:]
    
    # Randomized search: start from random K4-free graphs and try to improve
    rng = random.Random(42)
    
    for trial in range(500):
        if time.time() > deadline:
            break
        
        # Generate a random K4-free graph
        # Strategy: start with empty graph, add edges greedily avoiding K4
        adj = [[False] * n for _ in range(n)]
        
        # Random edge order
        all_edges = [(a, b) for a in range(n) for b in range(a + 1, n)]
        rng.shuffle(all_edges)
        
        # Greedy: add edge if it doesn't create a K4
        for a, b in all_edges:
            # Check if adding (a,b) creates a K4
            creates_k4 = False
            for c in range(n):
                if c == a or c == b or not adj[a][c] or not adj[b][c]:
                    continue
                for d in range(c + 1, n):
                    if d == a or d == b or d == c:
                        continue
                    if adj[a][d] and adj[b][d] and adj[c][d]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            if not creates_k4:
                adj[a][b] = True
                adj[b][a] = True
        
        # Evaluate
        score = _max_triangle_free_size_fast(adj)
        if score < best_score:
            best_score = score
            best_edges = []
            for a in range(n):
                for b in range(a + 1, n):
                    if adj[a][b]:
                        best_edges.append([a, b])
            
            # Try local improvements on this graph
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Try flipping edges (add or remove) to reduce max triangle-free size
                for a in range(n):
                    for b in range(a + 1, n):
                        if time.time() > deadline:
                            break
                        old = adj[a][b]
                        adj[a][b] = not old
                        adj[b][a] = not adj[a][b]
                        
                        if _is_k4_free(adj):
                            new_score = _max_triangle_free_size_fast(adj)
                            if new_score < best_score:
                                best_score = new_score
                                best_edges = []
                                for x in range(n):
                                    for y in range(x + 1, n):
                                        if adj[x][y]:
                                            best_edges.append([x, y])
                                improved = True
                            else:
                                adj[a][b] = old
                                adj[b][a] = old
                        else:
                            adj[a][b] = old
                            adj[b][a] = old
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
    
    return best_edges
# EVOLVE-BLOCK-END
