import time
import random
from itertools import combinations

def _is_quadratic_residue(a, p):
    return pow(a, (p - 1) // 2, p) == 1

def _paley_edges_17():
    """Generate edges of the Paley graph on 17 vertices (0..16)."""
    p = 17
    edges = []
    residues = set()
    for r in range(1, p):
        if _is_quadratic_residue(r, p):
            residues.add(r)
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in residues or (p - diff) in residues:
                edges.append([i, j])
    return edges

def _is_k4_free(edges, n=18):
    """Check if graph is K4-free."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    for a, b, c, d in combinations(range(n), 4):
        if (adj[a][b] and adj[a][c] and adj[a][d] and
            adj[b][c] and adj[b][d] and adj[c][d]):
            return False
    return True

def _max_triangle_free_size(edges, n=18):
    """Find the largest triangle-free induced subgraph size using branch and bound."""
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    
    # Precompute: for each pair, check if they form a triangle with any third vertex
    # A set S is triangle-free if no three vertices in S form a triangle
    # We want the max |S| such that S induces a triangle-free graph
    
    # Use branch and bound: try to build a triangle-free set
    # A vertex can be added to set S if it doesn't form a triangle with any two vertices in S
    # i.e., for any u, v in S, not (adj[u][v] and adj[u][w] and adj[v][w])
    
    best = [0]
    
    def backtrack(current_set, candidates):
        if len(candidates) + len(current_set) <= best[0]:
            return
        if not candidates:
            if len(current_set) > best[0]:
                best[0] = len(current_set)
            return
        
        # Try adding first candidate
        v = candidates[0]
        new_candidates = []
        for u in candidates[1:]:
            # Check if adding v and u would create a triangle with any w in current_set
            # Actually, we need: for all w in current_set, not (adj[v][w] and adj[u][w] and adj[v][u])
            # But also v and u themselves: if adj[v][u], then for any w in current_set with adj[v][w] and adj[u][w], that's a triangle
            can_add = True
            for w in current_set:
                if adj[v][w] and adj[u][w] and adj[v][u]:
                    can_add = False
                    break
            if can_add:
                new_candidates.append(u)
        
        # Also check v with current_set
        v_ok = True
        for w in current_set:
            for x in current_set:
                if w != x and adj[v][w] and adj[v][x] and adj[w][x]:
                    v_ok = False
                    break
            if not v_ok:
                break
        
        if v_ok:
            backtrack(current_set + [v], new_candidates)
        
        # Try not adding v
        backtrack(current_set, candidates[1:])
    
    backtrack([], list(range(n)))
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin
    
    # Start with Paley graph on 17 vertices + isolated vertex 17
    base_edges = _paley_edges_17()
    
    # Try different configurations for vertex 17
    # First, just use isolated vertex
    edges = base_edges[:]
    
    # Verify K4-free
    if not _is_k4_free(edges):
        # Fallback to complete tripartite
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    current_score = _max_triangle_free_size(edges)
    best_edges = edges[:]
    best_score = current_score
    
    # Try adding edges from vertex 17 to some vertices in 0..16
    # We need to maintain K4-free property
    # Vertex 17 connected to subset S of {0..16}
    # This creates K4 if there exist u,v in S such that u,v and two other vertices form K4
    # More precisely: if 17 is connected to u and v, and u,v are adjacent, and there exists w adjacent to both 17 and u and v... 
    # Actually K4 requires 4 mutually adjacent vertices. If 17 is in K4, the other 3 must be mutually adjacent and all adjacent to 17.
    
    # Let's try small subsets S for vertex 17's neighbors
    random.seed(42)
    
    # Try all subsets of size 0 to 3 for vertex 17's neighbors
    for size in range(0, 4):
        if time.time() > deadline:
            break
        for subset in combinations(range(17), size):
            if time.time() > deadline:
                break
            test_edges = base_edges[:] + [[17, v] for v in subset]
            if _is_k4_free(test_edges):
                score = _max_triangle_free_size(test_edges)
                if score < best_score:
                    best_score = score
                    best_edges = test_edges[:]
    
    # Try removing some edges from the Paley graph
    # This might allow adding more edges from vertex 17
    if time.time() < deadline - 30:
        # Try removing one edge and adding edges from 17
        for idx, (u, v) in enumerate(base_edges):
            if time.time() > deadline:
                break
            modified = base_edges[:idx] + base_edges[idx+1:]
            # Now try adding edges from 17
            for size in range(0, 4):
                if time.time() > deadline:
                    break
                # Sample some subsets rather than all
                for _ in range(20):
                    subset = random.sample(range(17), size)
                    test_edges = modified + [[17, v] for v in subset]
                    if _is_k4_free(test_edges):
                        score = _max_triangle_free_size(test_edges)
                        if score < best_score:
                            best_score = score
                            best_edges = test_edges[:]
    
    return best_edges
