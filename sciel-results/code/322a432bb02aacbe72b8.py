import random
import itertools
from functools import lru_cache

def construct():
    n = 18
    # We'll try to construct a K4-free graph minimizing max triangle-free induced subgraph
    # Strategy: Use 3 parts of 6, add internal edges carefully, and adjust cross-edges
    
    # Start with a base structure and optimize
    # For simplicity and reliability within time limits, use a known good construction
    
    # Let's try: 3 groups of 6. Within each group, a perfect matching (3 edges).
    # Between groups, use a bipartite graph that avoids creating K4s.
    
    # For K4-freedom with internal matchings:
    # If (u,v) is an internal edge in group A, then we cannot have x in B and y in C
    # such that N(u)∩N(v) contains x and y with x-y edge.
    # Simplest: make cross-edges such that for each internal edge, the common 
    # neighborhood between the two endpoints across other groups is limited.
    
    # Alternative simpler approach: Use a graph with no internal edges but 
    # non-complete bipartite between groups, designed so that no 12 vertices
    # form a triangle-free set. But two groups always form triangle-free set.
    
    # The only way to beat 12 is to have internal edges.
    
    # Let me try a specific construction:
    # Groups: A={0-5}, B={6-11}, C={12-17}
    # Internal edges: In each group, add edges (0,1),(2,3),(4,5) - perfect matching
    # Cross edges: For each pair of groups, use a bipartite graph where
    # each vertex has degree 3 (to limit common neighborhoods)
    
    edges = []
    
    # Internal matchings
    for base in [0, 6, 12]:
        for i in range(3):
            edges.append([base + 2*i, base + 2*i + 1])
    
    # Cross edges: between A and B, use a 3-regular bipartite graph
    # Connect A[i] to B[i], B[i+1], B[i+2] mod 6 (cyclic)
    for i in range(6):
        for j in [i, (i+1)%6, (i+2)%6]:
            edges.append([i, 6+j])
    
    # Between B and C: similar
    for i in range(6):
        for j in [i, (i+1)%6, (i+2)%6]:
            edges.append([6+i, 12+j])
    
    # Between C and A: similar  
    for i in range(6):
        for j in [i, (i+1)%6, (i+2)%6]:
            edges.append([12+i, j])
    
    # Verify K4-freedom and compute score
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    # Check K4-free
    def has_k4():
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    continue
                for k in range(j+1, n):
                    if not adj[i][k] or not adj[j][k]:
                        continue
                    for l in range(k+1, n):
                        if adj[i][l] and adj[j][l] and adj[k][l]:
                            return True
        return False
    
    if has_k4():
        # Fallback to K6,6,6 if our construction has K4
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    # Compute max triangle-free induced subgraph size
    def is_triangle_free(vertex_set):
        vs = list(vertex_set)
        for i in range(len(vs)):
            for j in range(i+1, len(vs)):
                for k in range(j+1, len(vs)):
                    u, v, w = vs[i], vs[j], vs[k]
                    if adj[u][v] and adj[v][w] and adj[u][w]:
                        return False
        return True
    
    def max_triangle_free():
        best = 0
        # Check all subsets up to size 13 (if we find 13, we can stop early if we know 12 is achievable)
        # For efficiency, check sizes from 13 down to 12
        for size in range(13, 11, -1):
            for comb in itertools.combinations(range(n), size):
                if is_triangle_free(comb):
                    return size
        # Check size 12
        for comb in itertools.combinations(range(n), 12):
            if is_triangle_free(comb):
                return 12
        return 11  # or lower
    
    score = max_triangle_free()
    
    # If score is still 12, try to improve with modifications
    if score >= 12:
        # Try adding more internal edges or modifying cross edges
        # For now, return what we have
        pass
    
    return edges
