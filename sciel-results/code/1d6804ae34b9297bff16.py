"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 16}

def construct():
    start = time.time()
    deadline = start + 50.0  # safety margin before 60s
    
    n = P['n']
    half = n // 2
    
    # Build K_{8,8} edges
    edges = []
    for u in range(half):
        for v in range(half, n):
            edges.append([u, v])
    
    # Check if we're still within time
    if time.time() > deadline:
        return edges
    
    # Try to see if we can do better with a search (but K_{8,8} is likely optimal)
    # The maximum number of edges in a graph on 16 vertices with diameter 2
    # where every edge is essential is at most C(16,2) = 120.
    # K_{8,8} gives 64 edges. Let's see if any denser construction exists.
    
    # For an edge to be essential (its removal increases diameter above 2 or disconnects),
    # it must be that the edge is the ONLY length-2 path between its endpoints, OR it's a bridge.
    # Actually, for diameter to increase above 2 after removing edge (u,v), we need that
    # dist(u,v) > 2 in G - (u,v). Since they were adjacent (dist=1), after removal they need
    # dist >= 3. This means there's NO common neighbor of u and v in G - (u,v), i.e., 
    # N(u) ∩ N(v) = ∅ in the original graph (since removing the edge doesn't change neighbor sets 
    # for other purposes... wait, removing edge (u,v) means u is no longer in N(v) and v is no longer in N(u).
    # So the condition is: in G, N(u) ∩ N(v) \ {u,v} must be empty, i.e., u and v have no common neighbors.
    
    # For K_{8,8}: if u in A, v in B, then N(u) = B, N(v) = A. N(u) ∩ N(v) = ∅. Good.
    # If we added an edge within A, say (u1, u2), then N(u1) ⊇ B, N(u2) ⊇ B, so N(u1) ∩ N(u2) ⊇ B ≠ ∅.
    # So that edge would NOT be essential. Hence K_{8,8} is maximal in this family.
    
    # Can we have a non-bipartite graph with more essential edges?
    # Consider a complete graph K_16: 120 edges. But every pair has 14 common neighbors, so no edge is essential.
    # We need a graph where for every edge (u,v), |N(u) ∩ N(v) \ {u,v}| = 0.
    # This is equivalent to saying no triangle contains the edge... wait, no. If u-v is an edge and they have a common neighbor w, then u-v-w-u is a triangle, and dist(u,v) = 1 in G, but in G - (u,v), dist(u,v) ≤ 2 via w. So the edge is NOT essential.
    # Therefore, for every edge to be essential, the graph must be TRIANGLE-FREE!
    # Because if there's a triangle u-v-w-u, then edge (u,v) has common neighbor w, so (u,v) is not essential.
    
    # So we need a triangle-free graph on 16 vertices, diameter exactly 2, and every edge is essential.
    # In a triangle-free graph, every edge IS essential (since no common neighbors for any edge).
    # Wait, is that true? In a triangle-free graph, N(u) ∩ N(v) = ∅ for any edge (u,v). So yes, removing (u,v) makes dist(u,v) > 2 (if the graph stays connected) or disconnects it.
    # But we also need the graph to be connected and have diameter exactly 2.
    
    # So the problem reduces to: find a triangle-free graph on 16 vertices with diameter 2, maximizing edges.
    # The maximum number of edges in a triangle-free graph on n vertices is given by Turán's theorem: floor(n²/4).
    # For n=16: floor(16²/4) = 64. This is achieved by K_{8,8}.
    
    # So K_{8,8} is optimal! 64 edges is the maximum.
    
    # Let's verify: K_{8,8} is triangle-free, has diameter 2 (any two vertices in different parts are adjacent; any two in the same part have a common neighbor in the other part), and every edge is essential.
    
    return edges
# EVOLVE-BLOCK-END
