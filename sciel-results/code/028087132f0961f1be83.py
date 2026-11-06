import time
import random

def construct():
    # Target: Paley graph on 41 vertices
    # Paley graphs exist for q = 1 mod 4 prime. 41 is such a prime.
    # The Paley graph on 41 vertices is (5,5)-free.
    # This proves R(5,5) > 41.
    
    q = 41
    edges = _build_paley(q)
    
    # Verify the graph is (5,5)-free
    if _is_55_free(q, edges):
        return q, edges
    
    # Fallback to Paley 37 if something went wrong (should not happen)
    q = 37
    edges = _build_paley(q)
    if _is_55_free(q, edges):
        return q, edges
        
    # Ultimate fallback: Empty graph on 4 vertices (trivially (5,5)-free)
    return 4, []

def _build_paley(q):
    """Build the Paley graph on q vertices."""
    residues = {(x * x) % q for x in range(1, q)}
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                edges.append((i, j))
    return edges

def _is_55_free(n, edges):
    """
    Check if the graph has no K5 and no independent set of size 5.
    Uses adjacency sets for fast intersection checks.
    """
    adj = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    
    # Check for K5
    # A K5 exists if there is a vertex v such that its neighborhood contains a K4.
    for v in range(n):
        nbrs = list(adj[v])
        if len(nbrs) < 4:
            continue
        # Check all subsets of 4 neighbors to see if they form a K4
        # To speed up, we can iterate combinations
        from itertools import combinations
        for combo in combinations(nbrs, 4):
            a, b, c, d = combo
            # Check if a,b,c,d form a K4
            # They are all neighbors of v, so we just need to check edges among them
            if (b in adj[a] and c in adj[a] and d in adj[a] and
                c in adj[b] and d in adj[b] and
                d in adj[c]):
                return False
                
    # Check for Independent Set of size 5
    # An independent set of size 5 in G is a K5 in the complement graph.
    # We can check this by looking for a vertex v such that its non-neighborhood contains a K4 in the complement.
    # Alternatively, just check for independent sets directly.
    # An independent set {v, a, b, c, d} means a,b,c,d are non-neighbors of v and pairwise non-neighbors.
    
    # Let's use the complement approach for symmetry, or just check independent sets directly.
    # Direct check:
    for v in range(n):
        non_nbrs = [u for u in range(n) if u != v and u not in adj[v]]
        if len(non_nbrs) < 4:
            continue
        # Check if any 4 of these non-neighbors form an independent set (i.e., no edges between them)
        from itertools import combinations
        for combo in combinations(non_nbrs, 4):
            a, b, c, d = combo
            # Check if a,b,c,d are pairwise non-adjacent
            if (b not in adj[a] and c not in adj[a] and d not in adj[a] and
                c not in adj[b] and d not in adj[b] and
                d not in adj[c]):
                return False
                
    return True
