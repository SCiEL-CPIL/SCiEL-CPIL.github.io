"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for original graph
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    # Build Bipartite Double Cover G*
    # Vertices 0..n-1 are U, n..2n-1 are V'
    # Edge (i, j) in G* if i in U, j in V' and (i, j-n) in E(G)
    N2 = 2 * n
    adj_star = [set() for _ in range(N2)]
    for u in range(n):
        for v in adj[u]:
            # u in U, v in V' (index v+n)
            adj_star[u].add(v + n)
            adj_star[v + n].add(u)
    
    # Bron-Kerbosch to find maximal cliques in G*
    # Since G* is bipartite, cliques are bicliques.
    # We only care about cliques with at least one node from U and one from V'.
    # Actually, any clique in a bipartite graph with >2 nodes must span both partitions.
    # Cliques of size 1 or 2 are trivial/invalid for biclique partition (sides must be non-empty and disjoint? 
    # Wait, "nonempty disjoint sides". A biclique [L, R] with L={u}, R={v} is valid if (u,v) is an edge.
    # In G*, this is an edge (u, v+n). This is a clique of size 2.
    # So we want all maximal cliques.
    
    maximal_cliques = []
    
    def bron_kerbosch(R, P_set, X):
        if not P_set and not X:
            if len(R) >= 2: # Must have at least one from each side? 
                # Check if it spans both sides
                has_u = any(x < n for x in R)
                has_v = any(x >= n for x in R)
                if has_u and has_v:
                    maximal_cliques.append(frozenset(R))
            return
        
        # Pivot selection
        # Choose pivot u from P_set union X that maximizes len(P_set & adj_star[u])
        P_union_X = P_set | X
        if not P_union_X:
            return
        pivot = max(P_union_X, key=lambda x: len(P_set & adj_star[x]))
        
        # Iterate over P_set \ N(pivot)
        candidates = P_set - adj_star[pivot]
        for v in list(candidates):
            bron_kerbosch(R | {v}, P_set & adj_star[v], X & adj_star[v])
            P_set.remove(v)
            X.add(v)
    
    # Start with all vertices in P_set
    bron_kerbosch(set(), set(range(N2)), set())
    
    # Convert cliques to bicliques [L, R]
    # L = {x in clique | x < n}
    # R = {x-n in clique | x >= n}
    bicliques = []
    for clique in maximal_cliques:
        L = [x for x in clique if x < n]
        R = [x - n for x in clique if x >= n]
        if L and R:
            # Ensure L and R are lists of ints
            bicliques.append(([int(x) for x in L], [int(x) for x in R]))
    
    # Fallback: If no bicliques found (should not happen), add single edges
    if not bicliques:
        for u, v in edges:
            bicliques.append(([u], [v]))
    
    # Greedy Set Cover
    uncovered = set(edge_set)
    result_bicliques = []
    
    # Pre-compute edge sets for each candidate biclique for faster lookup?
    # Or just compute on the fly.
    # To speed up, we can filter candidates that don't cover any uncovered edges.
    
    while uncovered:
        best_biclique = None
        best_count = -1
        
        for L, R in bicliques:
            count = 0
            # Check how many edges in this biclique are uncovered
            # Edges are (u, v) for u in L, v in R
            # We need to check if (min(u,v), max(u,v)) is in uncovered
            valid = True
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e in uncovered:
                        count += 1
                    else:
                        # If an edge is already covered, we can't use this biclique 
                        # because the partition must be exact (no overlap).
                        # Wait, the requirement is "cross-edge sets must partition all pinned graph edges exactly once".
                        # This means we cannot select a biclique that contains any edge already selected.
                        # So if ANY edge in the biclique is not in uncovered, we cannot use it.
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and count > best_count:
                best_count = count
                best_biclique = (L, R)
        
        if best_biclique is None or best_count == 0:
            # Fallback to single edge
            e = next(iter(uncovered))
            u, v = e
            best_biclique = ([u], [v])
            best_count = 1
        
        L, R = best_biclique
        # Remove covered edges
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                uncovered.discard(e)
        
        result_bicliques.append([list(L), list(R)])
        
    # Final cleanup: ensure types are int
    final_result = []
    for L, R in result_bicliques:
        final_result.append([[int(x) for x in L], [int(x) for x in R]])
        
    return final_result
# EVOLVE-BLOCK-END
