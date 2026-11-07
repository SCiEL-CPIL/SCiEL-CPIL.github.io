"""Biclique partition solver using greedy edge-cover with backtracking."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

N = P['n']
ALL_EDGES = set(frozenset(e) for e in P['edges'])

def build_adj():
    adj = [set() for _ in range(N)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def find_bicliques(adj, budget=30):
    """Find all maximal bicliques using a heuristic approach."""
    bicliques = []
    n = N
    for u in range(n):
        # Try to find large bicliques with u on the left
        # Use neighborhood intersection
        neigh = adj[u]
        if not neigh:
            continue
        # Greedily build right side: start with all neighbors, try to keep only those
        # that share many neighbors with u's other neighbors
        # Actually, let's use a different approach: find bicliques by considering
        # subsets of vertices
        
    # Better approach: enumerate candidate bicliques
    # A biclique is defined by (L, R) where L x R ⊆ edges
    # We want to find large ones
    
    # Strategy: for each pair of vertices, try to extend to a biclique
    # But that's too slow. Instead, use a greedy edge-cover approach.
    
    return bicliques

def greedy_biclique_partition(adj, deadline):
    """Greedy approach: repeatedly find the largest biclique covering uncovered edges."""
    remaining = set(ALL_EDGES)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Find the largest biclique that only uses edges in remaining
        best_biclique = None
        best_size = 0
        
        # Try each vertex as a potential left vertex
        for u in range(N):
            # Get uncovered edges incident to u
            u_edges = set()
            for e in remaining:
                es = set(e)
                if u in es:
                    u_edges.add(e)
            if not u_edges:
                continue
            
            # For each subset of u's neighbors that could form the right side
            # We want the largest R such that for all v in R, all edges (u', v) for u' in L are in remaining
            # Start with L = {u}, R = neighbors of u that have uncovered edges
            R_candidates = set()
            for e in u_edges:
                es = list(e)
                other = es[0] if es[1] == u else es[1]
                R_candidates.add(other)
            
            if not R_candidates:
                continue
            
            # Try to find a large biclique with u in left
            # Start with R = R_candidates, L = {u}
            # Then try to add more vertices to L
            
            # For efficiency, let's try a few strategies:
            # Strategy 1: L = {u}, R = all neighbors with uncovered edges
            L = {u}
            R = set(R_candidates)
            
            # Verify: all edges in L x R must be in remaining
            valid = True
            for v in R:
                if frozenset((u, v)) not in remaining:
                    valid = False
                    break
            
            if valid:
                size = len(R)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
            
            # Strategy 2: Try to extend L by adding vertices that connect to all of R
            # But this is expensive. Let's try a limited extension.
            # For now, just use the single-vertex left bicliques and see if that's enough.
        
        if best_biclique is None:
            # Fallback: just take one edge
            e = next(iter(remaining))
            u, v = list(e)
            best_biclique = ({u}, {v})
        
        L, R = best_biclique
        bicliques.append((list(L), list(R)))
        
        # Remove covered edges
        for u in L:
            for v in R:
                remaining.discard(frozenset((u, v)))
    
    return bicliques

def try_merge_bicliques(bicliques, adj, deadline):
    """Try to merge compatible bicliques to reduce count."""
    n = len(bicliques)
    if n <= 1:
        return bicliques
    
    # Two bicliques (L1, R1) and (L2, R2) can be merged if:
    # - L1, R1, L2, R2 are pairwise disjoint (or at least L1∩L2=∅, R1∩R2=∅, L1∩R2=∅, L2∩R1=∅)
    # - All cross edges in (L1∪L2) x (R1∪R2) are host edges
    # - The union covers exactly the union of their edge sets (no extra edges needed)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(bicliques)):
            for j in range(i+1, len(bicliques)):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                L1s, R1s = set(L1), set(R1)
                L2s, R2s = set(L2), set(R2)
                
                # Check disjointness
                if L1s & L2s or R1s & R2s or L1s & R2s or L2s & R1s:
                    continue
                
                # Check all cross edges exist
                valid = True
                for u in L1s:
                    for v in R2s:
                        if frozenset((u, v)) not in ALL_EDGES:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    for u in L2s:
                        for v in R1s:
                            if frozenset((u, v)) not in ALL_EDGES:
                                valid = False
                                break
                        if not valid:
                            break
                
                if not valid:
                    continue
                
                # Merge
                new_L = list(L1s | L2s)
                new_R = list(R1s | R2s)
                new_bicliques = []
                for k in range(len(bicliques)):
                    if k != i and k != j:
                        new_bicliques.append(bicliques[k])
                new_bicliques.append((new_L, new_R))
                bicliques = new_bicliques
                changed = True
                break
            if changed:
                break
    
    return bicliques

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    adj = build_adj()
    
    # Greedy partition
    bicliques = greedy_biclique_partition(adj, deadline)
    
    # Try to merge
    bicliques = try_merge_bicliques(bicliques, adj, deadline)
    
    # Convert to proper format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
