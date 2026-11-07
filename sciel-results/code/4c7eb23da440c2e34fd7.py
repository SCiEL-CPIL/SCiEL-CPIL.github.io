"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s limit
    
    n = P['n']
    
    # Build static adjacency sets for host graph
    host_adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        host_adj[u].add(v)
        host_adj[v].add(u)
        
    # Initialize remaining edges as a set of frozensets for O(1) lookup and removal
    remaining_edges = set()
    for u, v in P['edges']:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
            
    best_solution = None
    best_count = float('inf')
    
    # We will run multiple deterministic passes with different initial weightings or expansion heuristics.
    # Since the problem is small (n=30, ~200 edges), we can afford a robust search.
    
    # Strategy: 
    # 1. Maintain a set of remaining edges.
    # 2. At each step, find the "best" biclique to extract.
    #    To do this efficiently, we can use a heuristic based on edge density or specific vertex degrees.
    #    A stronger approach: Try to find a maximal biclique for several candidate starting edges, 
    #    and pick the one that removes the most edges.
    
    # Let's implement a robust greedy that, at each step, evaluates multiple candidate bicliques
    # and chooses the one that maximizes the number of removed edges.
    
    for attempt in range(20): # Fewer, heavier attempts
        if time.time() > deadline:
            break
            
        current_remaining = set(remaining_edges)
        bicliques = []
        valid_solution = True
        
        while current_remaining:
            if time.time() > deadline:
                valid_solution = False
                break
                
            # Heuristic: Pick a few candidate edges to start biclique expansion
            # We prioritize edges that are in "dense" areas or have high degree vertices.
            # To avoid bias, we can just pick the first few in a sorted list or random sample.
            # Given the "structural change" directive, let's use a more principled expansion.
            
            # Instead of random start, let's try to find the largest biclique in the remaining graph.
            # This is NP-hard, but for n=30 we can approximate well.
            
            # Approximation: 
            # 1. Select a vertex v with highest degree in remaining graph.
            # 2. Let R be its neighbors in remaining graph.
            # 3. Let L be {v}.
            # 4. Expand L by adding vertices connected to all of R.
            # 5. Expand R by adding vertices connected to all of L.
            # 6. Repeat.
            # 7. Compare this result with starting from other high-degree vertices?
            #    To save time, we'll just do one robust expansion per step but with a specific ordering.
            
            # Let's compute degrees in current_remaining
            deg = [0] * n
            for e in current_remaining:
                u, v = list(e)
                deg[u] += 1
                deg[v] += 1
            
            # Pick vertex with max degree
            if not any(deg):
                break
                
            v_start = max(range(n), key=lambda x: deg[x])
            
            # Initialize biclique
            # We need to decide which side v_start goes to.
            # Let's try both orientations? Or just fix L={v_start} and R=neighbors.
            # Actually, a biclique (L,R) covers edges between L and R.
            # If we start with L={v_start}, R must be a subset of neighbors of v_start in remaining.
            # To maximize size, start with R = all neighbors of v_start in remaining.
            
            L = {v_start}
            R = set()
            for e in current_remaining:
                if v_start in e:
                    other = list(e)[0] if list(e)[0] != v_start else list(e)[1]
                    R.add(other)
            
            # Expand
            changed = True
            while changed:
                changed = False
                
                # Expand L: find w not in L∪R such that w is connected to all in R via current_remaining
                # Optimization: Precompute common neighbors? 
                # For each w, check if R is subset of neighbors of w in current_remaining.
                
                # Efficient check:
                # Let N(w) be neighbors of w in current_remaining.
                # w can join L if R ⊆ N(w).
                
                # We can iterate through candidates.
                candidates_L = []
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if all r in R are connected to w in current_remaining
                    ok = True
                    for r in R:
                        if frozenset((w, r)) not in current_remaining:
                            ok = False
                            break
                    if ok:
                        candidates_L.append(w)
                
                # If we have candidates, pick the one that allows most R expansion?
                # Or just pick the one with highest degree in remaining?
                if candidates_L:
                    # Pick w that is connected to the most vertices in current_remaining (global degree)
                    # This tends to find larger bicliques
                    w = max(candidates_L, key=lambda x: deg[x])
                    L.add(w)
                    changed = True
                    
                # Expand R: find w not in L∪R such that w is connected to all in L via current_remaining
                candidates_R = []
                for w in range(n):
                    if w in L or w in R:
                        continue
                    ok = True
                    for l in L:
                        if frozenset((w, l)) not in current_remaining:
                            ok = False
                            break
                    if ok:
                        candidates_R.append(w)
                        
                if candidates_R:
                    w = max(candidates_R, key=lambda x: deg[x])
                    R.add(w)
                    changed = True
            
            # We have a maximal biclique (L, R) with respect to this greedy expansion.
            # Remove all cross edges from current_remaining
            removed_count = 0
            for l in L:
                for r in R:
                    e = frozenset((l, r))
                    if e in current_remaining:
                        current_remaining.remove(e)
                        removed_count += 1
            
            if removed_count == 0:
                # Should not happen if L,R non-empty and start edge existed, but safety check
                valid_solution = False
                break
                
            bicliques.append([sorted(L), sorted(R)])
            
        if valid_solution:
            # Verify solution
            covered_edges = set()
            for left, right in bicliques:
                for l in left:
                    for r in right:
                        e = frozenset((l, r))
                        if e in covered_edges:
                            valid_solution = False
                            break
                        covered_edges.add(e)
                        if e not in remaining_edges: # Check host edge
                            valid_solution = False
                            break
                    if not valid_solution:
                        break
                if not valid_solution:
                    break
            
            if not valid_solution:
                continue
                
            if covered_edges != remaining_edges:
                continue
                
            if len(bicliques) < best_count:
                best_count = len(bicliques)
                best_solution = bicliques
                
        if best_count == 1:
            break

    if best_solution is None:
        # Fallback: Star decomposition (guaranteed valid, though not minimal)
        # Each vertex u gets a biclique [u, neighbors(u)]? 
        # Wait, [u, neighbors] covers edges (u, v) for all v in neighbors.
        # But (u, v) is covered by biclique u. Is it covered by biclique v?
        # If we use stars centered at every vertex, we double cover edges.
        # We need a partition.
        # Simple partition: For each edge (u,v), create a biclique [[u],[v]].
        # This uses |E| bicliques. Valid but bad score.
        # Better fallback: 
        # Iterate vertices 0..n-1.
        # For vertex u, let R_u be neighbors of u with index > u.
        # Biclique: [u, R_u]. This covers all edges (u, v) where v > u.
        # This partitions all edges exactly once.
        bicliques = []
        for u in range(n):
            R_u = [v for v in host_adj[u] if v > u]
            if R_u:
                bicliques.append([[u], sorted(R_u)])
        best_solution = bicliques
        best_count = len(bicliques)
        
    # Convert to required format
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
        
    return final_result

# EVOLVE-BLOCK-END
