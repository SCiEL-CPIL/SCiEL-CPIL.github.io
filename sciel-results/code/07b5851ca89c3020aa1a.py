import time
import random
import sys
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix and set
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)

    def is_biclique(L, R):
        """Check if L, R form a valid biclique in the original graph."""
        if not L or not R:
            return False
        if set(L).intersection(R):
            return False
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_set:
                    return False
        return True

    def get_biclique_edges(L, R):
        """Get the set of edges covered by biclique (L, R)."""
        es = set()
        for u in L:
            for v in R:
                es.add((min(u,v), max(u,v)))
        return es

    def find_maximal_biclique_containing(u, v, remaining_edges):
        """
        Find a maximal biclique containing edge (u,v) that only covers edges in remaining_edges.
        Actually, the biclique must be valid in G. The set of edges it covers must be a subset of remaining_edges?
        No, the partition constraint says the cross-edge sets must partition ALL pinned edges.
        So if we pick a biclique B, it covers E(B). E(B) must be a subset of remaining_edges?
        Yes, because if E(B) contains an edge not in remaining_edges, that edge was already covered by a previous biclique,
        which would violate the "disjoint" partition rule.
        So, we need to find a biclique B such that E(B) is a subset of remaining_edges.
        
        Algorithm:
        1. Start with L={u}, R={v}.
        2. Try to expand L: add x if x is adjacent to all in R AND all edges (x, r) for r in R are in remaining_edges.
           Wait, if (x,r) is in G but not in remaining_edges, we cannot add x to L because it would cover an already covered edge.
           So the condition for adding x to L is:
           For all r in R: (x, r) is in G AND (x, r) is in remaining_edges.
           Similarly for adding y to R:
           For all l in L: (l, y) is in G AND (l, y) is in remaining_edges.
        
        This ensures the new biclique only covers edges in remaining_edges.
        """
        L = {u}
        R = {v}
        
        # Pre-check: if (u,v) not in remaining_edges, return None
        if (min(u,v), max(u,v)) not in remaining_edges:
            return None

        changed = True
        while changed:
            changed = False
            # Try to add vertices to L
            candidates_L = set()
            for x in range(n):
                if x in L or x in R:
                    continue
                # Check if x can be added to L
                valid = True
                for r in R:
                    e = (min(x, r), max(x, r))
                    if e not in edge_set or e not in remaining_edges:
                        valid = False
                        break
                if valid:
                    candidates_L.add(x)
            
            if candidates_L:
                # Add one candidate (greedy: pick one with max degree in remaining? or just first)
                # To maximize size, we might want to be careful. But any maximal is fine for greedy.
                # Let's pick the one that allows the most expansion? For simplicity, pick first.
                x = next(iter(candidates_L))
                L.add(x)
                changed = True
            
            # Try to add vertices to R
            candidates_R = set()
            for y in range(n):
                if y in L or y in R:
                    continue
                valid = True
                for l in L:
                    e = (min(l, y), max(l, y))
                    if e not in edge_set or e not in remaining_edges:
                        valid = False
                        break
                if valid:
                    candidates_R.add(y)
            
            if candidates_R:
                y = next(iter(candidates_R))
                R.add(y)
                changed = True

        return list(L), list(R)

    def solve_greedy(remaining_edges, deadline):
        bicliques = []
        remaining = set(remaining_edges)
        
        while remaining:
            if time.time() > deadline:
                return None
            
            # Pick an edge from remaining
            # Heuristic: pick edge with lowest degree in remaining graph?
            # Or just random to allow restarts.
            u, v = next(iter(remaining))
            
            L, R = find_maximal_biclique_containing(u, v, remaining)
            
            if L is None:
                # Should not happen if (u,v) is in remaining
                return None
            
            # Validate
            if not is_biclique(L, R):
                return None
            
            covered = get_biclique_edges(L, R)
            
            # Ensure covered is subset of remaining
            if not covered.issubset(remaining):
                return None
            
            bicliques.append([L, R])
            remaining.difference_update(covered)
            
        return bicliques

    def local_search(bicliques, deadline):
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_b = len(bicliques)
            for i in range(n_b):
                if time.time() > deadline:
                    return bicliques
                for j in range(i + 1, n_b):
                    if time.time() > deadline:
                        return bicliques
                    
                    # Get edges covered by i and j
                    e_i = get_biclique_edges(bicliques[i][0], bicliques[i][1])
                    e_j = get_biclique_edges(bicliques[j][0], bicliques[j][1])
                    combined = e_i.union(e_j)
                    
                    # Try to merge i and j into a single biclique
                    # The new biclique must cover exactly 'combined'.
                    # We can try to find a maximal biclique in the subgraph induced by 'combined' edges?
                    # Or simply, pick an edge from combined, and try to expand it using only edges in 'combined'.
                    u, v = next(iter(combined))
                    
                    # We need a biclique (L, R) such that E(L, R) == combined.
                    # Use the same expansion logic but restricted to 'combined'.
                    L, R = find_maximal_biclique_containing(u, v, combined)
                    
                    if L is not None and is_biclique(L, R):
                        new_covered = get_biclique_edges(L, R)
                        if new_covered == combined:
                            # Merge successful
                            # Remove j, then i
                            bicliques.pop(j)
                            bicliques.pop(i)
                            bicliques.append([L, R])
                            improved = True
                            break
                
                if improved:
                    break
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    start_time = time.time()
    deadline = start_time + 80 # 10s margin
    
    restart = 0
    while time.time() < deadline:
        random.seed(restart)
        
        # Shuffle edges to change the order of selection
        shuffled_edges = list(edge_set)
        random.shuffle(shuffled_edges)
        
        result = solve_greedy(set(shuffled_edges), deadline)
        
        if result is not None:
            # Local Search
            result = local_search(result, deadline)
            
            if len(result) < best_count:
                best_count = len(result)
                best_bicliques = result
                
            if best_count == 1:
                break
                
        restart += 1

    # Fallback: Star decomposition (always valid)
    if best_bicliques is None:
        best_bicliques = []
        for u, v in edge_set:
            best_bicliques.append([[u], [v]])

    # Convert to required format
    final_result = []
    for L, R in best_bicliques:
        L_int = [int(x) for x in L]
        R_int = [int(x) for x in R]
        final_result.append([L_int, R_int])
        
    return final_result
