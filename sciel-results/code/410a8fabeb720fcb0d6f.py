import time
import random
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Normalize edges
    edge_set = set()
    for u, v in edges:
        if u > v: u, v = v, u
        edge_set.add((u, v))
        
    # Build adjacency list for the full graph
    full_adj = [set() for _ in range(n)]
    for u, v in edge_set:
        full_adj[u].add(v)
        full_adj[v].add(u)

    def get_biclique_edges(L, R):
        es = set()
        for u in L:
            for v in R:
                if u == v: continue
                a, b = (u, v) if u < v else (v, u)
                es.add((a, b))
        return es

    def is_biclique(L, R, rem_edges):
        # Check if all cross pairs are in rem_edges
        for u in L:
            for v in R:
                if u == v: continue
                a, b = (u, v) if u < v else (v, u)
                if (a, b) not in rem_edges:
                    return False
        return True

    def find_large_biclique(rem_edges, adj_rem):
        # Heuristic: Pick a vertex with high degree, then expand
        if not rem_edges:
            return None, None, set()
        
        # Find vertex with max degree in remaining graph
        degs = [0] * n
        for u, v in rem_edges:
            degs[u] += 1
            degs[v] += 1
            
        # Try top 5 vertices by degree
        candidates = sorted(range(n), key=lambda x: degs[x], reverse=True)
        top_candidates = [c for c in candidates if degs[c] > 0][:5]
        
        best_biclique = None
        best_size = 0
        best_edges = set()
        
        for u_start in top_candidates:
            # Initialize L={u_start}, R=N(u_start)
            L = {u_start}
            R = set(adj_rem[u_start])
            
            # Expand greedily
            changed = True
            while changed:
                changed = False
                # Try to add to L
                # A vertex x can be added to L if it is connected to all in R
                for x in range(n):
                    if x in L or x in R: continue
                    # Check if x is connected to all in R
                    valid = True
                    for r in R:
                        a, b = (x, r) if x < r else (r, x)
                        if (a, b) not in rem_edges:
                            valid = False
                            break
                    if valid:
                        L.add(x)
                        changed = True
                        
                # Try to add to R
                # A vertex y can be added to R if it is connected to all in L
                for y in range(n):
                    if y in L or y in R: continue
                    valid = True
                    for l in L:
                        a, b = (y, l) if y < l else (l, y)
                        if (a, b) not in rem_edges:
                            valid = False
                            break
                    if valid:
                        R.add(y)
                        changed = True
            
            # Calculate edges covered
            edges_covered = get_biclique_edges(L, R)
            # Ensure all these edges are actually in rem_edges (should be by construction)
            if edges_covered.issubset(rem_edges):
                size = len(edges_covered)
                if size > best_size:
                    best_size = size
                    best_biclique = (L, R)
                    best_edges = edges_covered
                    
        return best_biclique, best_edges

    # Greedy Partition
    rem_edges = set(edge_set)
    result_bicliques = []
    
    # Build adjacency for remaining edges initially
    adj_rem = [set() for _ in range(n)]
    for u, v in rem_edges:
        adj_rem[u].add(v)
        adj_rem[v].add(u)
        
    start_time = time.time()
    
    while rem_edges:
        if time.time() - start_time > 55:
            # Fallback: just return whatever we have plus individual edges for the rest?
            # No, must be valid. If we run out of time, we should return a valid solution.
            # The current loop is greedy, so it's fast.
            break
            
        L, R, edges_to_remove = find_large_biclique(rem_edges, adj_rem)
        
        if not L or not R:
            # No biclique found? This shouldn't happen if rem_edges is not empty.
            # Maybe a single edge?
            u, v = next(iter(rem_edges))
            L = {u}
            R = {v}
            edges_to_remove = {(u, v) if u < v else (v, u)}
            
        result_bicliques.append([list(L), list(R)])
        
        # Remove edges from rem_edges and update adj_rem
        for e in edges_to_remove:
            u, v = e
            rem_edges.discard(e)
            adj_rem[u].discard(v)
            adj_rem[v].discard(u)
            
    # If we timed out and still have edges, we must handle them to be valid.
    # But the loop breaks on timeout. We need to ensure we return a valid partition.
    # If we break, we still have rem_edges. We must partition them.
    # Since we are in a tight loop, let's just add the remaining edges as individual bicliques.
    # This ensures validity.
    if rem_edges:
        for u, v in rem_edges:
            result_bicliques.append([[u], [v]])

    # Convert to required format
    final_result = []
    for L, R in result_bicliques:
        left = [int(x) for x in L]
        right = [int(x) for x in R]
        final_result.append([left, right])
        
    return final_result
