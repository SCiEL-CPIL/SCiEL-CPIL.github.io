import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 85
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u,v), max(u,v)))
    
    all_edges = list(edge_set)
    total_edges = len(all_edges)
    
    # Helper: check if a set of edges forms a biclique
    # Returns (left, right) if valid, else None
    def edges_to_biclique(edge_subset):
        if not edge_subset:
            return None
        # Get all vertices involved
        vertices = set()
        for u, v in edge_subset:
            vertices.add(u)
            vertices.add(v)
        
        # We need to partition vertices into L and R such that all edges are between L and R
        # and there are no edges within L or within R (wait, "Within-side edges are allowed" in task description?
        # "Every cross pair must be a host edge. The cross-edge sets must partition all pinned graph edges exactly once. Within-side edges are allowed."
        # This implies:
        # 1. If u in L, v in R, then (u,v) MUST be an edge.
        # 2. The set of all such cross pairs MUST be exactly the set of edges in the biclique.
        # 3. Edges within L or within R are IGNORED for the partition requirement?
        # "The cross-edge sets must partition all pinned graph edges exactly once."
        # This means the set of edges covered by the biclique is EXACTLY { (u,v) | u in L, v in R }.
        # If there is an edge (u,w) with u,w in L, it is NOT part of the cross-edge set of this biclique.
        # It must be covered by another biclique.
        # So, the condition is:
        # Let E_cross(L,R) = { (u,v) | u in L, v in R }.
        # We require E_cross(L,R) == edge_subset.
        # This implies:
        # 1. For all u in L, v in R, (u,v) is in edge_set.
        # 2. For all u in L, w in L, (u,w) is NOT in edge_subset.
        # 3. For all v in R, w in R, (v,w) is NOT in edge_subset.
        
        # To find such a partition:
        # The graph induced by edge_subset must be bipartite.
        # Let G_sub be the graph with vertices 'vertices' and edges 'edge_subset'.
        # If G_sub is not bipartite, return None.
        # If it is bipartite, the two color classes are the only possible candidates for L and R (or swapped).
        # However, we must also ensure that there are NO edges in G_sub within the same color class (which is true by definition of bipartite coloring).
        # But we also need to ensure that ALL pairs between L and R are edges in G_sub.
        # i.e., G_sub must be a complete bipartite graph K_{|L|, |R|}.
        
        # Check if G_sub is a complete bipartite graph.
        # 1. Bipartite check
        color = {}
        for v in vertices:
            if v not in color:
                stack = [v]
                color[v] = 0
                while stack:
                    curr = stack.pop()
                    for nb in adj[curr]:
                        if nb in vertices:
                            # Check if edge (curr, nb) is in edge_subset
                            if (min(curr, nb), max(curr, nb)) in edge_subset:
                                if nb not in color:
                                    color[nb] = 1 - color[curr]
                                    stack.append(nb)
                                elif color[nb] == color[curr]:
                                    return None # Not bipartite
        
        # 2. Check completeness
        L = [v for v in vertices if color[v] == 0]
        R = [v for v in vertices if color[v] == 1]
        
        # Verify all cross pairs are in edge_subset
        for u in L:
            for v in R:
                if (min(u,v), max(u,v)) not in edge_subset:
                    return None
        
        # Verify no edges within L or R in edge_subset
        # (This is guaranteed if the bipartite check passed on the subgraph defined by edge_subset)
        # Wait, the bipartite check only looks at edges IN edge_subset.
        # If there is an edge (u,w) in edge_subset with u,w in L, the bipartite check would have failed (odd cycle or direct conflict).
        # So if it's bipartite, there are no edges within L or within R in edge_subset.
        
        return [L, R]

    # Greedy Set Cover Initialization
    def greedy_init():
        remaining = set(all_edges)
        bicliques = []
        while remaining:
            best_bic = None
            best_count = 0
            # Try to find a biclique covering the most edges from 'remaining'
            # Heuristic: For each edge in remaining, find maximal biclique containing it?
            # Or just pick a random edge, find maximal biclique containing it, add it.
            # To be better, sample a few edges.
            
            sample_edges = list(remaining)
            random.shuffle(sample_edges)
            # Check top few samples
            candidates = []
            for e in sample_edges[:10]:
                # Find maximal biclique containing e
                # Start with L={u}, R={v}
                # Expand L and R as much as possible while maintaining biclique property
                # This is a local search on the biclique structure.
                
                u, v = e
                L = {u}
                R = {v}
                
                # Iteratively add vertices
                changed = True
                while changed:
                    changed = False
                    # Try to add a vertex to L
                    # Candidate x can be added to L if (x, r) is in remaining for all r in R
                    for x in range(n):
                        if x not in L and x not in R:
                            valid = True
                            for r in R:
                                edge = (min(x, r), max(x, r))
                                if edge not in remaining:
                                    valid = False
                                    break
                            if valid:
                                L.add(x)
                                changed = True
                    # Try to add a vertex to R
                    # Candidate y can be added to R if (l, y) is in remaining for all l in L
                    for y in range(n):
                        if y not in L and y not in R:
                            valid = True
                            for l in L:
                                edge = (min(l, y), max(l, y))
                                if edge not in remaining:
                                    valid = False
                                    break
                            if valid:
                                R.add(y)
                                changed = True
                            
                # Construct the edge set of this biclique
                # By construction, all (l, r) in L x R are in remaining
                bic_edges = set()
                for l in L:
                    for r in R:
                        bic_edges.add((min(l,r), max(l,r)))
                
                # Count how many of these are in 'remaining' (should be all)
                count = len(bic_edges & remaining)
                if count > best_count:
                    best_count = count
                    best_bic = [list(L), list(R)]
                    
            if best_bic is None:
                # Fallback: single edge
                u, v = next(iter(remaining))
                best_bic = [[u], [v]]
                
            bicliques.append(best_bic)
            # Remove covered edges
            covered = set()
            L, R = best_bic
            for l in L:
                for r in R:
                    covered.add((min(l,r), max(l,r)))
            remaining -= covered
            
        return bicliques

    # Local Search: Merge
    def try_merge(bicliques, adj, edge_set):
        n_bic = len(bicliques)
        for i in range(n_bic):
            for j in range(i+1, n_bic):
                L1, R1 = bicliques[i]
                L2, R2 = bicliques[j]
                
                # Get all edges in biclique i and j
                edges_i = set()
                for u in L1:
                    for v in R1:
                        edges_i.add((min(u,v), max(u,v)))
                
                edges_j = set()
                for u in L2:
                    for v in R2:
                        edges_j.add((min(u,v), max(u,v)))
                
                union_edges = edges_i | edges_j
                
                # Check if union_edges forms a biclique
                merged = edges_to_biclique(union_edges)
                if merged:
                    new_bicliques = [b for k, b in enumerate(bicliques) if k != i and k != j]
                    new_bicliques.append(merged)
                    return new_bicliques
        return None

    # Main Loop
    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts of Greedy Init + Local Search
    for restart in range(20):
        if time.time() > deadline - 10:
            break
            
        bicliques = greedy_init()
        
        # Local Search
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            new_bicliques = try_merge(bicliques, adj, edge_set)
            if new_bicliques is not None:
                bicliques = new_bicliques
                improved = True
                
        if len(bicliques) < best_count:
            best_count = len(bicliques)
            best_bicliques = bicliques
            
    # Fallback if no solution found (should not happen)
    if best_bicliques is None:
        best_bicliques = [[[u], [v]] for u, v in all_edges]
        
    # Convert to proper format
    result = []
    for left, right in best_bicliques:
        result.append([list(left), list(right)])
        
    return result

# EVOLVE-BLOCK-END
