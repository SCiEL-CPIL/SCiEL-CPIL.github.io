import time
import sys

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = P['n']
    edges = P['edges']

    # Build adjacency as sets for fast lookup
    adj = [set() for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        edge_set.add((min(u, v), max(u, v)))

    def time_left():
        return deadline - time.time()

    def edge_key(u, v):
        return (min(u, v), max(u, v))

    def expand_biclique(L, R, remaining_edges):
        """Expand L and R to maximal biclique within remaining edges."""
        L = set(L)
        R = set(R)
        if not L or not R:
            return L, R
        changed = True
        while changed and time_left() > 0.5:
            changed = False
            # Try adding vertices to L
            for w in range(n):
                if w in L or w in R:
                    continue
                # w can join L if all edges (w, r) for r in R are in remaining
                ok = True
                for r in R:
                    if edge_key(w, r) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    L.add(w)
                    changed = True
            # Try adding vertices to R
            for w in range(n):
                if w in L or w in R:
                    continue
                ok = True
                for l in L:
                    if edge_key(w, l) not in remaining_edges:
                        ok = False
                        break
                if ok:
                    R.add(w)
                    changed = True
        return L, R

    def biclique_edge_set(L, R):
        """Return set of edges covered by biclique (L, R)."""
        es = set()
        for u in L:
            for v in R:
                if u != v:
                    es.add(edge_key(u, v))
        return es

    def greedy_partition(remaining):
        """Greedy: repeatedly find largest biclique from remaining edges."""
        bicliques = []
        rem = set(remaining)
        
        while rem and time_left() > 1.0:
            # Find best biclique seed
            best_L, best_R = None, None
            best_size = 0
            
            # Try each edge as seed, pick top by degree
            candidates = sorted(rem, key=lambda e: (-(len(adj[e[0]]) + len(adj[e[1]])), e[0], e[1]))
            
            tried = 0
            for (u, v) in candidates:
                if time_left() < 0.5:
                    break
                tried += 1
                if tried > 60:
                    break
                
                # Try u in L, v in R
                L1, R1 = expand_biclique({u}, {v}, rem)
                if L1 and R1:
                    sz = len(L1) * len(R1)
                    if sz > best_size:
                        best_size = sz
                        best_L, best_R = L1, R1
                
                # Try v in L, u in R
                L2, R2 = expand_biclique({v}, {u}, rem)
                if L2 and R2:
                    sz = len(L2) * len(R2)
                    if sz > best_size:
                        best_size = sz
                        best_L, best_R = L2, R2
            
            if best_L is None or best_R is None:
                # Fallback: single edge
                e = next(iter(rem))
                best_L, best_R = {e[0]}, {e[1]}
            
            bicliques.append((best_L, best_R))
            
            # Remove covered edges
            covered = biclique_edge_set(best_L, best_R)
            for e in covered:
                rem.discard(e)
        
        # Handle any remaining edges as single-edge bicliques
        for e in rem:
            bicliques.append(({e[0]}, {e[1]}))
        
        return bicliques

    def validate_partition(bicliques, all_edges):
        """Check that the partition is valid."""
        if not bicliques:
            return False
        covered = set()
        for L, R in bicliques:
            if not L or not R:
                return False
            Ls, Rs = set(L), set(R)
            if Ls & Rs:
                return False
            for u in L:
                for v in R:
                    e = edge_key(u, v)
                    if e not in all_edges:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == all_edges

    def local_improve(bicliques, all_edges):
        """Try to reduce biclique count via merge attempts and edge reassignment."""
        bicliques = [(set(L), set(R)) for L, R in bicliques]
        improved = True
        max_iter = 20
        
        while improved and time_left() > 2.0:
            improved = False
            for _ in range(max_iter):
                if time_left() < 2.0:
                    break
                # Try merging pairs of bicliques
                merged = False
                for i in range(len(bicliques)):
                    if time_left() < 2.0:
                        break
                    for j in range(i + 1, len(bicliques)):
                        if time_left() < 2.0:
                            break
                        L_i, R_i = bicliques[i]
                        L_j, R_j = bicliques[j]
                        
                        # Try merging: (L_i ∪ L_j, R_i ∪ R_j)
                        new_L = L_i | L_j
                        new_R = R_i | R_j
                        if new_L & new_R:
                            continue
                        # Check all cross edges exist and are in all_edges
                        valid = True
                        for u in new_L:
                            for v in new_R:
                                e = edge_key(u, v)
                                if e not in all_edges:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        
                        # Check no overlap with other bicliques' edges
                        new_edges = biclique_edge_set(new_L, new_R)
                        other_edges = set()
                        for k in range(len(bicliques)):
                            if k == i or k == j:
                                other_edges |= biclique_edge_set(bicliques[k][0], bicliques[k][1])
                        
                        if new_edges & other_edges:
                            continue
                        
                        # Valid merge!
                        new_bicliques = []
                        for k in range(len(bicliques)):
                            if k != i and k != j:
                                new_bicliques.append(bicliques[k])
                        new_bicliques.append((new_L, new_R))
                        
                        if len(new_bicliques) < len(bicliques):
                            bicliques = new_bicliques
                            improved = True
                            merged = True
                            break
                    if merged:
                        break
                if merged:
                    continue
                
                # Try reassigning a single edge from one biclique to another
                for i in range(len(bicliques)):
                    if time_left() < 2.0:
                        break
                    L_i, R_i = bicliques[i]
                    if len(L_i) * len(R_i) < 2:
                        continue
                    # Get all edges in this biclique
                    edges_i = biclique_edge_set(L_i, R_i)
                    if len(edges_i) <= 1:
                        continue
                    
                    for e in edges_i:
                        if time_left() < 2.0:
                            break
                        u, v = e
                        # Try moving this edge to another biclique
                        for j in range(len(bicliques)):
                            if j == i:
                                continue
                            L_j, R_j = bicliques[j]
                            
                            # Can we add u to L_j and v to R_j (or vice versa)?
                            # Option 1: u in L_j, v in R_j
                            new_L = L_j | {u}
                            new_R = R_j | {v}
                            if new_L & new_R:
                                continue
                            valid = True
                            for a in new_L:
                                for b in new_R:
                                    if edge_key(a, b) not in all_edges:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if not valid:
                                continue
                            
                            # Check that removing edge e from biclique i still leaves valid biclique
                            # We need to remove u from L_i or v from R_i
                            # This is complex; skip for now and just try adding to j
                            
                            # Actually, let's just try: can biclique j absorb edge (u,v)?
                            # i.e., is (u,v) already covered by j? No, it's in i.
                            # We'd need to move it. This requires restructuring.
                            # For simplicity, skip edge reassignment and rely on merges.
                            break
                        
                        # Try option 2: v in L_j, u in R_j
                        new_L = L_j | {v}
                        new_R = R_j | {u}
                        if new_L & new_R:
                            continue
                        valid = True
                        for a in new_L:
                            for b in new_R:
                                if edge_key(a, b) not in all_edges:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            continue
                        break
                # End edge reassignment (simplified)
                
                if not improved:
                    break
        
        return bicliques

    def refine_partition(bicliques, all_edges):
        """Local search to reduce biclique count by merging or moving vertices."""
        bicliques = [(set(L), set(R)) for L, R in bicliques]
        improved = True
        max_iterations = 500
        
        while improved and time_left() > 1.0:
            improved = False
            for _ in range(max_iterations):
                if time_left() < 1.0:
                    break
                
                # 1. Try to merge any two bicliques
                merged = False
                for i in range(len(bicliques)):
                    if time_left() < 1.0:
                        break
                    for j in range(i + 1, len(bicliques)):
                        if time_left() < 1.0:
                            break
                        L_i, R_i = bicliques[i]
                        L_j, R_j = bicliques[j]
                        
                        # Attempt 1: (L_i U L_j, R_i U R_j)
                        new_L = L_i | L_j
                        new_R = R_i | R_j
                        if not (new_L & new_R):
                            valid = True
                            for u in new_L:
                                for v in new_R:
                                    if edge_key(u, v) not in all_edges:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                # Check coverage conflict? 
                                # If we merge, the union of edges must be exactly the union of the two biclique edges.
                                # Since they are disjoint in the partition, this holds if the new biclique is valid.
                                new_bicliques = []
                                for k in range(len(bicliques)):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                new_bicliques.append((new_L, new_R))
                                bicliques = new_bicliques
                                improved = True
                                merged = True
                                break
                        
                        # Attempt 2: (L_i U R_j, R_i U L_j)
                        new_L = L_i | R_j
                        new_R = R_i | L_j
                        if not (new_L & new_R):
                            valid = True
                            for u in new_L:
                                for v in new_R:
                                    if edge_key(u, v) not in all_edges:
                                        valid = False
                                        break
                                if not valid:
                                    break
                            if valid:
                                new_bicliques = []
                                for k in range(len(bicliques)):
                                    if k != i and k != j:
                                        new_bicliques.append(bicliques[k])
                                new_bicliques.append((new_L, new_R))
                                bicliques = new_bicliques
                                improved = True
                                merged = True
                                break
                    if merged:
                        break
                
                if merged:
                    continue

                # 2. Try to move a vertex from one side of a biclique to the other, or to another biclique
                # This is complex and prone to breaking validity. 
                # A simpler improvement: "Squeeze" - if a vertex in L_i has no edges to R_i except those already covered? 
                # No, every vertex in L_i must have edges to ALL vertices in R_i.
                
                # Alternative: Remove redundant vertices? No, vertices are part of the structure.
                
                # If no merge was found, try moving a single edge's endpoint?
                # This is effectively re-partitioning. 
                # Given the time constraint and complexity, merging is the most robust "structural" improvement.
                # If we can't merge, we stop.
                break
        
        return bicliques

    # Phase 1: Greedy to get initial solution
    initial = greedy_partition(edge_set)
    
    # Phase 2: Refine using local search (merging)
    refined = refine_partition(initial, edge_set)
    
    # Validate
    if validate_partition(refined, edge_set):
        best_result = refined
    else:
        # Fallback: single-edge bicliques (always valid)
        best_result = [({u}, {v}) for u, v in edge_set]
    
    # Convert to required format with strict types
    result = []
    for L, R in best_result:
        # Ensure integers are Python ints
        left = [int(v) for v in L]
        right = [int(v) for v in R]
        result.append([left, right])
    
    return result
