import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    n = P['n']
    edges = P['edges']
    
    adj = [[False] * n for _ in range(n)]
    edge_set = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        adj[u][v] = True
        adj[v][u] = True
        edge_set.add((u, v))
    
    total_edges = len(edge_set)
    
    def make_edge(u, v):
        return (min(u, v), max(u, v))
        
    def find_max_biclique(remaining_edges):
        if not remaining_edges:
            return None
            
        best_bc = None
        best_size = 0
        
        for (u, v) in remaining_edges:
            L = {u}
            R = {v}
            
            changed = True
            while changed:
                changed = False
                new_L = set()
                for w in range(n):
                    if w in L or w in R:
                        continue
                    valid = True
                    for r in R:
                        if make_edge(w, r) not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        new_L.add(w)
                if new_L:
                    L |= new_L
                    changed = True
                    
                new_R = set()
                for w in range(n):
                    if w in L or w in R:
                        continue
                    valid = True
                    for l in L:
                        if make_edge(w, l) not in remaining_edges:
                            valid = False
                            break
                    if valid:
                        new_R.add(w)
                if new_R:
                    R |= new_R
                    changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_bc = (L, R)
                
        return best_bc

    def greedy_partition(edge_set):
        bicliques = []
        remaining = set(edge_set)
        
        while remaining:
            if time.time() > deadline:
                break
            bc = find_max_biclique(remaining)
            if bc is None:
                for e in remaining:
                    u, v = e
                    bicliques.append((frozenset([u]), frozenset([v])))
                remaining = set()
                break
            
            L, R = bc
            bicliques.append((frozenset(L), frozenset(R)))
            
            to_remove = set()
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        if e in remaining:
                            to_remove.add(e)
            remaining -= to_remove
            
        return bicliques

    def local_search(bicliques, edge_set):
        # bicliques: list of (L, R)
        # Build edge_to_biclique
        edge_to_biclique = {}
        for i, (L, R) in enumerate(bicliques):
            for l in L:
                for r in R:
                    if l != r:
                        e = make_edge(l, r)
                        edge_to_biclique[e] = i
                        
        current_bcs = [list(bc) for bc in bicliques]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(current_bcs)):
                if time.time() > deadline:
                    return current_bcs
                L_i, R_i = current_bcs[i]
                L_i = set(L_i)
                R_i = set(R_i)
                
                # Try adding v to L_i
                for v in range(n):
                    if v in L_i or v in R_i:
                        continue
                    
                    valid = True
                    for r in R_i:
                        if not adj[v][r]:
                            valid = False
                            break
                    
                    if not valid:
                        continue
                    
                    # Simulate move
                    new_L_i = L_i | {v}
                    # Edges gained: (v, r) for r in R_i
                    # Check if this reduces biclique count
                    # We need to see if any biclique becomes empty.
                    
                    # For simplicity, let's just try the move and validate.
                    # But we need to update edge_to_biclique.
                    
                    # Let's collect edges to steal
                    stolen_edges = []
                    for r in R_i:
                        e = make_edge(v, r)
                        if e in edge_set:
                            stolen_edges.append(e)
                            
                    # Check if stealing these edges empties any biclique
                    # and if the new configuration is valid.
                    
                    # This is a bit heavy. Let's just try it.
                    
                    # Create a copy
                    new_bcs = [list(bc) for bc in current_bcs]
                    new_edge_to_biclique = dict(edge_to_biclique)
                    
                    # Update new_bcs[i]
                    new_bcs[i] = [list(new_L_i), list(R_i)]
                    
                    # Update edge ownership
                    for e in stolen_edges:
                        if e in new_edge_to_biclique:
                            old_owner = new_edge_to_biclique[e]
                            # Remove e from old_owner's coverage?
                            # We don't track coverage per biclique explicitly, just via L, R.
                            # So we just change the owner.
                            new_edge_to_biclique[e] = i
                            
                    # Check if any biclique is now empty
                    # A biclique is empty if it has no edges.
                    # We can check this by iterating its L, R and seeing if any edge is owned by it.
                    # This is O(K * N^2).
                    
                    # Let's just validate the whole partition.
                    # If valid and count is same or less, accept.
                    # Actually, if we steal edges, the count might stay same, but we might enable future merges.
                    # To be safe, only accept if count decreases.
                    
                    # Count check:
                    # If any biclique j != i has no edges owned by it, it can be removed.
                    # Let's check if any j != i has 0 edges.
                    
                    counts = [0] * len(new_bcs)
                    for e, owner in new_edge_to_biclique.items():
                        counts[owner] += 1
                        
                    non_empty = [k for k in range(len(new_bcs)) if counts[k] > 0]
                    
                    if len(non_empty) < len(current_bcs):
                        # Improvement!
                        # Reconstruct new_bcs and edge_to_biclique
                        current_bcs = [new_bcs[k] for k in non_empty]
                        # Rebuild edge_to_biclique for the new list
                        new_edge_to_biclique = {}
                        for idx, k in enumerate(non_empty):
                            L_k, R_k = current_bcs[idx]
                            for l in L_k:
                                for r in R_k:
                                    if l != r:
                                        e = make_edge(l, r)
                                        if e in edge_set:
                                            new_edge_to_biclique[e] = idx
                        edge_to_biclique = new_edge_to_biclique
                        improved = True
                        break # Restart loop
                        
                if improved:
                    break
                    
        return current_bcs

    def validate_partition(bcs, edge_set):
        covered = set()
        for L, R in bcs:
            L = set(L)
            R = set(R)
            if L & R:
                return False
            if not L or not R:
                return False
            for l in L:
                for r in R:
                    if l == r:
                        continue
                    e = make_edge(l, r)
                    if e not in edge_set:
                        return False
                    if e in covered:
                        return False
                    covered.add(e)
        return covered == edge_set

    best_result = None
    best_count = total_edges + 1
    
    for run in range(20):
        if time.time() > deadline:
            break
        
        bcs = greedy_partition(edge_set)
        
        if len(bcs) < best_count:
            best_count = len(bcs)
            best_result = bcs
            
        improved_bcs = local_search(bcs, edge_set)
        
        if len(improved_bcs) < best_count:
            best_count = len(improved_bcs)
            best_result = improved_bcs

    if best_result is None:
        result = []
        for u, v in edge_set:
            result.append([[u], [v]])
        return result

    result = []
    for L, R in best_result:
        result.append([list(L), list(R)])
    
    return result

