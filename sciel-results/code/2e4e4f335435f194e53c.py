"""Biclique partitioning using backtracking with forward checking."""
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']
    
    # Build edge set (canonical form)
    all_edges = []
    for u, v in edges:
        if u < v:
            all_edges.append((u, v))
        else:
            all_edges.append((v, u))
    
    total_edges = len(all_edges)
    
    # Adjacency for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Edge set for O(1) lookup
    edge_set = set(all_edges)
    
    def is_edge(u, v):
        if u == v:
            return False
        if u < v:
            return (u, v) in edge_set
        else:
            return (v, u) in edge_set
    
    # Greedy initial solution to get a starting upper bound
    def greedy_solution():
        remaining = set(all_edges)
        bicliques = []
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Find a maximal biclique in remaining edges
            # Pick vertex with max degree in remaining
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            if not deg:
                break
                
            start = max(deg, key=lambda x: deg[x])
            
            # Get neighbors of start in remaining
            neighbors = set()
            for u, v in remaining:
                if u == start:
                    neighbors.add(v)
                elif v == start:
                    neighbors.add(u)
            
            if not neighbors:
                # Isolated edge
                u, v = next(iter(remaining))
                bicliques.append(([u], [v]))
                remaining.discard((min(u,v), max(u,v)))
                continue
            
            # Try both orientations and find the larger biclique
            best_L, best_R = set(), set()
            best_size = 0
            
            for orient in [0, 1]:
                if orient == 0:
                    L = {start}
                    R = set(neighbors)
                else:
                    L = set(neighbors)
                    R = {start}
                
                # Refine iteratively
                changed = True
                while changed:
                    changed = False
                    new_R = set()
                    for v in R:
                        ok = True
                        for u in L:
                            if not is_edge(u, v):
                                ok = False
                                break
                        if ok:
                            new_R.add(v)
                        else:
                            changed = True
                    R = new_R
                    
                    new_L = set()
                    for u in L:
                        ok = True
                        for v in R:
                            if not is_edge(u, v):
                                ok = False
                                break
                        if ok:
                            new_L.add(u)
                        else:
                            changed = True
                    L = new_L
                
                if L and R:
                    size = len(L) * len(R)
                    if size > best_size:
                        best_size = size
                        best_L, best_R = L, R
            
            if not best_L or not best_R:
                u, v = next(iter(remaining))
                best_L, best_R = {u}, {v}
            
            # Remove covered edges
            for u in best_L:
                for v in best_R:
                    e = (min(u, v), max(u, v))
                    remaining.discard(e)
            
            bicliques.append((list(best_L), list(best_R)))
        
        return bicliques
    
    # Get initial greedy solution
    best_bicliques = greedy_solution()
    best_count = len(best_bicliques)
    
    if time.time() > deadline - 10.0:
        return [[list(L), list(R)] for L, R in best_bicliques]
    
    # Backtracking search with constraint propagation
    # State: list of bicliques (each is (L_set, R_set)), and remaining edges
    # We assign edges one at a time to existing bicliques or create new ones
    
    # Precompute: for each pair of vertices, whether they form an edge
    # Use bitmasks for efficiency with 30 vertices
    
    # Represent biclique by (L_mask, R_mask) where masks are integers
    def masks_to_biclique(L_mask, R_mask):
        L = [i for i in range(n) if (L_mask >> i) & 1]
        R = [i for i in range(n) if (R_mask >> i) & 1]
        return L, R
    
    def can_add_edge_to_biclique(edge, L_mask, R_mask):
        """Check if edge (u,v) can be added to biclique (L,R).
        Edge (u,v) can go in if:
        - u in L, v in R and all (u',v) for u' in L are edges, all (u,v') for v' in R are edges
        - OR v in L, u in R and similar
        - OR u not in L∪R and v not in L∪R: then u can go to L (need all edges to R) or to R (need all edges to L)
        - OR one is in and other isn't
        """
        u, v = edge
        
        u_in_L = (L_mask >> u) & 1
        u_in_R = (R_mask >> u) & 1
        v_in_L = (L_mask >> v) & 1
        v_in_R = (R_mask >> v) & 1
        
        # Case 1: both already in the biclique (one in L, one in R)
        if (u_in_L and v_in_R) or (v_in_L and u_in_R):
            return True  # Edge already implied
        
        # Case 2: u in L, v not in L∪R -> v must go to R
        if u_in_L and not v_in_L and not v_in_R:
            # Check v connects to all in L
            for i in range(n):
                if (L_mask >> i) & 1 and not is_edge(i, v):
                    return False
            return True
        
        # Case 3: u in R, v not in L∪R -> v must go to L
        if u_in_R and not v_in_L and not v_in_R:
            # Check v connects to all in R
            for i in range(n):
                if (R_mask >> i) & 1 and not is_edge(i, v):
                    return False
            return True
        
        # Case 4: v in L, u not in L∪R -> u must go to R
        if v_in_L and not u_in_L and not u_in_R:
            for i in range(n):
                if (L_mask >> i) & 1 and not is_edge(i, u):
                    return False
            return True
        
        # Case 5: v in R, u not in L∪R -> u must go to L
        if v_in_R and not u_in_L and not u_in_R:
            for i in range(n):
                if (R_mask >> i) & 1 and not is_edge(i, u):
                    return False
            return True
        
        # Case 6: neither u nor v in L∪R
        if not u_in_L and not u_in_R and not v_in_L and not v_in_R:
            # u can go to L (if connects to all R) and v to R (if connects to all L)
            opt1 = True
            for i in range(n):
                if (R_mask >> i) & 1 and not is_edge(u, i):
                    opt1 = False
                    break
            if opt1:
                for i in range(n):
                    if (L_mask >> i) & 1 and not is_edge(i, v):
                        opt1 = False
                        break
            if opt1:
                return True
            
            # Or u to R, v to L
            opt2 = True
            for i in range(n):
                if (L_mask >> i) & 1 and not is_edge(u, i):
                    opt2 = False
                    break
            if opt2:
                for i in range(n):
                    if (R_mask >> i) & 1 and not is_edge(i, v):
                        opt2 = False
                        break
            if opt2:
                return True
            
            return False
        
        # Case 7: both in same side (both in L or both in R) - can't add
        if u_in_L and v_in_L:
            return False
        if u_in_R and v_in_R:
            return False
        
        # Edge case: one vertex is in both L and R (shouldn't happen with valid bicliques)
        return False
    
    def add_edge_to_biclique(edge, L_mask, R_mask):
        """Add edge to biclique, return new (L_mask, R_mask) or None if impossible."""
        u, v = edge
        
        u_in_L = (L_mask >> u) & 1
        u_in_R = (R_mask >> u) & 1
        v_in_L = (L_mask >> v) & 1
        v_in_R = (R_mask >> v) & 1
        
        if (u_in_L and v_in_R) or (v_in_L and u_in_R):
            return (L_mask, R_mask)  # Already covered
        
        if u_in_L and not v_in_L and not v_in_R:
            return (L_mask, R_mask | (1 << v))
        
        if u_in_R and not v_in_L and not v_in_R:
            return (L_mask | (1 << v), R_mask)
        
        if v_in_L and not u_in_L and not u_in_R:
            return (L_mask, R_mask | (1 << u))
        
        if v_in_R and not u_in_L and not u_in_R:
            return (L_mask | (1 << u), R_mask)
        
        if not u_in_L and not u_in_R and not v_in_L and not v_in_R:
            # Try u->L, v->R
            opt1 = True
            for i in range(n):
                if (R_mask >> i) & 1 and not is_edge(u, i):
                    opt1 = False
                    break
            if opt1:
                for i in range(n):
                    if (L_mask >> i) & 1 and not is_edge(i, v):
                        opt1 = False
                        break
            if opt1:
                return (L_mask | (1 << u), R_mask | (1 << v))
            
            # Try u->R, v->L
            opt2 = True
            for i in range(n):
                if (L_mask >> i) & 1 and not is_edge(u, i):
                    opt2 = False
                    break
            if opt2:
                for i in range(n):
                    if (R_mask >> i) & 1 and not is_edge(i, v):
                        opt2 = False
                        break
            if opt2:
                return (L_mask | (1 << v), R_mask | (1 << u))
        
        return None
    
    # Convert greedy solution to bitmask form
    def bicliques_to_masks(bicliques):
        masks = []
        for L, R in bicliques:
            L_mask = 0
            for v in L:
                L_mask |= (1 << v)
            R_mask = 0
            for v in R:
                R_mask |= (1 << v)
            masks.append((L_mask, R_mask))
        return masks
    
    # Backtracking search
    def backtrack(remaining_edges, biclique_masks, depth):
        """Try to assign remaining edges to bicliques. Returns True if successful."""
        nonlocal best_bicliques, best_count
        
        if time.time() > deadline:
            return False
        
        if not remaining_edges:
            # Found a complete solution
            if len(biclique_masks) < best_count:
                best_count = len(biclique_masks)
                best_bicliques = [(masks_to_biclique(lm, rm)) for lm, rm in biclique_masks]
            return True
        
        # Pruning: if current biclique count + minimum additional needed >= best_count
        # Minimum additional: at least 1 if there are remaining edges
        if len(biclique_masks) + 1 >= best_count and remaining_edges:
            # We could still tie, but can't improve. Only continue if we want ties.
            # Actually we can't improve, so prune.
            return False
        
        # Select the next edge to assign (most constrained first)
        # Count how many bicliques each remaining edge can go into
        edge_constraints = {}
        for e in remaining_edges:
            count = 0
            for lm, rm in biclique_masks:
                if can_add_edge_to_biclique(e, lm, rm):
                    count += 1
            # Also count "new biclique" option
            edge_constraints[e] = count
        
        # Pick edge with fewest options (MRV heuristic)
        edge = min(remaining_edges, key=lambda e: edge_constraints[e])
        
        remaining_set = set(remaining_edges)
        remaining_set.discard(edge)
        
        # Option 1: assign edge to each existing biclique
        for i, (lm, rm) in enumerate(biclique_masks):
            if not can_add_edge_to_biclique(edge, lm, rm):
                continue
            
            new_lm, new_rm = add_edge_to_biclique(edge, lm, rm)
            new_masks = list(biclique_masks)
            new_masks[i] = (new_lm, new_rm)
            
            if backtrack(list(remaining_set), new_masks, depth + 1):
                return True
        
        # Option 2: create a new biclique with just this edge
        u, v = edge
        new_L_mask = 1 << u
        new_R_mask = 1 << v
        
        new_masks = list(biclique_masks)
        new_masks.append((new_L_mask, new_R_mask))
        
        if backtrack(list(remaining_set), new_masks, depth + 1):
            return True
        
        return False
    
    # Try backtracking from the greedy solution
    initial_masks = bicliques_to_masks(best_bicliques)
    
    # Sort edges by degree (process high-degree edges first for better pruning)
    deg = {}
    for u, v in all_edges:
        deg[u] = deg.get(u, 0) + 1
        deg[v] = deg.get(v, 0) + 1
    
    sorted_edges = sorted(all_edges, key=lambda e: -(deg[e[0]] + deg[e[1]]))
    
    # Start with empty biclique list and try to build up
    # But first, let's try to improve on the greedy solution
    
    # Alternative: start with no bicliques and build from scratch
    # This is more expensive but can find better solutions
    
    # Let's try a limited backtracking: start with greedy, then try to merge bicliques
    # Or: do full backtracking with time limit
    
    # Full backtracking from scratch with time limit
    found = backtrack(sorted_edges, [], 0)
    
    if time.time() > deadline - 5.0:
        return [[list(L), list(R)] for L, R in best_bicliques]
    
    # If backtracking didn't find anything better, try local search
    # Try merging pairs of bicliques
    def try_merge(bicliques):
        """Try to merge two bicliques into one."""
        bicliques = [list(b) for b in bicliques]
        improved = True
        while improved:
            improved = False
            for i in range(len(bicliques)):
                if time.time() > deadline - 5.0:
                    return bicliques
                for j in range(i + 1, len(bicliques)):
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    # Try all 4 combinations of merging
                    # (L1∪L2, R1∪R2), (L1∪R2, R1∪L2), (L1∪L2, R1), etc.
                    # Actually, we need the merged set to form a valid biclique
                    # containing all edges from both bicliques
                    
                    # Option A: L = L1∪L2, R = R1∪R2
                    # Check: all edges between L and R must exist
                    L = set(L1) | set(L2)
                    R = set(R1) | set(R2)
                    if L & R:  # Must be disjoint
                        continue
                    valid = True
                    for u in L:
                        for v in R:
                            if not is_edge(u, v):
                                valid = False
                                break
                        if not valid:
                            break
                    if valid and L and R:
                        # Verify all original edges are covered
                        bicliques[j] = (list(L), list(R))
                        del bicliques[i]
                        improved = True
                        break
                
                if improved:
                    break
        
        return bicliques
    
    # Convert best_bicliques to list format
    bc = [list(b) for b in best_bicliques]
    bc = try_merge(bc)
    if len(bc) < len(best_bicliques):
        best_bicliques = bc
        best_count = len(bc)
    
    # Final conversion
    result = []
    for L, R in best_bicliques:
        L = [int(v) for v in L]
        R = [int(v) for v in R]
        result.append([L, R])
    
    return result
