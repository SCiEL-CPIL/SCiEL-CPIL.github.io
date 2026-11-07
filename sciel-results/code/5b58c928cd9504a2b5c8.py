import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin

    n = P['n']
    edges = P['edges']
    
    # Build adjacency mask for fast edge existence check
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    # Convert edges to a set of normalized tuples for tracking remaining edges
    def norm_edge(e):
        return (min(e[0], e[1]), max(e[0], e[1]))
    
    initial_edges = set(norm_edge(e) for e in edges)
    
    best_bicliques = None
    best_count = float('inf')

    def is_biclique(L, R):
        """Check if (L, R) is a biclique in the original graph."""
        for u in L:
            mask_u = adj_mask[u]
            for v in R:
                if not (mask_u & (1 << v)):
                    return False
        return True

    def try_greedy(restarts=50):
        nonlocal best_bicliques, best_count
        
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            remaining_edges = set(initial_edges)
            current_bicliques = []
            
            while remaining_edges:
                if time.time() > deadline:
                    return
                
                # Pick a random edge to start a biclique
                # Using random selection helps with diversity
                edge = random.choice(list(remaining_edges))
                u, v = edge
                
                # Expand biclique greedily
                L = {u}
                R = {v}
                
                changed = True
                while changed:
                    changed = False
                    
                    # Try to add to L: nodes connected to all in R
                    # Candidate L nodes are those in remaining edges' endpoints
                    # that are connected to all nodes in R
                    # Optimization: Only consider nodes that appear in remaining edges
                    # But for simplicity and correctness, we can check all nodes 0..n-1
                    # However, n is small (30), so this is fine.
                    
                    new_L_candidates = []
                    for x in range(n):
                        if x in L:
                            continue
                        # Check if x is connected to all in R in the ORIGINAL graph
                        # AND if there are remaining edges connecting x to some node in R
                        # Actually, for the biclique to cover remaining edges, 
                        # we only care about edges that exist in remaining_edges.
                        # But the biclique property must hold for the original graph.
                        # If we add x to L, we must ensure all (x, y) for y in R are in the graph.
                        # Then, when we remove edges, we remove (x, y) if they are in remaining_edges.
                        
                        # Check graph connectivity
                        if all((adj_mask[x] & (1 << y)) for y in R):
                            # Check if x is incident to any remaining edge with a node in R
                            # If not, adding x doesn't help cover edges, but doesn't hurt validity.
                            # However, it might prevent merging later? No, larger L/R is better for merging.
                            # But if x has no remaining edges to R, it's "empty" in this context.
                            # Let's only add if it helps cover at least one remaining edge.
                            has_rem_edge = False
                            for y in R:
                                if (min(x, y), max(x, y)) in remaining_edges:
                                    has_rem_edge = True
                                    break
                            if has_rem_edge:
                                new_L_candidates.append(x)
                    
                    if new_L_candidates:
                        # Pick the one that connects to the most remaining edges in R?
                        # Or just pick one. Let's pick the one with max degree in remaining edges to R.
                        best_x = None
                        max_deg = 0
                        for x in new_L_candidates:
                            deg = sum(1 for y in R if (min(x, y), max(x, y)) in remaining_edges)
                            if deg > max_deg:
                                max_deg = deg
                                best_x = x
                        if best_x is not None:
                            L.add(best_x)
                            changed = True
                            
                    # Try to add to R: nodes connected to all in L
                    new_R_candidates = []
                    for y in range(n):
                        if y in R:
                            continue
                        if all((adj_mask[y] & (1 << x)) for x in L):
                            has_rem_edge = False
                            for x in L:
                                if (min(x, y), max(x, y)) in remaining_edges:
                                    has_rem_edge = True
                                    break
                            if has_rem_edge:
                                new_R_candidates.append(y)
                    
                    if new_R_candidates:
                        best_y = None
                        max_deg = 0
                        for y in new_R_candidates:
                            deg = sum(1 for x in L if (min(x, y), max(x, y)) in remaining_edges)
                            if deg > max_deg:
                                max_deg = deg
                                best_y = y
                        if best_y is not None:
                            R.add(best_y)
                            changed = True

                # We have a biclique (L, R).
                # Remove all edges between L and R from remaining_edges
                edges_to_remove = set()
                for u in L:
                    for v in R:
                        e = norm_edge((u, v))
                        if e in remaining_edges:
                            edges_to_remove.add(e)
                
                # If no edges were removed, we might be stuck or L/R are empty (shouldn't happen)
                if not edges_to_remove:
                    # Fallback: just take the starting edge
                    current_bicliques.append([[u], [v]])
                    remaining_edges.remove((u, v))
                    continue

                remaining_edges -= edges_to_remove
                current_bicliques.append([list(L), list(R)])

            # Local Search: Merge bicliques
            merged = True
            while merged:
                merged = False
                i = 0
                while i < len(current_bicliques):
                    if time.time() > deadline:
                        break
                    j = i + 1
                    while j < len(current_bicliques):
                        if time.time() > deadline:
                            break
                        L1, R1 = current_bicliques[i]
                        L2, R2 = current_bicliques[j]
                        
                        # Try merging L1 and L2 into L12, keeping R1 and R2 separate?
                        # No, a biclique is defined by (L, R).
                        # If we merge two bicliques, we must form a new valid biclique.
                        # Options:
                        # 1. (L1 U L2, R1) - Valid if all L2 nodes connect to all R1 nodes.
                        # 2. (L1, R1 U R2) - Valid if all L1 nodes connect to all R2 nodes.
                        # 3. (L1 U L2, R1 U R2) - Valid if all cross edges exist.
                        
                        # Check Option 3 first (strongest)
                        new_L = set(L1) | set(L2)
                        new_R = set(R1) | set(R2)
                        if is_biclique(new_L, new_R):
                            current_bicliques[i] = [list(new_L), list(new_R)]
                            del current_bicliques[j]
                            merged = True
                            break
                            
                        # Check Option 1
                        new_L1 = set(L1) | set(L2)
                        if is_biclique(new_L1, R1):
                            current_bicliques[i] = [list(new_L1), list(R1)]
                            # We discarded R2? No, this logic is flawed.
                            # If we merge into (L1 U L2, R1), what happens to edges covered by (L2, R2)?
                            # The biclique (L1 U L2, R1) covers edges between L1 U L2 and R1.
                            # The old bicliques covered edges between L1-R1 and L2-R2.
                            # If we replace them with (L1 U L2, R1), we cover L1-R1 and L2-R1.
                            # We LOSE coverage of L2-R2 unless L2-R2 is a subset of L2-R1 (i.e. R2 subset R1).
                            # So simple union of sides is not enough. We need the new biclique to cover ALL edges
                            # that the old two covered.
                            # Edges covered by B1: E1 = { (u,v) | u in L1, v in R1 }
                            # Edges covered by B2: E2 = { (u,v) | u in L2, v in R2 }
                            # New biclique B_new must cover E1 U E2.
                            # If we choose (L1 U L2, R1), it covers E1 and (L2, R1).
                            # It does NOT cover (L2, R2) unless R2 is subset of R1.
                            # So this merge is only valid if R2 is subset of R1 AND is_biclique(L1 U L2, R1).
                            # Similarly for other options.
                            
                            # Let's stick to the most common valid merge:
                            # If L1 == L2, we can merge R1 and R2 if is_biclique(L1, R1 U R2).
                            # If R1 == R2, we can merge L1 and L2 if is_biclique(L1 U L2, R1).
                            # If neither, we can try (L1 U L2, R1 U R2) if is_biclique.
                            
                            # Let's re-evaluate.
                            pass # Skip complex partial merges for now, stick to full union or subset checks.
                        
                        # Check if L1 == L2
                        if set(L1) == set(L2):
                            new_R = set(R1) | set(R2)
                            if is_biclique(L1, new_R):
                                current_bicliques[i] = [list(L1), list(new_R)]
                                del current_bicliques[j]
                                merged = True
                                break
                                
                        # Check if R1 == R2
                        if set(R1) == set(R2):
                            new_L = set(L1) | set(L2)
                            if is_biclique(new_L, R1):
                                current_bicliques[i] = [list(new_L), list(R1)]
                                del current_bicliques[j]
                                merged = True
                                break

                        j += 1
                    i += 1

            if len(current_bicliques) < best_count:
                best_count = len(current_bicliques)
                best_bicliques = [ [list(b[0]), list(b[1])] for b in current_bicliques ]
                if best_count == 1:
                    break

    try_greedy()

    if best_bicliques is None:
        # Fallback: trivial solution (each edge is a biclique)
        best_bicliques = []
        for e in edges:
            best_bicliques.append([[e[0]], [e[1]]])

    result = []
    for b in best_bicliques:
        left = [int(x) for x in b[0]]
        right = [int(x) for x in b[1]]
        result.append([left, right])
        
    return result
