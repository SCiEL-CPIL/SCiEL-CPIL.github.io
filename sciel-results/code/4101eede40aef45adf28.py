"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _build_edge_set():
    es = set()
    for u, v in P['edges']:
        es.add((min(u, v), max(u, v)))
    return es


def _get_adjacency(edge_set, n):
    adj = [set() for _ in range(n)]
    for u, v in edge_set:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _is_biclique(L, R, adj):
    """Check if all cross pairs between L and R are edges using adjacency."""
    for u in L:
        for v in R:
            if v not in adj[u]:
                return False
    return True


def _extract_biclique_from_edges(edge_subset, adj, n):
    """
    Given a set of edges, try to extract a maximal biclique from them.
    Returns (L, R) or None.
    """
    if not edge_subset:
        return None
    
    # Collect vertices
    verts = set()
    for u, v in edge_subset:
        verts.add(u)
        verts.add(v)
    
    # Start with one edge
    u0, v0 = next(iter(edge_subset))
    L = {u0}
    R = {v0}
    
    # Expand
    changed = True
    while changed:
        changed = False
        # Try adding to L
        for x in verts:
            if x in L or x in R:
                continue
            if all(x in adj[r] for r in R):
                # Check that x has edges to all in R within our edge set context
                L.add(x)
                changed = True
        # Try adding to R
        for x in verts:
            if x in L or x in R:
                continue
            if all(x in adj[l] for l in L):
                R.add(x)
                changed = True
    
    # Verify all cross edges exist
    if not _is_biclique(list(L), list(R), adj):
        return None
    
    # Only include edges that are actually in edge_subset
    valid_edges = set()
    for u in L:
        for v in R:
            e = (min(u, v), max(u, v))
            if e in edge_subset:
                valid_edges.add(e)
    
    if not valid_edges:
        return None
    
    # Re-verify it's a proper biclique with only these edges
    # The biclique [L,R] covers all cross edges, but we only claim the ones in edge_subset
    # For the partition, we need the biclique to cover exactly its assigned edges
    # So we need L×R ⊆ edge_subset (all cross pairs must be in the subset)
    for u in L:
        for v in R:
            e = (min(u, v), max(u, v))
            if e not in edge_subset:
                return None
    
    return (list(L), list(R))


def _greedy_star_partition(edge_set, adj, n, deadline):
    """
    Simple star-based partition: for each vertex, group its incident edges
    into bicliques where the other endpoints form independent sets (no edges between them).
    """
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Pick vertex with most remaining edges
        deg = defaultdict(int)
        for u, v in remaining:
            deg[u] += 1
            deg[v] += 1
        
        if not deg:
            break
        
        center = max(deg, key=lambda x: deg[x])
        
        # Get all neighbors of center in remaining
        neighbors = set()
        for u, v in remaining:
            if u == center:
                neighbors.add(v)
            elif v == center:
                neighbors.add(u)
        
        if not neighbors:
            # Isolated edge
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
            continue
        
        # Group neighbors into bicliques: [center + group1, group2] etc.
        # Actually simpler: for each neighbor, check if we can form a biclique
        # with center and some subset of other neighbors
        
        # Try to build a biclique with center on left
        L = {center}
        R = set()
        
        # Greedily add neighbors to R
        for nb in sorted(neighbors, key=lambda x: -len(adj[x])):
            # Check if nb connects to all in L (just center, so always yes if edge exists)
            # and if we want to keep R as an independent set... 
            # Actually for a biclique [L, R], we need all cross edges.
            # If L = {center}, R can be any subset of neighbors.
            # But we want to maximize |R|.
            R.add(nb)
        
        # Now try to expand L: add other vertices that connect to all in R
        changed = True
        while changed:
            changed = False
            for x in range(n):
                if x in L or x in R:
                    continue
                if all(x in adj[r] for r in R):
                    # Check that edges (x, r) are in remaining
                    all_in_remaining = True
                    for r in R:
                        e = (min(x, r), max(x, r))
                        if e not in remaining:
                            all_in_remaining = False
                            break
                    if all_in_remaining:
                        L.add(x)
                        changed = True
        
        # Verify
        if L and R and _is_biclique(list(L), list(R), adj):
            # Check all cross edges are in remaining
            all_in_remaining = True
            for u in L:
                for v in R:
                    e = (min(u, v), max(u, v))
                    if e not in remaining:
                        all_in_remaining = False
                        break
                if not all_in_remaining:
                    break
            
            if all_in_remaining:
                bicliques.append((list(L), list(R)))
                for u in L:
                    for v in R:
                        remaining.discard((min(u, v), max(u, v)))
            else:
                # Fallback: just use center star
                bicliques.append(([center], list(neighbors)))
                for nb in neighbors:
                    remaining.discard((min(center, nb), max(center, nb)))
        else:
            bicliques.append(([center], list(neighbors)))
            for nb in neighbors:
                remaining.discard((min(center, nb), max(center, nb)))
    
    return bicliques


def _greedy_max_biclique_partition(edge_set, adj, n, deadline):
    """
    Greedy: repeatedly find largest valid biclique in remaining edges, remove it.
    """
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        
        best = None
        best_size = 0
        
        # Try all edges as seeds
        for u, v in remaining:
            if time.time() > deadline:
                break
            
            L = {u}
            R = {v}
            
            # Expand
            changed = True
            while changed:
                changed = False
                for x in range(n):
                    if x in L or x in R:
                        continue
                    if all(x in adj[r] for r in R):
                        # Check edges in remaining
                        ok = True
                        for r in R:
                            if (min(x, r), max(x, r)) not in remaining:
                                ok = False
                                break
                        if ok:
                            L.add(x)
                            changed = True
                for x in range(n):
                    if x in L or x in R:
                        continue
                    if all(x in adj[l] for l in L):
                        ok = True
                        for l in L:
                            if (min(x, l), max(x, l)) not in remaining:
                                ok = False
                                break
                        if ok:
                            R.add(x)
                            changed = True
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best = (L, R)
        
        if best is None or best_size == 0:
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
        else:
            L, R = best
            bicliques.append((list(L), list(R)))
            for u in L:
                for v in R:
                    remaining.discard((min(u, v), max(u, v)))
    
    return bicliques


def _local_search_partition(bicliques, edge_set, adj, n, deadline):
    """
    Local search: try to merge two bicliques if their union forms a valid biclique.
    Also try moving edges between bicliques to enable merges.
    """
    # Represent partition as list of (L, R) pairs
    partition = [(list(L), list(R)) for L, R in bicliques]
    
    def partition_edges(part):
        """Get the set of edges covered by this partition."""
        edges = set()
        for L, R in part:
            for u in L:
                for v in R:
                    edges.add((min(u, v), max(u, v)))
        return edges
    
    def is_valid_partition(part):
        """Check partition is valid: covers all edges exactly once, each biclique valid."""
        edges = partition_edges(part)
        if edges != edge_set:
            return False
        for L, R in part:
            if not L or not R:
                return False
            if not _is_biclique(L, R, adj):
                return False
            # Check no overlap: this is handled by edge set equality
        return True
    
    current = list(partition)
    current_edges = partition_edges(current)
    
    if current_edges != edge_set:
        # Invalid initial partition, rebuild
        return bicliques
    
    best = list(current)
    best_count = len(current)
    
    # Try merging pairs
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current) and time.time() < deadline:
            j = i + 1
            while j < len(current) and time.time() < deadline:
                L1, R1 = current[i]
                L2, R2 = current[j]
                
                # Try merging: new biclique should cover all edges from both
                # The merged biclique must have all cross edges present
                # Option 1: L = L1 ∪ L2, R = R1 ∪ R2 (if valid)
                # Option 2: L = L1 ∪ R2, R = R1 ∪ L2 (if valid)
                # Option 3: L = L1 ∪ L2, R = R1 (if R1 ⊆ R2 context)
                # ... many options
                
                # Simplest: try L1∪L2, R1∪R2
                new_L = set(L1) | set(L2)
                new_R = set(R1) | set(R2)
                if new_L and new_R and _is_biclique(list(new_L), list(new_R), adj):
                    # Check all cross edges are in edge_set
                    ok = True
                    for u in new_L:
                        for v in new_R:
                            if (min(u, v), max(u, v)) not in edge_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_partition = current[:i] + current[i+1:j] + current[j+1:]
                        new_partition.insert(i, (list(new_L), list(new_R)))
                        if len(new_partition) < best_count:
                            current = new_partition
                            best = list(current)
                            best_count = len(current)
                            improved = True
                            break
                
                # Try L1∪R2, R1∪L2
                new_L2 = set(L1) | set(R2)
                new_R2 = set(R1) | set(L2)
                if new_L2 and new_R2 and _is_biclique(list(new_L2), list(new_R2), adj):
                    ok = True
                    for u in new_L2:
                        for v in new_R2:
                            if (min(u, v), max(u, v)) not in edge_set:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        new_partition = current[:i] + current[i+1:j] + current[j+1:]
                        new_partition.insert(i, (list(new_L2), list(new_R2)))
                        if len(new_partition) < best_count:
                            current = new_partition
                            best = list(current)
                            best_count = len(current)
                            improved = True
                            break
                
                j += 1
            i += 1
    
    return best


def _edge_based_biclique_extraction(edge_set, adj, n, deadline):
    """
    Extract bicliques based on edge co-occurrence patterns.
    Edges that form K_{2,2} patterns suggest biclique structure.
    """
    remaining = set(edge_set)
    bicliques = []
    
    while remaining and time.time() < deadline:
        # Find a K_{2,2} pattern or just pick an edge
        found_pattern = False
        for u, v in list(remaining):
            if time.time() > deadline:
                break
            # Find common neighbors
            common = adj[u] & adj[v]
            if common:
                # Try to build a biclique around this edge
                L = {u}
                R = {v}
                
                # Add common neighbors to appropriate side
                for w in common:
                    # w connects to both u and v
                    # Could add w to L (if w connects to all in R) or R
                    if all(w in adj[r] for r in R):
                        L.add(w)
                    elif all(w in adj[l] for l in L):
                        R.add(w)
                
                # Expand further
                changed = True
                while changed:
                    changed = False
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[r] for r in R):
                            ok = True
                            for r in R:
                                if (min(x, r), max(x, r)) not in remaining:
                                    ok = False
                                    break
                            if ok:
                                L.add(x)
                                changed = True
                    for x in range(n):
                        if x in L or x in R:
                            continue
                        if all(x in adj[l] for l in L):
                            ok = True
                            for l in L:
                                if (min(x, l), max(x, l)) not in remaining:
                                    ok = False
                                    break
                            if ok:
                                R.add(x)
                                changed = True
                
                if L and R and _is_biclique(list(L), list(R), adj):
                    ok = True
                    for u2 in L:
                        for v2 in R:
                            if (min(u2, v2), max(u2, v2)) not in remaining:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        bicliques.append((list(L), list(R)))
                        for u2 in L:
                            for v2 in R:
                                remaining.discard((min(u2, v2), max(u2, v2)))
                        found_pattern = True
                        break
        
        if not found_pattern:
            # Fallback: single edge
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
    
    return bicliques


def _exact_cover_small(remaining_edges, adj, n, deadline):
    """
    For small remaining edge sets, use exact backtracking to find minimum biclique cover.
    """
    if not remaining_edges:
        return []
    
    if len(remaining_edges) > 12:
        # Too large for exact search, use greedy
        return _greedy_max_biclique_partition(remaining_edges, adj, n, deadline)
    
    edge_list = sorted(remaining_edges)
    
    # Generate all valid bicliques within these edges
    candidates = []
    for mask in range(1, 1 << len(edge_list)):
        if time.time() > deadline:
            break
        edges_in_mask = set()
        for i in range(len(edge_list)):
            if mask & (1 << i):
                edges_in_mask.add(edge_list[i])
        
        # Try to extract a biclique from these edges
        bc = _extract_biclique_from_edges(edges_in_mask, adj, n)
        if bc:
            L, R = bc
            # Verify all cross edges are in edges_in_mask
            ok = True
            for u in L:
                for v in R:
                    if (min(u, v), max(u, v)) not in edges_in_mask:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                candidates.append((L, R, edges_in_mask))
    
    # Remove duplicates
    unique = []
    seen = set()
    for L, R, edges in candidates:
        key = frozenset(edges)
        if key not in seen:
            seen.add(key)
            unique.append((L, R, edges))
    
    if not unique:
        # Each edge as separate biclique
        return [[(u, v)] for u, v in edge_list]
    
    # Backtracking
    best_solution = None
    best_count = len(edge_list)  # Upper bound: one biclique per edge
    
    def backtrack(remaining_mask, current, depth):
        nonlocal best_solution, best_count
        
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if depth < best_count:
                best_count = depth
                best_solution = list(current)
            return
        
        if depth >= best_count:
            return
        
        # Lower bound
        if depth + 1 > best_count:
            return
        
        # Pick first uncovered edge
        first_edge_idx = None
        for i in range(len(edge_list)):
            if remaining_mask & (1 << i):
                first_edge_idx = i
                break
        
        if first_edge_idx is None:
            return
        
        first_edge = edge_list[first_edge_idx]
        
        # Try each candidate that covers this edge
        for L, R, edges in unique:
            if first_edge in edges:
                new_mask = remaining_mask & ~_mask_from_edges(edges, edge_list)
                current.append((L, R))
                backtrack(new_mask, current, depth + 1)
                current.pop()
    
    def _mask_from_edges(edges, edge_list):
        mask = 0
        for e in edges:
            for i, el in enumerate(edge_list):
                if el == e:
                    mask |= (1 << i)
                    break
        return mask
    
    all_mask = (1 << len(edge_list)) - 1
    backtrack(all_mask, [], 0)
    
    if best_solution:
        return best_solution
    else:
        # Fallback
        return [[(u, v)] for u, v in edge_list]


def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    edge_set = _build_edge_set()
    n = P['n']
    adj = _get_adjacency(edge_set, n)
    
    # Phase 1: Multiple greedy initializations
    t1 = time.time()
    greedy1 = _greedy_star_partition(edge_set, adj, n, deadline)
    greedy2 = _greedy_max_biclique_partition(edge_set, adj, n, deadline)
    greedy3 = _edge_based_biclique_extraction(edge_set, adj, n, deadline)
    
    # Phase 2: Local search on each
    t2 = time.time()
    best = None
    best_count = float('inf')
    
    for g in [greedy1, greedy2, greedy3]:
        if time.time() > deadline:
            break
        improved = _local_search_partition(g, edge_set, adj, n, deadline)
        if len(improved) < best_count:
            best_count = len(improved)
            best = improved
    
    # Phase 3: If still large, try exact on small residuals
    t3 = time.time()
    if best is None:
        best = greedy1
    
    # Verify and finalize
    result = []
    for L, R in best:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    # Final verification
    covered = set()
    valid = True
    for L, R in result:
        for u in L:
            for v in R:
                e = (min(u, v), max(u, v))
                if e not in edge_set:
                    valid = False
                    break
                if e in covered:
                    valid = False
                    break
                covered.add(e)
        if not valid:
            break
    
    if not valid or covered != edge_set:
        # Fallback to star partition
        result = []
        for L, R in greedy1:
            L_list = [int(v) for v in L]
            R_list = [int(v) for v in R]
            if L_list and R_list:
                result.append([L_list, R_list])
    
    return result
# EVOLVE-BLOCK-END
