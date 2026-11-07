import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def _edge_key(u, v):
    if u > v:
        u, v = v, u
    return (u, v)


def _build_adjacency(edges, n):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _expand_biclique(adj, n, seed_u, seed_v, remaining_edges):
    """
    Expand a biclique starting from edge (seed_u, seed_v) in the remaining_edges subgraph.
    Returns (L, R) as frozensets.
    """
    L = {seed_u}
    R = {seed_v}
    changed = True
    while changed:
        changed = False
        # Try to add to L
        new_L = set(L)
        for w in range(n):
            if w not in L and w not in R:
                # w must be connected to all in R in remaining graph
                all_connected = True
                for r in R:
                    if _edge_key(w, r) not in remaining_edges:
                        all_connected = False
                        break
                if all_connected:
                    new_L.add(w)
                    changed = True
        L = new_L
        
        # Try to add to R
        new_R = set(R)
        for w in range(n):
            if w not in R and w not in L:
                # w must be connected to all in L in remaining graph
                all_connected = True
                for l in L:
                    if _edge_key(w, l) not in remaining_edges:
                        all_connected = False
                        break
                if all_connected:
                    new_R.add(w)
                    changed = True
        R = new_R
        
    return frozenset(L), frozenset(R)


def _find_maximal_bicliques(edges, n, deadline, max_bicliques=200):
    """
    Find a set of maximal (or near-maximal) bicliques in the graph.
    Uses multiple starting edges to find diverse bicliques.
    """
    adj = _build_adjacency(edges, n)
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    bicliques = []
    seen = set()
    
    # Compute degrees
    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    
    # Sort edges by degree sum for priority
    sorted_edges = sorted(edge_set, key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
    
    # Try to find bicliques from high-degree edges first
    for edge in sorted_edges:
        if time.time() > deadline:
            break
        if len(bicliques) >= max_bicliques:
            break
        
        u, v = edge
        L, R = _expand_biclique(adj, n, u, v, edge_set)
        
        # Check if this biclique is new
        biclique_key = (L, R)
        if biclique_key not in seen:
            seen.add(biclique_key)
            bicliques.append((L, R))
    
    # Also try some random edges for diversity
    random.shuffle(sorted_edges)
    for edge in sorted_edges[:50]:
        if time.time() > deadline:
            break
        if len(bicliques) >= max_bicliques:
            break
        
        u, v = edge
        L, R = _expand_biclique(adj, n, u, v, edge_set)
        
        biclique_key = (L, R)
        if biclique_key not in seen:
            seen.add(biclique_key)
            bicliques.append((L, R))
    
    return bicliques


def _greedy_set_cover(bicliques, edge_set, n, deadline, seed=None):
    """
    Greedy set cover: select bicliques that cover the most uncovered edges.
    Uses a ratio heuristic to prioritize dense, non-overlapping coverage.
    """
    if seed is not None:
        random.seed(seed)
    
    remaining = set(edge_set)
    selected = []
    
    # Precompute edge sets for each biclique
    biclique_edges = []
    for L, R in bicliques:
        edges_in_biclique = set()
        for l in L:
            for r in R:
                ek = _edge_key(l, r)
                if ek in edge_set:
                    edges_in_biclique.add(ek)
        biclique_edges.append(edges_in_biclique)
    
    while remaining:
        if time.time() > deadline:
            break
        
        best_idx = -1
        best_score = -1
        best_cover = 0
        
        for i, (L, R) in enumerate(bicliques):
            if time.time() > deadline:
                break
            
            edges_i = biclique_edges[i]
            uncovered = edges_i & remaining
            if not uncovered:
                continue
            
            cover_count = len(uncovered)
            
            # Ratio heuristic: uncovered edges / (1 + number of already selected bicliques that overlap)
            overlap_count = 0
            for j, sel_L, sel_R in selected:
                sel_edges = biclique_edges[j]
                if sel_edges & edges_i:
                    overlap_count += 1
            
            score = cover_count / (1 + overlap_count * 0.5)
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_cover = cover_count
        
        if best_idx == -1 or best_cover == 0:
            # Fallback: pick a random edge as a single-edge biclique
            edge = next(iter(remaining))
            u, v = edge
            selected.append((frozenset({u}), frozenset({v})))
            remaining.discard(edge)
        else:
            L, R = bicliques[best_idx]
            selected.append((L, R))
            edges_i = biclique_edges[best_idx]
            remaining -= edges_i
    
    return selected


def _local_search(bicliques, edge_set, n, deadline):
    """
    Try to improve the solution by removing or replacing bicliques.
    """
    if not bicliques:
        return bicliques
    
    adj = _build_adjacency(edge_set, n)
    current = list(bicliques)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        
        # Try removing each biclique and see if the rest still covers all edges
        for i in range(len(current)):
            if time.time() > deadline:
                break
            
            test = current[:i] + current[i+1:]
            
            # Check if test covers all edges
            covered = set()
            for L, R in test:
                for l in L:
                    for r in R:
                        ek = _edge_key(l, r)
                        if ek in edge_set:
                            covered.add(ek)
            
            if covered == edge_set:
                current = test
                improved = True
                break
        
        # Try merging two bicliques if their union is still a valid biclique
        for i in range(len(current)):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(current)):
                if time.time() > deadline:
                    break
                
                L1, R1 = current[i]
                L2, R2 = current[j]
                
                # Try merging: L = L1 ∪ L2, R = R1 ∪ R2
                # This is valid if all cross edges exist
                # But we need to check that the merged biclique only covers edges in edge_set
                # and that removing i and j and adding the merged one still covers everything
                
                # Actually, a simpler merge: check if we can absorb one biclique into another
                # by extending its sides
                pass  # Complex, skip for now
    
    return current


def _try_rebuild_with_fewer(bicliques, edge_set, n, deadline):
    """
    Try to rebuild the partition with fewer bicliques using a different strategy.
    """
    if not bicliques:
        return bicliques
    
    # Strategy: use the largest bicliques first
    biclique_sizes = []
    for i, (L, R) in enumerate(bicliques):
        size = len(L) * len(R)
        biclique_sizes.append((size, i))
    
    biclique_sizes.sort(reverse=True)
    
    remaining = set(edge_set)
    new_solution = []
    
    for size, idx in biclique_sizes:
        if time.time() > deadline:
            break
        if not remaining:
            break
        
        L, R = bicliques[idx]
        
        # Check which edges of this biclique are still uncovered
        edges_in = set()
        for l in L:
            for r in R:
                ek = _edge_key(l, r)
                if ek in edge_set and ek in remaining:
                    edges_in.add(ek)
        
        if not edges_in:
            continue
        
        new_solution.append((L, R))
        remaining -= edges_in
    
    # Handle leftovers
    for edge in remaining:
        u, v = edge
        new_solution.append((frozenset({u}), frozenset({v})))
    
    return new_solution


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    n = P['n']
    edges = P['edges']
    
    edge_set = set()
    for u, v in edges:
        edge_set.add(_edge_key(u, v))
    
    best_solution = None
    best_count = float('inf')
    
    # Phase 1: Find maximal bicliques
    bicliques = _find_maximal_bicliques(edges, n, deadline, max_bicliques=100)
    
    # Phase 2: Multiple greedy set cover attempts with different seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        
        solution = _greedy_set_cover(bicliques, edge_set, n, deadline, seed=seed)
        count = len(solution)
        
        if count < best_count:
            best_count = count
            best_solution = solution
        
        if count <= 8:
            break
    
    # Phase 3: Local search on the best solution
    if best_solution and time.time() < deadline:
        improved = _local_search(best_solution, edge_set, n, deadline)
        if len(improved) < best_count:
            best_count = len(improved)
            best_solution = improved
    
    # Phase 4: Try rebuilding with a different ordering
    if best_solution and time.time() < deadline:
        rebuilt = _try_rebuild_with_fewer(best_solution, edge_set, n, deadline)
        if len(rebuilt) < best_count:
            best_count = len(rebuilt)
            best_solution = rebuilt
    
    # Fallback
    if best_solution is None:
        best_solution = [(frozenset({u}), frozenset({v})) for u, v in edge_set]
    
    # Convert to required format
    result = []
    for L, R in best_solution:
        result.append([list(L), list(R)])
    
    return result
