import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    edges = P['edges']
    n = P['n']
    
    # Build adjacency matrix
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    all_edges = list(edge_set)
    m = len(all_edges)
    
    # Precompute for each vertex, the set of neighbors
    neighbors = [set() for _ in range(n)]
    for u, v in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    
    # Enumerate maximal bicliques
    # A biclique is defined by (L, R) where all pairs (l, r) with l in L, r in R are edges
    # Two bicliques are incompatible if they share an edge
    # We want to find all maximal bicliques
    
    # For efficiency, use a different approach:
    # For each subset of edges that could form a biclique, check if it's maximal
    # But this is exponential. Instead, use the fact that n=30 is small.
    
    # Alternative: use the structure. A biclique (L,R) is maximal if:
    # - No vertex can be added to L while maintaining all edges to R
    # - No vertex can be added to R while maintaining all edges to L
    
    # Let's enumerate by trying all possible L subsets (or R subsets)
    # Since n=30, we can't try all 2^30 subsets.
    
    # Better approach: use the edge-based enumeration
    # For each edge, try to extend it to a maximal biclique
    
    maximal_bicliques = []
    seen_bicliques = set()
    
    def extend_biclique(L, R):
        """Extend (L, R) to a maximal biclique. Returns the maximal biclique."""
        L = set(L)
        R = set(R)
        changed = True
        while changed:
            changed = False
            # Try to add to L: vertex w not in L such that w is connected to all in R
            for w in range(n):
                if w in L:
                    continue
                if all(adj[w][r] for r in R):
                    L.add(w)
                    changed = True
            # Try to add to R: vertex w not in R such that w is connected to all in L
            for w in range(n):
                if w in R:
                    continue
                if all(adj[l][w] for l in L):
                    R.add(w)
                    changed = True
        return (frozenset(L), frozenset(R))
    
    # Enumerate maximal bicliques by starting from each edge
    for u, v in all_edges:
        # Try (L={u}, R={v})
        L, R = extend_biclique({u}, {v})
        key = (L, R)
        if key not in seen_bicliques:
            seen_bicliques.add(key)
            maximal_bicliques.append((L, R))
        
        # Also try (L={v}, R={u}) - might give different maximal biclique
        L2, R2 = extend_biclique({v}, {u})
        key2 = (L2, R2)
        if key2 not in seen_bicliques:
            seen_bicliques.add(key2)
            maximal_bicliques.append((L2, R2))
    
    # Also try single vertices as potential sides
    for u in range(n):
        for v in range(n):
            if u != v and adj[u][v]:
                L, R = extend_biclique({u}, {v})
                key = (L, R)
                if key not in seen_bicliques:
                    seen_bicliques.add(key)
                    maximal_bicliques.append((L, R))
    
    # Filter: keep only bicliques that actually cover at least one edge
    filtered_bicliques = []
    for L, R in maximal_bicliques:
        # Check if this biclique covers at least one edge from edge_set
        has_edge = False
        for l in L:
            for r in R:
                if (l, r) in edge_set or (r, l) in edge_set:
                    has_edge = True
                    break
            if has_edge:
                break
        if has_edge:
            filtered_bicliques.append((L, R))
    
    # For each biclique, compute which edges it covers
    biclique_edges = []
    for L, R in filtered_bicliques:
        covered = set()
        for l in L:
            for r in R:
                if l < r:
                    if (l, r) in edge_set:
                        covered.add((l, r))
                else:
                    if (r, l) in edge_set:
                        covered.add((r, l))
        if covered:
            biclique_edges.append(covered)
    
    # Now solve exact cover: select minimum number of bicliques to cover all edges
    # Use backtracking with memoization
    
    # Map edges to indices
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    all_edge_indices = set(range(m))
    
    # For each biclique, which edge indices does it cover?
    biclique_edge_indices = []
    for covered in biclique_edges:
        idx_set = frozenset(edge_to_idx[e] for e in covered)
        biclique_edge_indices.append(idx_set)
    
    # For each edge, which bicliques cover it?
    edge_to_bicliques = [set() for _ in range(m)]
    for bi, idx_set in enumerate(biclique_edge_indices):
        for ei in idx_set:
            edge_to_bicliques[ei].add(bi)
    
    # Remove dominated bicliques: if biclique A's edges are subset of biclique B's edges, A is dominated
    # (since B covers everything A covers and more, using B is at least as good)
    # Actually, we want to keep minimal bicliques in terms of coverage... 
    # No, we want to remove bicliques whose edge set is a subset of another's
    # because if A ⊆ B, then any solution using A can use B instead (or both)
    # But using B might block other bicliques. So we can't simply remove.
    # Instead, let's just proceed with all bicliques.
    
    # Sort bicliques by size (larger first) for better pruning
    biclique_order = sorted(range(len(biclique_edge_indices)), 
                           key=lambda i: -len(biclique_edge_indices[i]))
    
    # Memoization: frozenset of uncovered edges -> min bicliques needed
    memo = {}
    
    def solve(remaining_edges, current_count, best_count):
        if time.time() > deadline:
            return best_count
        
        if not remaining_edges:
            return current_count
        
        if current_count >= best_count:
            return best_count
        
        key = frozenset(remaining_edges)
        if key in memo:
            if memo[key] <= current_count:
                return best_count
            # memo[key] gives the min additional bicliques from this state
            result = current_count + memo[key]
            return min(best_count, result)
        
        # Find the edge with fewest candidate bicliques
        min_candidates = None
        min_count = float('inf')
        for e in remaining_edges:
            candidates = edge_to_bicliques[e] & set(range(len(biclique_edge_indices)))
            # Only consider bicliques that can cover e and don't conflict
            count = len(candidates)
            if count < min_count:
                min_count = count
                min_candidates = e
                if count == 1:
                    break
        
        if min_candidates is None:
            return best_count
        
        candidates = edge_to_bicliques[min_candidates]
        
        # Try each candidate biclique
        best = best_count
        for bi in sorted(candidates, key=lambda x: -len(biclique_edge_indices[x])):
            covered = biclique_edge_indices[bi]
            new_remaining = remaining_edges - covered
            result = solve(new_remaining, current_count + 1, best)
            best = result
            if best == current_count + 1:
                break
        
        memo[key] = best - current_count
        return best
    
    # Initial call
    best_result = solve(all_edge_indices, 0, len(biclique_edge_indices) + 1)
    
    # If we found a solution, reconstruct it
    # Actually, the above just gives the count. We need to reconstruct the actual bicliques.
    # Let's redo with reconstruction.
    
    # Clear memo and redo with path tracking
    memo2 = {}
    
    def solve_with_path(remaining_edges, current_path, current_count, best_count):
        if time.time() > deadline:
            return None, best_count
        
        if not remaining_edges:
            return list(current_path), current_count
        
        if current_count >= best_count:
            return None, best_count
        
        key = frozenset(remaining_edges)
        if key in memo2:
            if memo2[key] <= current_count:
                return None, best_count
        
        # Find the edge with fewest candidate bicliques
        min_candidates = None
        min_count = float('inf')
        for e in remaining_edges:
            candidates = edge_to_bicliques[e]
            count = len(candidates)
            if count < min_count:
                min_count = count
                min_candidates = e
                if count == 1:
                    break
        
        if min_candidates is None:
            return None, best_count
        
        candidates = edge_to_bicliques[min_candidates]
        
        best_path = None
        best = best_count
        for bi in sorted(candidates, key=lambda x: -len(biclique_edge_indices[x])):
            covered = biclique_edge_indices[bi]
            new_remaining = remaining_edges - covered
            path, result = solve_with_path(new_remaining, current_path + [bi], current_count + 1, best)
            if path is not None and result < best:
                best = result
                best_path = path
                if best == current_count + 1:
                    break
        
        if best_path is None:
            memo2[key] = float('inf')
        else:
            memo2[key] = best - current_count
        
        return best_path, best
    
    path, best_count = solve_with_path(all_edge_indices, [], 0, len(biclique_edge_indices) + 1)
    
    if path is None:
        # Fallback: greedy
        remaining = set(all_edge_indices)
        greedy_path = []
        while remaining:
            if time.time() > deadline:
                break
            # Pick the biclique that covers the most uncovered edges
            best_bi = None
            best_covered = 0
            for bi in range(len(biclique_edge_indices)):
                covered = biclique_edge_indices[bi] & remaining
                if len(covered) > best_covered:
                    best_covered = len(covered)
                    best_bi = bi
            if best_bi is None:
                break
            greedy_path.append(best_bi)
            remaining -= biclique_edge_indices[best_bi]
        path = greedy_path
    
    # Convert path to bicliques
    result = []
    for bi in path:
        L, R = filtered_bicliques[bi]
        result.append([list(L), list(R)])
    
    return result
