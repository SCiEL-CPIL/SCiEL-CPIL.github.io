import time
import random
import sys
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency matrix for O(1) edge check
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
        
    # Normalize edges
    norm_edges = set()
    for u, v in edges:
        if u < v:
            norm_edges.add((u, v))
        else:
            norm_edges.add((v, u))
            
    all_edges = list(norm_edges)
    m = len(all_edges)
    
    start_time = time.time()
    deadline = start_time + 55
    
    def is_biclique(L, R):
        """Check if (L, R) forms a valid biclique in the original graph."""
        for l in L:
            for r in R:
                if not adj[l][r]:
                    return False
        return True
    
    def edges_of_biclique(L, R):
        """Get the set of edges covered by biclique (L, R)."""
        result = set()
        for l in L:
            for r in R:
                e = (min(l, r), max(l, r))
                result.add(e)
        return result
    
    def expand_biclique(seed_u, seed_v, uncovered_edges):
        """
        Expand the seed edge (seed_u, seed_v) into a maximal biclique
        using only edges present in uncovered_edges.
        """
        L = {seed_u}
        R = {seed_v}
        
        changed = True
        while changed:
            changed = False
            
            # Try to add vertices to L
            candidates_L = []
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for r in R:
                    edge = (min(w, r), max(w, r))
                    if edge not in uncovered_edges:
                        can_add = False
                        break
                if can_add:
                    candidates_L.append(w)
            
            if candidates_L:
                L.update(candidates_L)
                changed = True
                continue
            
            # Try to add vertices to R
            candidates_R = []
            for w in range(n):
                if w in L or w in R:
                    continue
                can_add = True
                for l in L:
                    edge = (min(l, w), max(l, w))
                    if edge not in uncovered_edges:
                        can_add = False
                        break
                if can_add:
                    candidates_R.append(w)
            
            if candidates_R:
                R.update(candidates_R)
                changed = True
                continue
        
        covered_edges = set()
        for l in L:
            for r in R:
                covered_edges.add((min(l, r), max(l, r)))
                
        return L, R, covered_edges
    
    def greedy_set_cover(uncovered_edges, strategy='random'):
        """
        Greedy set cover: repeatedly pick a biclique covering the most uncovered edges.
        """
        uncovered = set(uncovered_edges)
        bicliques = []
        
        while uncovered:
            if time.time() > deadline:
                return None
            
            best_L = None
            best_R = None
            best_covered = None
            best_count = -1
            
            sample_size = min(len(uncovered), 30)
            candidates_seeds = random.sample(list(uncovered), sample_size)
            
            for seed_edge in candidates_seeds:
                u, v = seed_edge
                L, R, covered = expand_biclique(u, v, uncovered)
                
                if not L or not R:
                    continue
                
                count = len(covered)
                if count > best_count:
                    best_count = count
                    best_L = L
                    best_R = R
                    best_covered = covered
            
            if best_L is None:
                u, v = next(iter(uncovered))
                best_L = {u}
                best_R = {v}
                best_covered = {(u, v)}
            
            bicliques.append((list(best_L), list(best_R)))
            uncovered -= best_covered
            
        return bicliques
    
    def get_edges_from_bicliques(bicliques):
        edges = set()
        for L, R in bicliques:
            for u in L:
                for v in R:
                    edges.add((min(u, v), max(u, v)))
        return edges
    
    def local_search(bicliques):
        """
        Try to improve the partition by merging pairs of bicliques.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                break
                
            n_bicliques = len(bicliques)
            for i in range(n_bicliques):
                for j in range(i + 1, n_bicliques):
                    if time.time() > deadline:
                        return bicliques
                        
                    edges_i = get_edges_from_bicliques([bicliques[i]])
                    edges_j = get_edges_from_bicliques([bicliques[j]])
                    combined_edges = edges_i.union(edges_j)
                    
                    new_bicliques = greedy_set_cover(combined_edges, strategy='random')
                    
                    if new_bicliques is not None and len(new_bicliques) == 1:
                        new_b = new_bicliques[0]
                        bicliques.pop(j)
                        bicliques.pop(i)
                        bicliques.append(new_b)
                        improved = True
                        break
                if improved:
                    break
                    
        return bicliques
    
    def backtracking_partition(remaining_edges, max_bicliques, current_bicliques):
        """
        Backtracking search: try to cover remaining_edges with at most max_bicliques bicliques.
        Returns a list of bicliques if found, None otherwise.
        """
        if not remaining_edges:
            return current_bicliques[:]
        
        if len(current_bicliques) >= max_bicliques:
            return None
        
        if time.time() > deadline:
            return None
        
        # Pick the first uncovered edge
        first_edge = min(remaining_edges)
        u, v = first_edge
        
        # Generate candidate bicliques containing this edge
        # A biclique containing (u,v) must have u and v on opposite sides
        # Option 1: u in L, v in R
        # Option 2: v in L, u in R
        
        # For each option, find all maximal bicliques containing the edge
        # that only cover edges from remaining_edges
        
        candidates = []
        
        # Option 1: u in L, v in R
        L1 = {u}
        R1 = {v}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L1 or w in R1:
                    continue
                # Can we add w to L1?
                can_add_L = True
                for r in R1:
                    if (min(w, r), max(w, r)) not in remaining_edges:
                        can_add_L = False
                        break
                if can_add_L:
                    L1.add(w)
                    changed = True
            if changed:
                continue
            for w in range(n):
                if w in L1 or w in R1:
                    continue
                can_add_R = True
                for l in L1:
                    if (min(l, w), max(l, w)) not in remaining_edges:
                        can_add_R = False
                        break
                if can_add_R:
                    R1.add(w)
                    changed = True
        
        covered1 = edges_of_biclique(L1, R1)
        # Only keep edges that are in remaining
        covered1 = covered1 & remaining_edges
        if first_edge in covered1:
            candidates.append((L1, R1, covered1))
        
        # Option 2: v in L, u in R
        L2 = {v}
        R2 = {u}
        changed = True
        while changed:
            changed = False
            for w in range(n):
                if w in L2 or w in R2:
                    continue
                can_add_L = True
                for r in R2:
                    if (min(w, r), max(w, r)) not in remaining_edges:
                        can_add_L = False
                        break
                if can_add_L:
                    L2.add(w)
                    changed = True
            if changed:
                continue
            for w in range(n):
                if w in L2 or w in R2:
                    continue
                can_add_R = True
                for l in L2:
                    if (min(l, w), max(l, w)) not in remaining_edges:
                        can_add_R = False
                        break
                if can_add_R:
                    R2.add(w)
                    changed = True
        
        covered2 = edges_of_biclique(L2, R2)
        covered2 = covered2 & remaining_edges
        if first_edge in covered2:
            candidates.append((L2, R2, covered2))
        
        # Sort candidates by number of edges covered (more is better)
        candidates.sort(key=lambda x: -len(x[2]))
        
        # Try each candidate
        for L, R, covered in candidates:
            new_remaining = remaining_edges - covered
            result = backtracking_partition(new_remaining, max_bicliques, current_bicliques + [(list(L), list(R))])
            if result is not None:
                return result
        
        return None
    
    def exact_search_from_greedy(greedy_result, max_improve=3):
        """
        Starting from a greedy solution, try to find a better partition
        using backtracking with a tighter bound.
        """
        greedy_count = len(greedy_result)
        
        # Try to find a partition with fewer bicliques
        for target in range(1, greedy_count):
            if time.time() > deadline:
                break
            
            # Use backtracking to try to find a partition with 'target' bicliques
            result = backtracking_partition(set(all_edges), target, [])
            if result is not None:
                return result
        
        return greedy_result
    
    best_bicliques = None
    best_count = float('inf')
    
    # Phase 1: Greedy + local search with multiple restarts
    for restart in range(100):
        if time.time() > deadline:
            break
        
        random.seed(restart)
        
        result = greedy_set_cover(all_edges, strategy='random')
        
        if result is None:
            continue
            
        result = local_search(result)
        
        if len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            
        if best_count <= 2:
            break
    
    # Phase 2: Try exact backtracking search with a small number of bicliques
    if best_bicliques is not None:
        # Try to find a partition with fewer bicliques using backtracking
        for target in range(1, best_count):
            if time.time() > deadline:
                break
            
            result = backtracking_partition(set(all_edges), target, [])
            if result is not None:
                best_bicliques = result
                best_count = target
                break
    
    # Fallback
    if best_bicliques is None:
        best_bicliques = []
        uncovered = set(all_edges)
        while uncovered:
            u, v = next(iter(uncovered))
            best_bicliques.append([[u], [v]])
            uncovered.remove((u, v))

    # Convert to required format
    final_result = []
    for L, R in best_bicliques:
        final_result.append([list(L), list(R)])
        
    return final_result
