import time
import random
import sys
from collections import defaultdict
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
    
    # Precompute adjacency as bitsets for faster operations
    adj_bits = [0] * n
    for u, v in edges:
        adj_bits[u] |= (1 << v)
        adj_bits[v] |= (1 << u)
    
    start_time = time.time()
    deadline = start_time + 55
    
    def expand_biclique(seed_u, seed_v, uncovered_edges):
        """
        Expand the seed edge (seed_u, seed_v) into a maximal biclique
        using only edges present in uncovered_edges. Uses bitsets for speed.
        """
        # Build residual adjacency bitsets from uncovered_edges
        res_adj = [0] * n
        for u, v in uncovered_edges:
            res_adj[u] |= (1 << v)
            res_adj[v] |= (1 << u)
        
        L = 1 << seed_u
        R = 1 << seed_v
        
        changed = True
        while changed:
            changed = False
            
            # Candidates for L: not in L or R, and connected to all in R
            # A vertex w can join L iff (res_adj[w] & R) == R
            candidates_L = 0
            for w in range(n):
                if (L >> w) & 1 or (R >> w) & 1:
                    continue
                if (res_adj[w] & R) == R:
                    candidates_L |= (1 << w)
            
            if candidates_L:
                L |= candidates_L
                changed = True
                continue
            
            # Candidates for R: not in L or R, and connected to all in L
            candidates_R = 0
            for w in range(n):
                if (L >> w) & 1 or (R >> w) & 1:
                    continue
                if (res_adj[w] & L) == L:
                    candidates_R |= (1 << w)
            
            if candidates_R:
                R |= candidates_R
                changed = True
                continue
        
        # Convert bitsets to lists
        L_list = [w for w in range(n) if (L >> w) & 1]
        R_list = [w for w in range(n) if (R >> w) & 1]
        
        # The resulting biclique covers all edges between L and R
        covered_edges = set()
        for l in L_list:
            for r in R_list:
                covered_edges.add((min(l, r), max(l, r)))
                
        return L_list, R_list, covered_edges

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
            
            # Compute degree of each edge in the residual graph for better seeding
            edge_degree = {}
            for edge in uncovered:
                u, v = edge
                edge_degree[edge] = 0
            
            # For each vertex, count how many uncovered edges it's incident to
            vert_degree = defaultdict(int)
            for u, v in uncovered:
                vert_degree[u] += 1
                vert_degree[v] += 1
            
            # Score edges by product of endpoint degrees (prefer high-degree edges)
            scored_edges = []
            for edge in uncovered:
                u, v = edge
                score = vert_degree[u] * vert_degree[v]
                scored_edges.append((score, edge))
            
            # Sort by score descending, take top candidates
            scored_edges.sort(reverse=True)
            
            # Sample from top candidates and some random ones
            top_k = min(len(scored_edges), 30)
            top_candidates = [edge for _, edge in scored_edges[:top_k]]
            
            # Add some random candidates for diversity
            remaining = len(uncovered) - top_k
            if remaining > 0:
                random_candidates = random.sample(list(uncovered), min(15, remaining))
                top_candidates.extend(random_candidates)
            
            for seed_edge in top_candidates:
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
                # Fallback
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
                        
                    # Get edges covered by biclique i and j
                    edges_i = get_edges_from_bicliques([bicliques[i]])
                    edges_j = get_edges_from_bicliques([bicliques[j]])
                    combined_edges = edges_i.union(edges_j)
                    
                    # Try to cover combined_edges with 1 biclique
                    # First try direct merge checks (fast), then fall back to greedy
                    
                    L1, R1 = bicliques[i]
                    L2, R2 = bicliques[j]
                    
                    merged = False
                    # Try all 4 orientations of combining the two bicliques
                    orientations = [
                        (set(L1) | set(L2), set(R1) | set(R2)),
                        (set(L1) | set(R2), set(R1) | set(L2)),
                        (set(R1) | set(L2), set(L1) | set(R2)),
                        (set(R1) | set(R2), set(L1) | set(L2)),
                    ]
                    
                    for new_L_set, new_R_set in orientations:
                        if not new_L_set or not new_R_set:
                            continue
                        # Check if all cross edges exist in the original graph
                        valid = True
                        for u in new_L_set:
                            for v in new_R_set:
                                if not adj[u][v]:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            # Verify that the combined biclique covers exactly combined_edges
                            covered = set()
                            for u in new_L_set:
                                for v in new_R_set:
                                    covered.add((min(u, v), max(u, v)))
                            if covered == combined_edges:
                                bicliques.pop(j)
                                bicliques.pop(i)
                                bicliques.append((list(new_L_set), list(new_R_set)))
                                improved = True
                                merged = True
                                break
                        if merged:
                            break
                    
                    if not merged:
                        # Fall back to greedy on combined edges
                        new_bicliques = greedy_set_cover(combined_edges, strategy='random')
                        if new_bicliques is not None and len(new_bicliques) == 1:
                            new_b = new_bicliques[0]
                            bicliques.pop(j)
                            bicliques.pop(i)
                            bicliques.append(new_b)
                            improved = True
                    
                    if improved:
                        break # Restart the outer loop as indices changed
                if improved:
                    break
                    
        return bicliques

    best_bicliques = None
    best_count = float('inf')
    
    # Run multiple restarts with different random seeds
    restart = 0
    while time.time() < deadline:
        random.seed(restart)
        
        # Initial Greedy
        result = greedy_set_cover(all_edges, strategy='random')
        
        if result is None:
            restart += 1
            continue
            
        # Local Search
        result = local_search(result)
        
        if len(result) < best_count:
            best_count = len(result)
            best_bicliques = result
            
        if best_count == 1:
            break
            
        # If we have time, try a second local search pass with different randomization
        if time.time() < deadline - 5:
            result2 = list(result)
            random.seed(restart + 1000)
            result2 = local_search(result2)
            if len(result2) < best_count:
                best_count = len(result2)
                best_bicliques = result2
            if best_count == 1:
                break
        
        restart += 1

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
        # Ensure types are list of int
        final_result.append([list(L), list(R)])
        
    return final_result
