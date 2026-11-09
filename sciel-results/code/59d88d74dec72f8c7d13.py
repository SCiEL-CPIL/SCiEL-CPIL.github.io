"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    n = P['n']
    edges = P['edges']
    
    # Build adjacency sets for quick lookup
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Convert edges to frozenset for O(1) lookup
    edge_set = set()
    for u, v in edges:
        if u < v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    
    def is_biclique(L, R):
        """Check if L x R is a biclique (all cross edges exist)."""
        for u in L:
            for v in R:
                if u < v:
                    if (u, v) not in edge_set:
                        return False
                elif v < u:
                    if (v, u) not in edge_set:
                        return False
                else:
                    return False  # self-loop
        return True
    
    def biclique_edges(L, R):
        """Get the set of edges covered by biclique (L, R)."""
        es = set()
        for u in L:
            for v in R:
                if u < v:
                    es.add((u, v))
                elif v < u:
                    es.add((v, u))
        return es
    
    def find_max_biclique_from_vertex(v, remaining_edges, adj_remaining):
        """Find a large biclique containing vertex v using greedy expansion."""
        # Start with v on the left
        L = {v}
        # Right side: all neighbors of v in remaining graph
        R = set(adj_remaining[v])
        
        # Greedily expand: try adding vertices to either side
        improved = True
        while improved:
            improved = False
            
            # Try adding a vertex to L (must be connected to all of R)
            candidates_L = []
            for u in remaining_edges:
                if u in L:
                    continue
                # u must be connected to all vertices in R
                if all(w in adj_remaining[u] for w in R):
                    # Check that adding u to L doesn't break anything
                    # (it's fine as long as u connects to all R)
                    candidates_L.append(u)
            
            # Try adding a vertex to R (must be connected to all of L)
            candidates_R = []
            for u in remaining_edges:
                if u in R:
                    continue
                # u must be connected to all vertices in L
                if all(w in adj_remaining[u] for w in L):
                    candidates_R.append(u)
            
            # Choose the side that gives more coverage
            if candidates_L and candidates_R:
                # Score each candidate by how many edges it would add
                best_gain = -1
                best_choice = None
                
                for u in candidates_L[:20]:  # limit for speed
                    # Adding u to L: gains edges from u to all R
                    gain = len(R)
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = ('L', u)
                
                for u in candidates_R[:20]:
                    gain = len(L)
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = ('R', u)
                
                if best_choice:
                    side, u = best_choice
                    if side == 'L':
                        L.add(u)
                    else:
                        R.add(u)
                    improved = True
            elif candidates_L:
                # Pick the one with highest degree
                best = max(candidates_L[:20], key=lambda x: len(adj_remaining[x]))
                L.add(best)
                improved = True
            elif candidates_R:
                best = max(candidates_R[:20], key=lambda x: len(adj_remaining[x]))
                R.add(best)
                improved = True
        
        # Ensure non-empty
        if not L or not R:
            return None
        
        # Verify it's a valid biclique
        if not is_biclique(list(L), list(R)):
            return None
        
        return (L, R)
    
    def greedy_biclique_partition(remaining_edges_set, adj_remaining, time_limit):
        """Greedy set-cover: repeatedly pick the biclique covering most uncovered edges."""
        bicliques = []
        covered = set()
        
        while remaining_edges_set and time.time() < time_limit:
            best_biclique = None
            best_coverage = 0
            
            # Try to find a good biclique from each vertex in remaining edges
            vertices_in_edges = set()
            for e in remaining_edges_set:
                vertices_in_edges.add(e[0])
                vertices_in_edges.add(e[1])
            
            # Sample vertices for speed
            if len(vertices_in_edges) > 50:
                sample = random.sample(list(vertices_in_edges), 50)
            else:
                sample = list(vertices_in_edges)
            
            for v in sample:
                L, R = find_max_biclique_from_vertex(v, remaining_edges_set, adj_remaining)
                if L is None:
                    continue
                
                # Only count uncovered edges
                edges_here = biclique_edges(L, R)
                uncovered_here = edges_here - covered
                cov = len(uncovered_here)
                
                if cov > best_coverage:
                    best_coverage = cov
                    best_biclique = (L, R, uncovered_here)
            
            if best_biclique is None or best_coverage == 0:
                # Fallback: single edge
                e = next(iter(remaining_edges_set))
                bicliques.append(([e[0]], [e[1]]))
                covered.add(e)
                remaining_edges_set.discard(e)
                continue
            
            L, R, unc = best_biclique
            bicliques.append((list(L), list(R)))
            covered.update(unc)
            
            # Remove covered edges from remaining
            remaining_edges_set -= covered
            
            # Update adj_remaining
            for e in remaining_edges_set:
                pass  # adj_remaining is static; we just track which edges remain
        
        return bicliques
    
    # Initial greedy partition
    remaining = set(edge_set)
    bicliques = greedy_biclique_partition(remaining, adj, deadline - 5)
    
    # Local improvement: try to merge adjacent bicliques
    def try_merge(bicliques_list):
        """Try to merge pairs of bicliques if their union forms a valid biclique."""
        improved = True
        while improved:
            improved = False
            for i in range(len(bicliques_list)):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, len(bicliques_list)):
                    L1, R1 = bicliques_list[i]
                    L2, R2 = bicliques_list[j]
                    
                    # Try merging: new L = L1 ∪ L2, new R = R1 ∪ R2
                    # But we need to check if this is a valid biclique
                    new_L = set(L1) | set(L2)
                    new_R = set(R1) | set(R2)
                    
                    # For a merge to work, we need all cross edges between new_L and new_R
                    # Check: edges between L1 and R2, and between L2 and R1 must exist
                    valid = True
                    for u in L1:
                        for v in R2:
                            if u < v:
                                if (u, v) not in edge_set:
                                    valid = False
                                    break
                            elif v < u:
                                if (v, u) not in edge_set:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        for u in L2:
                            for v in R1:
                                if u < v:
                                    if (u, v) not in edge_set:
                                        valid = False
                                        break
                                elif v < u:
                                    if (v, u) not in edge_set:
                                        valid = False
                                        break
                            if not valid:
                                break
                    
                    if valid and new_L and new_R:
                        # Merge
                        bicliques_list[i] = (list(new_L), list(new_R))
                        bicliques_list.pop(j)
                        improved = True
                        break
                if improved:
                    break
        return bicliques_list
    
    bicliques = try_merge(bicliques)
    
    # Convert to required format
    result = []
    for L, R in bicliques:
        result.append([list(L), list(R)])
    
    return result
# EVOLVE-BLOCK-END
