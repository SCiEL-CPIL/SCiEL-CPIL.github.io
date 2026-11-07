"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # Clique on first k
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # Cross edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        """Solve one instance using edge-disjoint matching decomposition."""
        edges = instance(n, k, p, seed)
        
        # Separate internal and cross edges
        internal_edges = set()
        cross_adj = {}  # v in I -> set of u in C
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((min(u,v), max(u,v)))
            else:
                if u < k:
                    cross_adj.setdefault(v, set()).add(u)
                else:
                    cross_adj.setdefault(u, set()).add(v)
        
        uncovered_internal = set(internal_edges)
        cliques = []
        
        # Key insight: For each I-vertex v, we want to partition N(v) into subsets
        # S_1, S_2, ... such that each S_i forms a valid clique with v (all internal
        # edges within S_i are uncovered). We want to minimize the number of subsets.
        # 
        # Better approach: Process I-vertices in order of decreasing degree, and for
        # each, greedily build cliques that maximize internal edge coverage.
        
        # Sort I-vertices by degree (descending) to prioritize high-degree vertices
        i_vertices = sorted(cross_adj.keys(), key=lambda x: -len(cross_adj[x]))
        
        for v in i_vertices:
            if time.time() > deadline:
                break
            neighbors = set(cross_adj[v])
            remaining = set(neighbors)
            
            while remaining:
                # Greedy: build a maximal subset S of remaining where all pairs
                # have uncovered internal edges.
                # Use a smarter heuristic: try multiple starting points and pick
                # the best resulting clique.
                
                best_S = set()
                best_score = -1
                
                # Try each vertex in remaining as a starting point
                for start in list(remaining):
                    S = {start}
                    # Try to add other vertices
                    for u in remaining:
                        if u == start:
                            continue
                        # Check if u can be added to S
                        ok = True
                        for w in S:
                            e = (min(u, w), max(u, w))
                            if e not in uncovered_internal:
                                ok = False
                                break
                        if ok:
                            S.add(u)
                    
                    # Score: number of internal edges this clique would cover
                    s_list = list(S)
                    score = 0
                    for i in range(len(s_list)):
                        for j in range(i+1, len(s_list)):
                            e = (min(s_list[i], s_list[j]), max(s_list[i], s_list[j]))
                            if e in uncovered_internal:
                                score += 1
                    
                    if score > best_score or (score == best_score and len(S) > len(best_S)):
                        best_score = score
                        best_S = set(S)
                
                if not best_S:
                    # Fallback: single vertex
                    best_S = {next(iter(remaining))}
                
                # Form clique
                clique = [v] + sorted(best_S)
                cliques.append(clique)
                
                # Mark internal edges in best_S as covered
                s_list = list(best_S)
                for i in range(len(s_list)):
                    for j in range(i+1, len(s_list)):
                        e = (min(s_list[i], s_list[j]), max(s_list[i], s_list[j]))
                        uncovered_internal.discard(e)
                
                remaining -= best_S
        
        # Cover remaining internal edges as 2-cliques
        for e in uncovered_internal:
            cliques.append(list(e))
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return edge-by-edge cliques
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([list(e) for e in edges])
            continue
            
        n, k, p, seed = spec
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
    
    # Ensure all types are correct
    final_results = []
    for cliques in results:
        clean_cliques = []
        for c in cliques:
            if isinstance(c, (list, tuple)) and len(c) >= 2:
                clean_cliques.append([int(x) for x in c])
        final_results.append(clean_cliques)
    
    return final_results

# EVOLVE-BLOCK-END
