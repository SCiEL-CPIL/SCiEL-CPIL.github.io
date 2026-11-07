"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 145

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def min_clique_cover(adj, n):
        """Find minimum clique cover of a graph given by adjacency sets.
        adj[v] is a set of neighbors. Returns list of cliques (each a list of ints)."""
        if n == 0:
            return []
        
        # Build edge set
        edge_set = set()
        for v in range(n):
            for u in adj[v]:
                if u > v:
                    edge_set.add((v, u))
        
        if not edge_set:
            return []
        
        # If it's a complete graph, one clique
        if len(edge_set) == n * (n - 1) // 2:
            return [list(range(n))]
        
        # Greedy: repeatedly find a maximal clique, remove its edges
        cliques = []
        remaining_edges = set(edge_set)
        adj_rem = [set() for _ in range(n)]
        for (a, b) in remaining_edges:
            adj_rem[a].add(b)
            adj_rem[b].add(a)
        
        while remaining_edges:
            # Try to find a large clique greedily
            best_clique = None
            best_size = 0
            
            # Try starting from each vertex with remaining edges
            for start in range(n):
                if not adj_rem[start]:
                    continue
                # Greedy clique expansion
                clique = [start]
                candidates = list(adj_rem[start])
                for w in candidates:
                    if all(w in adj_rem[x] for x in clique):
                        clique.append(w)
                
                if len(clique) > best_size:
                    best_size = len(clique)
                    best_clique = clique
            
            if best_clique is None or len(best_clique) < 2:
                # Should not happen if there are edges
                break
            
            # Remove edges within this clique
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    a, b = best_clique[i], best_clique[j]
                    e = (a, b) if a < b else (b, a)
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    adj_rem[a].discard(b)
                    adj_rem[b].discard(a)
            
            cliques.append(best_clique)
        
        return cliques

    def clique_partition(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build edge set for quick lookup
        edge_set = set()
        for (a, b) in edges:
            edge_set.add((a, b) if a < b else (b, a))
        
        # Find neighbors in C for each I-vertex
        N_C = {}
        for v in range(k, n):
            neigh = frozenset(u for u in range(k) if (u, v) in edge_set)
            N_C[v] = neigh
        
        # Sort I-vertices by degree (descending)
        i_vertices = sorted(range(k, n), key=lambda v: len(N_C[v]), reverse=True)
        
        cliques = []
        used_cc_edges = set()  # C-C edges already consumed by Type B cliques
        
        # For each I-vertex, try to use one big clique with all its neighbors
        # Check for conflicts with already used C-C edges
        for v in i_vertices:
            neighbors = list(N_C[v])
            d = len(neighbors)
            
            if d == 0:
                continue
            
            if d == 1:
                # Must use {v, u}
                u = neighbors[0]
                cliques.append([v, u])
                continue
            
            # Try to use {v} ∪ neighbors as one clique
            # Check if all C-C edges within neighbors are available
            conflict = False
            for i in range(d):
                for j in range(i + 1, d):
                    a, b = neighbors[i], neighbors[j]
                    e = (a, b) if a < b else (b, a)
                    if e in used_cc_edges:
                        conflict = True
                        break
                if conflict:
                    break
            
            if not conflict:
                # Use one big clique
                cliques.append([v] + neighbors)
                # Mark C-C edges as used
                for i in range(d):
                    for j in range(i + 1, d):
                        a, b = neighbors[i], neighbors[j]
                        e = (a, b) if a < b else (b, a)
                        used_cc_edges.add(e)
            else:
                # Fall back to individual edge cliques
                for u in neighbors:
                    cliques.append([v, u])
        
        # Cover remaining C-C edges with C-only cliques
        all_cc_edges = set((u, v) for u in range(k) for v in range(u + 1, k))
        remaining_cc = all_cc_edges - used_cc_edges
        
        if remaining_cc:
            # Build adjacency for remaining C-C edges
            rem_adj = [set() for _ in range(k)]
            for (a, b) in remaining_cc:
                rem_adj[a].add(b)
                rem_adj[b].add(a)
            
            # Find minimum clique cover
            cc_cliques = min_clique_cover(rem_adj, k)
            cliques.extend(cc_cliques)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline - 5:
            # Fallback: just do the simple thing quickly
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            cliques = []
            if k >= 2:
                cliques.append(list(range(k)))
            for (u, v) in edges:
                if not (u < k and v < k):
                    cliques.append([u, v])
            results.append(cliques)
        else:
            results.append(clique_partition(*spec))
    
    return results
# EVOLVE-BLOCK-END
