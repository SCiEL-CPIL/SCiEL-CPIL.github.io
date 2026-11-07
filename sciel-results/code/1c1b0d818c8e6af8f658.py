"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    from random import Random
    
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = []
        # Edges within the clique C (first k vertices)
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        
        # Edges between C and I (independent set)
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        # C vertices: 0 to k-1
        # I vertices: k to n-1
        
        N_C = {v: set() for v in range(k, n)}
        internal_edges = set()
        
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((min(u, v), max(u, v)))
            elif u < k and v >= k:
                N_C[v].add(u)
            elif u >= k and v < k:
                N_C[u].add(v)
            else:
                # No edges in I
                pass

        cliques = []
        E_rem = set(internal_edges)
        
        # Process I vertices
        for v in range(k, n):
            N_v = N_C[v]
            if not N_v:
                continue
                
            # We need to partition N_v into cliques of the graph (N_v, E_rem restricted to N_v)
            # Greedy: Find maximal clique, remove, repeat.
            remaining_nodes = set(N_v)
            
            while remaining_nodes:
                # Find a maximal clique in the subgraph induced by remaining_nodes with edges in E_rem
                # Start with an arbitrary node
                start = next(iter(remaining_nodes))
                clique = {start}
                candidates = list(remaining_nodes - {start})
                
                # Greedy expansion
                # Sort candidates by degree in current clique? Or just iterate.
                # For efficiency, simple greedy: add if connected to all in clique.
                # To get a large clique, we can try to order by degree in the induced subgraph.
                # But k is small (<=30), so simple is fine.
                
                # Let's just iterate through candidates and add if valid.
                # To maximize size, we might want to be smarter, but this is a valid heuristic.
                new_clique = set(clique)
                for c in candidates:
                    if all((min(c, x), max(c, x)) in E_rem for x in new_clique):
                        new_clique.add(c)
                
                # If new_clique is larger, use it
                if len(new_clique) > len(clique):
                    clique = new_clique
                else:
                    # Try starting with other nodes? 
                    # For now, just use the first maximal one found.
                    # To improve, we could try a few starts.
                    pass

                # Add clique {v} U clique to result
                if len(clique) > 0:
                    cliques.append([v] + list(clique))
                    
                    # Remove edges covered by this clique from E_rem
                    for u, w in combinations(clique, 2):
                        e = (min(u, w), max(u, w))
                        if e in E_rem:
                            E_rem.remove(e)
                            
                # Remove clique nodes from remaining_nodes
                remaining_nodes -= clique
                
        # Process remaining internal edges with C-cliques
        while E_rem:
            # Find maximal clique in (C, E_rem)
            # Vertices are 0..k-1
            # Get vertices involved in E_rem
            vertices_in_rem = set()
            for u, w in E_rem:
                vertices_in_rem.add(u)
                vertices_in_rem.add(w)
                
            best_clique = set()
            # Try to find a large clique
            # Simple greedy: pick a vertex, expand
            # To be more robust, try a few starting vertices
            starts = list(vertices_in_rem)
            for start in starts:
                clique = {start}
                candidates = list(vertices_in_rem - {start})
                new_clique = set(clique)
                for c in candidates:
                    if all((min(c, x), max(c, x)) in E_rem for x in new_clique):
                        new_clique.add(c)
                if len(new_clique) > len(best_clique):
                    best_clique = new_clique
            
            if not best_clique:
                # Should not happen if E_rem is not empty
                break
                
            cliques.append(list(best_clique))
            
            for u, w in combinations(best_clique, 2):
                e = (min(u, w), max(u, w))
                if e in E_rem:
                    E_rem.remove(e)

        # Convert to lists of ints
        final_cliques = [list(map(int, c)) for c in cliques]
        
        return final_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for n, k, p, seed in specs:
        cliques = solve_instance(n, k, p, seed)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
