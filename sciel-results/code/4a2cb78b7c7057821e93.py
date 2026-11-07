"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin for 120s limit

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def bron_kerbosch_maximal_cliques(adj, vertices, pivot=None):
        """
        Yields all maximal cliques in the graph defined by adj and restricted to vertices.
        Uses Bron-Kerbosch with pivot.
        """
        if not vertices:
            return
        if pivot is None:
            pivot = set()
        
        # Choose pivot: vertex with max degree in the candidate set (or neighbors of pivot)
        # Standard pivot: choose u from P U X that maximizes |N(u) ∩ P|
        # Here we simplify: just pick a vertex from P if possible, else from X.
        # Actually, for finding *maximal* cliques, we iterate over P \ N(pivot).
        
        # For efficiency in small graphs, we can just iterate over P if P is small.
        # But let's implement standard pivot.
        if pivot is None:
            pivot_set = set()
        else:
            pivot_set = pivot
            
        # Find best pivot
        best_pivot = None
        max_count = -1
        # Candidates for pivot are in P or X (which is empty in recursive calls if we manage it right, but here we pass current state)
        # Actually, the standard algorithm passes P, X.
        # Let's use a simpler version: Just enumerate maximal cliques containing a specific vertex?
        # No, we need all maximal cliques to choose the best one.
        
        # Given k<=30, a simpler DFS for maximal cliques might be sufficient if we prune.
        # Let's try a recursive backtracking to find maximal cliques.
        
        def expand(clique, candidates, excluded):
            if not candidates:
                if not excluded:
                    yield list(clique)
                return
            
            # Choose pivot to reduce branching
            # Pick vertex in candidates with max degree in candidates
            pivot_vertex = max(candidates, key=lambda v: len(adj[v] & candidates))
            # Branch on candidates \ N(pivot_vertex)
            for v in list(candidates - adj[pivot_vertex]):
                new_clique = clique + [v]
                new_candidates = candidates & adj[v]
                new_excluded = excluded & adj[v]
                yield from expand(new_clique, new_candidates, new_excluded)
                candidates.remove(v)
                excluded.add(v)
                
        yield from expand([], set(vertices), set())

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        if not edges:
            return []
        
        # Build adjacency sets
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
        
        cliques = []
        remaining_edges = set(edge_set)
        
        # Step 1: Cover cross edges
        for w in range(k, n):
            if time.time() > deadline:
                break
            neighbors = [u for u in range(k) if (u, w) in remaining_edges]
            if not neighbors:
                continue
            
            clique = neighbors + [w]
            # Remove all edges in this clique
            for i in range(len(clique)):
                for j in range(i + 1, len(clique)):
                    u, v = clique[i], clique[j]
                    e = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e)
            cliques.append(clique)
        
        # Step 2: Cover remaining edges within 0..k-1
        # Build adjacency for the residual graph on 0..k-1
        res_adj = [set() for _ in range(k)]
        for u, v in remaining_edges:
            res_adj[u].add(v)
            res_adj[v].add(u)
            
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return remaining edges as individual cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                break
            
            # Find all maximal cliques in the residual graph
            # We restrict to vertices that have at least one edge in remaining_edges
            active_vertices = set()
            for u, v in remaining_edges:
                active_vertices.add(u)
                active_vertices.add(v)
                
            maximal_cliques = list(bron_kerbosch_maximal_cliques(res_adj, active_vertices))
            
            if not maximal_cliques:
                # Should not happen if remaining_edges is not empty, but just in case
                # Pick any edge
                u, v = next(iter(remaining_edges))
                maximal_cliques = [[u, v]]
            
            # Select the clique that covers the most edges
            best_clique = None
            max_edges_covered = -1
            
            for mc in maximal_cliques:
                # Count edges in mc that are in remaining_edges
                count = 0
                for i in range(len(mc)):
                    for j in range(i + 1, len(mc)):
                        u, v = mc[i], mc[j]
                        e = (u, v) if u < v else (v, u)
                        if e in remaining_edges:
                            count += 1
                if count > max_edges_covered:
                    max_edges_covered = count
                    best_clique = mc
            
            if best_clique is None or max_edges_covered == 0:
                # Fallback: pick a single edge
                u, v = next(iter(remaining_edges))
                best_clique = [u, v]
            
            cliques.append(best_clique)
            
            # Remove all edges in this clique from remaining_edges and update res_adj
            for i in range(len(best_clique)):
                for j in range(i + 1, len(best_clique)):
                    u, v = best_clique[i], best_clique[j]
                    e = (u, v) if u < v else (v, u)
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        res_adj[u].discard(v)
                        res_adj[v].discard(u)
        
        return cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # If we run out of time, we have to return something valid.
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            # Ensure types are correct
            clean_res = []
            for clique in res:
                clean_clique = [int(v) for v in clique]
                clean_res.append(clean_clique)
            results.append(clean_res)
            
    return results
# EVOLVE-BLOCK-END
