import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        # K edges
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        # I-K edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def bron_kerbosch(R, P, X, adj, cliques):
        """
        R: current clique (list of vertices)
        P: candidates (set of vertices)
        X: excluded vertices (set of vertices)
        adj: adjacency list (dict of sets)
        cliques: list to store maximal cliques
        """
        if not P and not X:
            if len(R) >= 2:
                cliques.append(list(R))
            return
        
        # Pivot selection: choose vertex in P ∪ X with max degree in P
        pivot = None
        max_deg = -1
        for v in P | X:
            deg = len(P & adj[v])
            if deg > max_deg:
                max_deg = deg
                pivot = v
        
        # Iterate over candidates that are NOT neighbors of pivot
        # This is the optimization from Tomita et al.
        candidates = P - adj[pivot] if pivot is not None else P
        
        for v in list(candidates):
            new_R = R + [v]
            new_P = P & adj[v]
            new_X = X & adj[v]
            bron_kerbosch(new_R, new_P, new_X, adj, cliques)
            P.remove(v)
            X.add(v)

    def solve_split(n, k, p, seed):
        if time.time() > deadline:
            return []

        # Regenerate graph to get adjacency
        rng = random.Random(seed)
        neighbors_in_k = []
        for v in range(k, n):
            nks = set()
            for u in range(k):
                if rng.random() < p:
                    nks.add(u)
            neighbors_in_k.append(nks)
        
        cliques = []
        covered_in_k = set()
        
        # Step 1: Process I vertices
        # Each I vertex v forms a clique with its neighbors in K.
        for i, nks in enumerate(neighbors_in_k):
            v = k + i
            if nks:
                clique = list(nks) + [v]
                cliques.append(clique)
                # Mark edges in nks as covered
                nks_list = list(nks)
                for j in range(len(nks_list)):
                    for l in range(j + 1, len(nks_list)):
                        u, w = nks_list[j], nks_list[l]
                        covered_in_k.add((min(u, w), max(u, w)))
        
        # Step 2: Process remaining K edges
        # Find all pairs in K not in covered_in_k
        remaining_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_in_k:
                    remaining_edges.add((u, w))
        
        if remaining_edges:
            # Build graph for remaining edges
            rem_adj = {u: set() for u in range(k)}
            for u, w in remaining_edges:
                rem_adj[u].add(w)
                rem_adj[w].add(u)
            
            # Find all maximal cliques in rem_adj
            maximal_cliques = []
            bron_kerbosch([], set(range(k)), set(), rem_adj, maximal_cliques)
            
            # Greedy Set Cover
            uncovered = set(remaining_edges)
            
            # Precompute edges covered by each maximal clique
            # Note: A clique C covers all pairs in C.
            # We only care about pairs in `uncovered`.
            clique_edges = []
            for mc in maximal_cliques:
                edges_in_mc = set()
                for i in range(len(mc)):
                    for j in range(i + 1, len(mc)):
                        e = (min(mc[i], mc[j]), max(mc[i], mc[j]))
                        if e in uncovered:
                            edges_in_mc.add(e)
                if edges_in_mc:
                    clique_edges.append((mc, edges_in_mc))
            
            # Sort cliques by number of edges covered (descending) initially?
            # Or just greedy pick best at each step.
            # Since the number of maximal cliques is likely small for sparse graphs,
            # we can just loop.
            
            while uncovered:
                if time.time() > deadline:
                    # Fallback: add remaining edges as single-edge cliques
                    for e in uncovered:
                        cliques.append([e[0], e[1]])
                    uncovered.clear()
                    break
                
                best_clique = None
                best_edges = None
                best_count = -1
                
                for mc, edges_in_mc in clique_edges:
                    # Intersect with currently uncovered
                    current_cover = edges_in_mc & uncovered
                    count = len(current_cover)
                    if count > best_count:
                        best_count = count
                        best_clique = mc
                        best_edges = current_cover
                
                if best_clique is None or best_count == 0:
                    # Should not happen if uncovered is not empty and clique_edges is correct
                    # But if a vertex is isolated in rem_adj, it won't be in any maximal clique of size >= 2?
                    # Wait, Bron-Kerbosch finds cliques of size >= 1 usually.
                    # My bron_kerbosch requires len(R) >= 2.
                    # If there is an isolated vertex, it won't be in any clique.
                    # But if there is an edge, it must be in a maximal clique.
                    # So if uncovered is not empty, there must be a clique.
                    # Unless my Bron-Kerbosch missed something.
                    # Let's add a fallback.
                    e = next(iter(uncovered))
                    cliques.append([e[0], e[1]])
                    uncovered.remove(e)
                    continue
                
                cliques.append(best_clique)
                uncovered.difference_update(best_edges)
            
        # Clean up types
        clean_cliques = []
        for c in cliques:
            clean_cliques.append([int(x) for x in c])
        return clean_cliques

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            res = solve_split(*spec)
            results.append(res)
            
    return results
