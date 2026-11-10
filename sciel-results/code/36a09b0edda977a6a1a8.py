import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)

        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        C_vertices = list(range(k))
        I_vertices = list(range(k, n))

        # For each i in I, get its neighbors in C
        N_C = [adj[i] & set(C_vertices) for i in I_vertices]

        result_cliques = []
        covered_CC = set()

        # Step 1: Create mixed cliques for all active I vertices
        for idx, i in enumerate(I_vertices):
            if N_C[idx]:
                clique = [i] + list(N_C[idx])
                result_cliques.append(clique)
                # Mark all C-C edges in this clique as covered
                c_part = list(N_C[idx])
                for a, b in combinations(c_part, 2):
                    covered_CC.add((min(a, b), max(a, b)))

        # Step 2: Identify remaining C-C edges
        all_CC = set()
        for a, b in combinations(C_vertices, 2):
            all_CC.add((a, b))

        remaining = all_CC - covered_CC
        remaining_list = list(remaining)

        # Step 3: Cover remaining C-C edges with minimum number of cliques
        if remaining_list:
            # Build graph G_rem on C_vertices with edges = remaining
            # We need min edge clique cover of G_rem
            
            # Find all maximal cliques in G_rem
            # Since k is small (<=30) and G_rem is likely sparse, we can use a simple algorithm
            # Or just use the edges themselves if the graph is a matching
            # Let's find all cliques of size >= 2 in G_rem
            
            # Adjacency for G_rem
            G_rem_adj = {v: set() for v in C_vertices}
            for u, v in remaining_list:
                G_rem_adj[u].add(v)
                G_rem_adj[v].add(u)
                
            # Find maximal cliques
            # Simple approach: iterate all subsets? No, 2^30 is too big.
            # Use Bron-Kerbosch or similar?
            # Given time constraints, let's try a greedy set cover with a better heuristic or exact B&B on edges
            
            # If the number of remaining edges is small, we can solve exactly
            # Let's try a simple B&B on the edges
            # State: set of uncovered edges
            # Action: pick a clique, remove its edges
            
            # To make it fast, we only consider cliques that are maximal in G_rem
            # Generate maximal cliques
            maximal_cliques = []
            
            # Helper to find maximal cliques
            def expand(clique, candidates):
                if not candidates:
                    if len(clique) >= 2:
                        maximal_cliques.append(frozenset(clique))
                    return
                # Pick a vertex
                v = next(iter(candidates))
                # Branch 1: include v
                expand(clique | {v}, candidates & G_rem_adj[v])
                # Branch 2: exclude v
                expand(clique, candidates - {v})
                
            # Start with all vertices that have edges
            active = set()
            for u, v in remaining_list:
                active.add(u)
                active.add(v)
                
            # This might be slow if G_rem is dense, but G_rem is sparse
            # To avoid exponential blowup, we limit the search or use a heuristic if it takes too long
            # For now, assume it's fast enough for k=30 sparse graph
            
            # Optimization: Only expand if candidates are non-empty
            # We need to be careful with the recursion depth and time
            # If time is running out, fall back to greedy
            
            if time.time() < DEADLINE - 10:
                # Try to find maximal cliques
                # To speed up, we can just use the edges as cliques if the graph is a matching
                # Check if G_rem is a matching
                is_matching = True
                deg = {v: len(G_rem_adj[v]) for v in active}
                for v in active:
                    if deg[v] > 1:
                        is_matching = False
                        break
                
                if is_matching:
                    # Each edge is a clique
                    for u, v in remaining_list:
                        result_cliques.append([u, v])
                else:
                    # Find maximal cliques
                    # Use a simpler greedy to generate candidate cliques for set cover
                    # Or just use the greedy from seed but with a twist
                    # Let's stick to a robust greedy for now if B&B is risky
                    # Actually, let's just use the seed's greedy but ensure it's correct
                    # The seed's greedy was: find max clique, remove edges
                    # This is equivalent to greedy set cover where sets are cliques
                    # To improve, we can try multiple starting points or use a better clique finder
                    
                    # For now, let's just replicate the seed's logic for remaining edges
                    # but ensure we don't exceed time
                    rem_set = set(remaining_list)
                    while rem_set:
                        # Find max clique in G_rem induced by active vertices
                        active_v = set()
                        for u, v in rem_set:
                            active_v.add(u)
                            active_v.add(v)
                        
                        if not active_v:
                            break
                        
                        # Greedy max clique
                        best_clique = set()
                        for start in active_v:
                            clique = {start}
                            cands = set(G_rem_adj[start]) & active_v
                            while cands:
                                # Pick vertex with highest degree in cands
                                best_cand = max(cands, key=lambda x: len(cands & G_rem_adj[x]))
                                clique.add(best_cand)
                                cands = cands & G_rem_adj[best_cand]
                            if len(clique) > len(best_clique):
                                best_clique = clique
                                
                        if not best_clique:
                            break
                            
                        # Add clique
                        result_cliques.append(sorted(best_clique))
                        
                        # Remove edges
                        clique_edges = set()
                        for u, v in combinations(sorted(best_clique), 2):
                            pair = (min(u, v), max(u, v))
                            if pair in rem_set:
                                clique_edges.add(pair)
                        rem_set -= clique_edges

        # Convert to lists of ints
        final_result = []
        for clique in result_cliques:
            final_clique = [int(v) for v in clique]
            if len(final_clique) >= 2:
                final_result.append(final_clique)
        
        return final_result

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > DEADLINE:
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            trivial = [[int(u), int(v)] for u, v in edges]
            results.append(trivial)
        else:
            res = solve_instance(*spec)
            results.append(res)
            
    return results
