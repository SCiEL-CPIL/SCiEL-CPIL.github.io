"""Strategy: Global Greedy Set Cover for Clique Partition."""

def construct():
    import time
    from itertools import combinations

    def instance(n, k, p, seed):
        # The task contract says: "The seed program contains this exact generator, 
        # under the name instance; keep it."
        # However, the seed code provided in the prompt defines instance locally.
        # I will replicate the exact logic from the prompt's seed program to ensure 
        # the graph is identical.
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def _solve_instance(n, k, p, seed, deadline):
        # 1. Generate Graph
        from random import Random
        rng = Random(seed)
        
        # Core adjacency (0..k-1)
        core_adj = [set() for _ in range(k)]
        for u in range(k):
            for v in range(u+1, k):
                core_adj[u].add(v)
                core_adj[v].add(u)
                
        # Cross edges and I-vertex neighbors
        # i is in [k, n)
        i_neighbors = [set() for _ in range(n)] # Only k..n-1 used
        cross_edges = []
        for i in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    i_neighbors[i].add(u)
                    cross_edges.append((u, i))
        
        # Total edges to cover
        all_edges = set()
        for u in range(k):
            for v in range(u+1, k):
                all_edges.add((u, v))
        for e in cross_edges:
            all_edges.add(e)
            
        # 2. Generate Candidate Cliques
        # A candidate clique is either:
        # A) {i} U S where S is a clique in Core induced by N_i
        # B) S where S is a clique in Core (pure core clique)
        
        # We will use a greedy set cover approach.
        # Universe: all_edges
        # Sets: Candidates
        
        # To avoid generating too many candidates (exponential in worst case),
        # we will generate "maximal" cliques in N_i.
        # Since k is small (10, 20, 30), we can use Bron-Kerbosch to find maximal cliques.
        
        def bron_kerbosch_maximal_cliques(vertices, adj):
            # vertices: set of ints
            # adj: list of sets
            # Yields maximal cliques
            if not vertices:
                return
            
            # Pivot selection
            # Choose u in vertices U (excluded) maximizing |vertices intersect N(u)|
            # Here we don't have 'excluded' in the simple recursive step for maximal cliques 
            # unless we implement the standard BK with P, X.
            # Let's implement standard BK for maximal cliques.
            
            def bk(P, X):
                if not P and not X:
                    yield set()
                    return
                    
                # Pivot
                if P:
                    # Union P and X for pivot candidates
                    all_cands = P.union(X)
                    max_count = -1
                    pivot = None
                    for u in all_cands:
                        count = 0
                        for v in P:
                            if v in adj[u]:
                                count += 1
                        if count > max_count:
                            max_count = count
                            pivot = u
                    # Candidates are P \ N(pivot)
                    candidates = [v for v in P if pivot is None or v not in adj[pivot]]
                else:
                    candidates = []
                
                for v in candidates:
                    new_P = P.intersection(adj[v])
                    new_X = X.intersection(adj[v])
                    for clique in bk(new_P, new_X):
                        yield clique.union({v})
                    P.remove(v)
                    X.add(v)
                    
            return bk(set(vertices), set())

        candidates = []
        
        # Time check
        if time.time() > deadline:
             return []
             
        # Generate candidates for each I-vertex
        for i in range(k, n):
            N_i = i_neighbors[i]
            if not N_i:
                continue
                
            # Find maximal cliques in the subgraph induced by N_i
            # We need the core adjacency restricted to N_i
            # bron_kerbosch_maximal_cliques works on global adj, but we pass N_i as vertices.
            # It will only use edges between vertices in N_i.
            
            for S in bron_kerbosch_maximal_cliques(N_i, core_adj):
                if len(S) < 2:
                    # A clique of size 1 with i is a 2-clique {i, c}.
                    # We might want these as candidates too if they cover edges.
                    # But 2-cliques are always available as fallback.
                    # Including them as candidates helps if they are part of a "good" set.
                    # However, set cover prefers larger sets.
                    # Let's include only S size >= 2 for now to reduce candidate count,
                    # and rely on fallback for singletons.
                    # Actually, if N_i is large, maximal cliques are likely large.
                    pass 
                else:
                    # Candidate: {i} U S
                    # Edges covered:
                    # 1. Cross: (c, i) for c in S
                    # 2. Core: (u, v) for u, v in S
                    covered = set()
                    for c in S:
                        covered.add((c, i))
                    for u in S:
                        for v in S:
                            if u < v:
                                covered.add((u, v))
                    candidates.append((set([i]).union(S), covered))
                    
        # Also add pure core cliques?
        # Pure core cliques don't cover cross edges.
        # If we have a core edge (u,v) not covered by any {i}U S, we need a core clique.
        # Greedy set cover will pick {i}U S first if it covers more.
        # If we run out of {i}U S candidates that cover uncovered edges, we need core cliques.
        # Generating all core cliques is expensive.
        # Strategy: Only generate core cliques if necessary in the loop?
        # Or, simpler: The fallback handles 2-cliques.
        # Is it ever beneficial to use a core clique of size > 2?
        # Yes, if there are many core edges left.
        # But usually, the {i}U S cliques cover the core edges incident to "popular" core vertices.
        # Let's stick to {i}U S candidates for the main pool.
        # If uncovered core edges remain, we cover them with 2-cliques (fallback).
        # This might be suboptimal compared to large core cliques, but likely better than sequential greedy.
        
        # To improve, we can also add the full core clique if it exists? No, core is complete.
        # The core is a clique. So any subset is a clique.
        # So we can cover any set of core edges with one clique (the set of vertices involved).
        # But that clique must be a valid clique in the graph. Since core is complete, any subset is.
        # So, if we have remaining core edges, we can just take the set of vertices involved in them
        # and make one big clique.
        # BUT, that clique cannot include any I-vertex (as it's a core clique).
        # So, we can add a "dynamic" candidate: the set of all vertices involved in uncovered core edges.
        
        # 3. Greedy Set Cover
        uncovered = set(all_edges)
        selected_cliques = []
        
        # Pre-compute candidate coverage as sets for fast intersection?
        # Candidates might be many. k=30, n=90.
        # Maximal cliques in N_i. N_i size ~ 21.
        # Number of maximal cliques can be exponential, but usually manageable for random graphs.
        
        # Sort candidates by size descending? Or just iterate.
        # Greedy: Pick candidate with max |covered ∩ uncovered|
        
        # To make this faster, we can pre-filter candidates that cover at least 1 edge.
        # All our candidates cover at least 2 cross edges and 1 core edge (if |S|>=2).
        
        # Optimization: 
        # Instead of checking all candidates every step, we can just iterate.
        # Number of candidates M. Number of edges E.
        # E ~ n*k*p + k^2/2. For 90,30: 90*30*0.7 + 435 = 1890 + 435 = 2325 edges.
        # If M is small (e.g., < 1000), checking all M for each of E steps is 2.3M ops. Fast.
        
        # Let's implement the greedy loop.
        # We need to be careful with time.
        
        # Convert candidates to list of (clique_vertices, covered_edges_set)
        # We already have this in 'candidates'
        
        # To allow for "core-only" cliques to cover remaining core edges efficiently:
        # We will check after the main loop if any core edges remain.
        # If so, we will add cliques covering them.
        
        while uncovered:
            if time.time() > deadline:
                break
                
            best_candidate_idx = -1
            best_count = 0
            best_clique = None
            best_covered = None
            
            # Iterate through all candidates
            for idx, (clique_verts, cov_set) in enumerate(candidates):
                # Count how many edges in cov_set are in uncovered
                # cov_set is a set of tuples. uncovered is a set of tuples.
                # Intersection size
                # To speed up, we can just do len(cov_set & uncovered)
                # This is O(|cov_set|).
                # cov_set size is |S| (cross) + |S|(|S|-1)/2 (core).
                # If |S| is 5, size is 5 + 10 = 15.
                # This is very fast.
                
                count = len(cov_set & uncovered)
                if count > best_count:
                    best_count = count
                    best_candidate_idx = idx
                    best_clique = clique_verts
                    best_covered = cov_set
                    
            if best_count == 0:
                # No candidate covers any uncovered edge.
                # This happens if uncovered edges are pure core edges not covered by any N_i clique,
                # or if we ran out of candidates.
                # Fallback: Cover remaining edges with 2-cliques.
                # To be smarter, we can group remaining core edges into cliques.
                
                # Separate core and cross
                remaining_cross = [e for e in uncovered if e[0] < k and e[1] >= k]
                remaining_core = [e for e in uncovered if e[0] < k and e[1] < k]
                
                # Cover cross edges with 2-cliques
                for e in remaining_cross:
                    selected_cliques.append([e[0], e[1]])
                    uncovered.discard(e)
                
                # Cover core edges
                if remaining_core:
                    # Get all vertices involved in remaining core edges
                    verts = set()
                    for u, v in remaining_core:
                        verts.add(u)
                        verts.add(v)
                    # Since core is a clique, any subset is a clique.
                    # We can just add one big clique containing all these vertices.
                    # This covers all remaining core edges in 1 clique!
                    if len(verts) >= 2:
                        selected_cliques.append(list(verts))
                        # Remove all core edges among these verts from uncovered
                        # Actually, just clear all remaining core edges
                        for e in remaining_core:
                            uncovered.discard(e)
                break
                
            # Select best candidate
            selected_cliques.append(list(best_clique))
            uncovered.difference_update(best_covered)
            
            # Remove the candidate from the pool to avoid re-picking it?
            # Yes, a clique is used once.
            candidates.pop(best_candidate_idx)
            
        # If we broke out of while due to timeout, cover rest with 2-cliques
        if uncovered:
            for e in uncovered:
                selected_cliques.append([e[0], e[1]])
                
        return selected_cliques

    results = []
    start_time = time.time()
    deadline = start_time + 100 # Safety margin
    
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        n, k, p, seed = spec
        cliques = _solve_instance(n, k, p, seed, deadline)
        results.append(cliques)
        
    return results
