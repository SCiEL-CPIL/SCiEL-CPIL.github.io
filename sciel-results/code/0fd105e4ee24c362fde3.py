def construct():
    import time
    from random import Random

    deadline = time.time() + 145.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        # Build cross edges
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed, variant_seed=None, order_strategy="desc"):
        cross_edges = instance(n, k, p, seed)
        
        # N[v] is list of neighbors in C for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)
            
        # Available C-edges
        available = set()
        for u in range(k):
            for w in range(u + 1, k):
                available.add((u, w))
                
        mixed_cliques = []
        
        # Determine order of I vertices
        if variant_seed is None:
            if order_strategy == "desc":
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=True)
            else: # asc
                I_vertices = sorted(range(k, n), key=lambda x: len(N[x]), reverse=False)
            rng_var = None
        else:
            rng_var = Random(variant_seed)
            # Strategy 1: Random order
            # Strategy 2: Degree order with noise
            # Let's alternate or use seed to decide
            if variant_seed % 2 == 0:
                I_vertices = list(range(k, n))
                rng_var.shuffle(I_vertices)
            else:
                # Sort by degree, but break ties randomly? Or just sort.
                # To make it different from base, let's sort by (degree + small_noise)
                if order_strategy == "desc":
                     I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.1, reverse=True)
                else:
                     I_vertices = sorted(range(k, n), key=lambda x: len(N[x]) + rng_var.random() * 0.1, reverse=False)

        for v in I_vertices:
            if not N[v]:
                continue
                
            remaining_neighbors = set(N[v])
            
            while remaining_neighbors:
                if time.time() > deadline:
                    # Fallback: just add 2-cliques for remaining
                    for u in remaining_neighbors:
                        mixed_cliques.append([int(v), int(u)])
                    break
                
                S = []
                S_set = set()
                cand_set = set(remaining_neighbors)
                
                while cand_set:
                    # Find vertex with max degree in cand_set
                    best_v = None
                    best_deg = -1
                    candidates_for_best = []
                    for w in cand_set:
                        deg = 0
                        for x in cand_set:
                            if x == w: continue
                            e = (min(w, x), max(w, x))
                            if e in available:
                                deg += 1
                        if deg > best_deg:
                            best_deg = deg
                            best_v = w
                            candidates_for_best = [w]
                        elif deg == best_deg:
                            candidates_for_best.append(w)
                            
                    if best_v is None:
                        break
                    
                    # Tie-breaking
                    if len(candidates_for_best) > 1 and rng_var is not None:
                        best_v = rng_var.choice(candidates_for_best)
                    
                    S.append(best_v)
                    S_set.add(best_v)
                    
                    # Update cand_set: keep only neighbors of best_v in cand_set
                    new_cand = set()
                    for w in cand_set:
                        if w == best_v: continue
                        e = (min(best_v, w), max(best_v, w))
                        if e in available:
                            new_cand.add(w)
                    cand_set = new_cand
                    
                if S:
                    mixed_cliques.append([int(v)] + [int(u) for u in S])
                    # Remove edges in S from available
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            e = (min(S[i], S[j]), max(S[i], S[j]))
                            available.discard(e)
                    # Remove S from remaining_neighbors
                    remaining_neighbors.difference_update(S_set)
                else:
                    # Should not happen if remaining_neighbors not empty
                    pass

        # Now cover remaining available edges with pure cliques
        pure_cliques = []
        rem_edges = set(available)
        
        while rem_edges:
            if time.time() > deadline:
                # Fallback: add edges as 2-cliques
                for e in rem_edges:
                    pure_cliques.append([int(e[0]), int(e[1])])
                break
                
            # Find a maximal clique in the graph defined by rem_edges
            vertices = set()
            for u, w in rem_edges:
                vertices.add(u)
                vertices.add(w)
                
            # Greedy maximal clique
            S = []
            S_set = set()
            cand_set = set(vertices)
            
            while cand_set:
                # Pick vertex with max degree in cand_set (within rem_edges)
                best_v = None
                best_deg = -1
                candidates_for_best = []
                for w in cand_set:
                    deg = 0
                    for x in cand_set:
                        if x == w: continue
                        e = (min(w, x), max(w, x))
                        if e in rem_edges:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = w
                        candidates_for_best = [w]
                    elif deg == best_deg:
                        candidates_for_best.append(w)
                        
                if best_v is None:
                    break
                
                # Tie-breaking for pure cliques too
                if len(candidates_for_best) > 1 and rng_var is not None:
                    best_v = rng_var.choice(candidates_for_best)
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Update cand_set
                new_cand = set()
                for w in cand_set:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in rem_edges:
                        new_cand.add(w)
                cand_set = new_cand
                
            if S:
                pure_cliques.append([int(u) for u in S])
                # Remove edges in S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        rem_edges.discard(e)
            else:
                # If no clique found, add 2-clique
                e = next(iter(rem_edges))
                pure_cliques.append([int(e[0]), int(e[1])])
                rem_edges.discard(e)

        return mixed_cliques + pure_cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    
    for (n, k, p, seed) in specs:
        best_cliques = None
        best_count = float('inf')
        
        # Run multiple variants
        # 1. Base Desc
        # 2. Base Asc
        # 3-12. Variants
        
        if time.time() > deadline:
            # Fallback for safety
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
            continue
            
        # Try Base Desc
        cliques = solve_instance(n, k, p, seed, None, "desc")
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques
            
        # Try Base Asc
        cliques = solve_instance(n, k, p, seed, None, "asc")
        count = len(cliques)
        if count < best_count:
            best_count = count
            best_cliques = cliques

        # Try variants
        for i in range(10):
            if time.time() > deadline - 5.0: # Safety margin
                break
            v_seed = seed + 1000 + i
            # Alternate strategies in variants
            strat = "desc" if i % 2 == 0 else "asc"
            cliques = solve_instance(n, k, p, seed, v_seed, strat)
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
                
        result.append(best_cliques)

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
