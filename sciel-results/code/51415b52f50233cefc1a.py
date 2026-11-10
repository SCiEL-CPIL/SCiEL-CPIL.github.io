def construct():
    import time
    from random import Random

    t0 = time.time()
    DEADLINE = 142.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return cross_edges

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)

        # Build neighborhood: N[v] = list of clique-part neighbors for v in I
        N = {v: [] for v in range(k, n)}
        for u, v in cross_edges:
            N[v].append(u)

        # Set of all clique-part edges
        clique_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                clique_edges.add((u, w))

        # Helper: check if two clique-part vertices are connected (they always are in C)
        # Actually in a split graph, all clique-part vertices form a clique, so any pair is an edge.
        # But we need to track which edges are already covered.

        def edge_key(a, b):
            return (min(a, b), max(a, b))

        def greedy_max_clique_on_clique_part(available_edges, vertices, rng):
            """
            Find a large clique within the clique-part vertices, using only available edges.
            Since all clique-part vertices form a clique in the original graph,
            the constraint is only which edges are still available.
            We want a set S of clique-part vertices such that all pairs in S are in available_edges.
            This is a clique problem on the graph defined by available_edges restricted to vertices.
            """
            if not vertices:
                return []
            
            # Greedy: pick vertex with max degree in available graph, then intersect
            verts = list(vertices)
            # Sort by degree in available_edges
            def deg_in_avail(v):
                cnt = 0
                for w in verts:
                    if w == v:
                        continue
                    if edge_key(v, w) in available_edges:
                        cnt += 1
                return cnt
            
            verts.sort(key=lambda x: -deg_in_avail(x))
            
            S = []
            S_set = set()
            cand = list(verts)
            
            while cand:
                # Pick best candidate
                best_v = None
                best_score = -1
                for v in cand:
                    # Score: how many candidates are adjacent to v in available_edges
                    score = 0
                    for w in cand:
                        if w == v:
                            continue
                        if edge_key(v, w) in available_edges:
                            score += 1
                    # Tie-break: prefer vertices with more total available edges
                    if score > best_score or (score == best_score and best_v is not None and deg_in_avail(v) > deg_in_avail(best_v)):
                        best_score = score
                        best_v = v
                
                if best_v is None:
                    break
                
                S.append(best_v)
                S_set.add(best_v)
                
                # Filter candidates: must be adjacent to best_v in available_edges
                new_cand = []
                for w in cand:
                    if w == best_v:
                        continue
                    if edge_key(best_v, w) in available_edges:
                        new_cand.append(w)
                cand = new_cand
            
            return S

        # Main algorithm:
        # 1. Start with the big clique C = {0, ..., k-1}
        # 2. For each independent vertex v, create cliques that cover its cross edges
        #    by grouping its neighbors into cliques that share clique-part edges
        # 3. Track which clique-part edges are still "free" (not yet used in a cross clique)
        # 4. After processing all I vertices, cover remaining free clique-part edges

        def run_algorithm(order_strategy, rng_seed=None):
            """
            Run the split-graph-aware algorithm.
            order_strategy: 'desc' (process high-degree I vertices first), 'asc', 'random'
            """
            available = set(clique_edges)  # all clique-part edges start as available
            cliques = []
            
            # The big clique C
            C = list(range(k))
            cliques.append(C)
            
            # Determine order of I vertices
            I_vertices = list(range(k, n))
            if rng_seed is not None:
                rng = Random(rng_seed)
            else:
                rng = Random(seed)
            
            if order_strategy == 'desc':
                I_vertices.sort(key=lambda x: -len(N[x]))
            elif order_strategy == 'asc':
                I_vertices.sort(key=lambda x: len(N[x]))
            else:  # random
                rng.shuffle(I_vertices)
            
            for v in I_vertices:
                if not N[v]:
                    continue
                
                neighbors = list(N[v])
                # We need to partition neighbors into groups where each group
                # forms a clique with v. Since all neighbors are in C and C is a clique,
                # any subset of neighbors forms a clique with v.
                # The constraint is: the clique-part edges within the group must be available.
                # We want to maximize the size of each group to minimize the number of cliques.
                
                remaining_neighbors = set(neighbors)
                
                while remaining_neighbors:
                    if time.time() > DEADLINE:
                        # Fallback: add 2-cliques for remaining
                        for u in remaining_neighbors:
                            cliques.append([v, u])
                        break
                    
                    # Find a large subset S of remaining_neighbors such that
                    # all pairs in S have available edges, and S + {v} is a clique.
                    # Since v is in I and all u in S are in C, v-u edges exist by construction.
                    # We just need all u-w edges in S to be in available.
                    
                    S = greedy_max_clique_on_clique_part(available, remaining_neighbors, rng)
                    
                    if not S:
                        # Shouldn't happen if remaining_neighbors is non-empty
                        # because each individual vertex is a clique of size 1
                        u = next(iter(remaining_neighbors))
                        S = [u]
                    
                    # Add clique [v] + S
                    cliques.append([v] + S)
                    
                    # Mark clique-part edges within S as used
                    for i in range(len(S)):
                        for j in range(i + 1, len(S)):
                            e = edge_key(S[i], S[j])
                            available.discard(e)
                    
                    # Remove S from remaining_neighbors
                    remaining_neighbors.difference_update(S)
            
            # Now cover remaining available clique-part edges
            rem_edges = set(available)
            while rem_edges:
                if time.time() > DEADLINE:
                    for e in rem_edges:
                        cliques.append([e[0], e[1]])
                    break
                
                # Find vertices involved in remaining edges
                verts = set()
                for u, w in rem_edges:
                    verts.add(u)
                    verts.add(w)
                
                S = greedy_max_clique_on_clique_part(rem_edges, verts, rng)
                
                if not S:
                    e = next(iter(rem_edges))
                    S = [e[0], e[1]]
                
                cliques.append(S)
                
                # Remove edges within S from rem_edges
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        rem_edges.discard(edge_key(S[i], S[j]))
            
            return cliques

        # Try multiple variants and pick the best
        best_cliques = None
        best_count = 10**9
        
        variants = [
            ('desc', None),
            ('asc', None),
            ('random', seed),
            ('random', seed + 1),
            ('random', seed + 2),
        ]
        
        for strategy, rseed in variants:
            if time.time() > DEADLINE:
                break
            cliques = run_algorithm(strategy, rseed)
            if len(cliques) < best_count:
                best_count = len(cliques)
                best_cliques = [list(c) for c in cliques]
        
        if best_cliques is None:
            # Fallback: big clique + 2-cliques for each cross edge
            best_cliques = [list(range(k))]
            for u, v in cross_edges:
                best_cliques.append([u, v])
            best_count = len(best_cliques)
        
        # Local search: try to merge or reduce cliques
        # Build adjacency for validation
        adj = [[False] * n for _ in range(n)]
        for u in range(k):
            for w in range(u + 1, k):
                adj[u][w] = adj[w][u] = True
        for u, v in cross_edges:
            adj[u][v] = adj[v][u] = True
        
        def is_clique(S):
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if not adj[S[i]][S[j]]:
                        return False
            return True
        
        # Local search: try to move vertices between cliques to enable deletion
        cov = [[0] * n for _ in range(n)]
        for S in best_cliques:
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] += 1
                    cov[S[j]][S[i]] += 1
        
        def remove_clique(idx):
            S = best_cliques[idx]
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] -= 1
                    cov[S[j]][S[i]] -= 1
            best_cliques.pop(idx)
        
        def add_clique(S):
            best_cliques.append(S)
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] += 1
                    cov[S[j]][S[i]] += 1
        
        def is_redundant(idx):
            S = best_cliques[idx]
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    if cov[S[i]][S[j]] <= 1:
                        return False
            return True
        
        # Delete redundant cliques
        changed = True
        while changed:
            changed = False
            for idx in range(len(best_cliques) - 1, -1, -1):
                if len(best_cliques[idx]) < 2:
                    remove_clique(idx)
                    changed = True
                elif is_redundant(idx):
                    remove_clique(idx)
                    changed = True
        
        # Try local moves
        rng_ls = Random(seed * 31 + 7)
        t_start = time.time()
        while time.time() - t_start < 3.0 and time.time() < DEADLINE - 5:
            # Pick a random clique with >= 3 vertices
            sources = [i for i, S in enumerate(best_cliques) if len(S) >= 3]
            if not sources:
                break
            idx = rng_ls.choice(sources)
            S = best_cliques[idx]
            v = rng_ls.choice(S)
            
            # Find target clique where v can be added
            best_target = None
            best_gain = 0
            for j, T in enumerate(best_cliques):
                if j == idx or len(T) < 2:
                    continue
                if all(adj[v][t] for t in T):
                    gain = len(T)
                    if len(S) - 1 < 2:
                        gain += 1000
                    if gain > best_gain:
                        best_gain = gain
                        best_target = j
            
            if best_target is None:
                continue
            
            T = best_cliques[best_target]
            old_S = list(S)
            old_T = list(T)
            
            S.remove(v)
            T.append(v)
            
            # Update coverage
            for i in range(len(old_S)):
                for j in range(i + 1, len(old_S)):
                    cov[old_S[i]][old_S[j]] -= 1
                    cov[old_S[j]][old_S[i]] -= 1
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] += 1
                    cov[S[j]][S[i]] += 1
            for i in range(len(old_T)):
                for j in range(i + 1, len(old_T)):
                    cov[old_T[i]][old_T[j]] -= 1
                    cov[old_T[j]][old_T[i]] -= 1
            for i in range(len(T)):
                for j in range(i + 1, len(T)):
                    cov[T[i]][T[j]] += 1
                    cov[T[j]][T[i]] += 1
            
            # Delete redundant
            for k2 in range(len(best_cliques) - 1, -1, -1):
                if len(best_cliques[k2]) < 2:
                    remove_clique(k2)
                elif is_redundant(k2):
                    remove_clique(k2)
            
            if len(best_cliques) >= best_count:
                # Revert
                best_cliques[best_cliques.index(T)] = old_T
                # Actually this is getting complicated; just accept and track best
                pass
        
        # Final validation and repair
        # Rebuild coverage
        cov = [[0] * n for _ in range(n)]
        for S in best_cliques:
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] += 1
                    cov[S[j]][S[i]] += 1
        
        # Check all edges are covered
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] and cov[u][v] < 1:
                    missing.append((u, v))
        
        for u, v in missing:
            best_cliques.append([u, v])
        
        # Validate each clique
        cleaned = []
        for S in best_cliques:
            S = list(S)
            if len(S) < 2:
                continue
            if is_clique(S):
                cleaned.append(S)
            else:
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        if adj[S[i]][S[j]]:
                            cleaned.append([S[i], S[j]])
        
        # Final coverage check
        cov = [[0] * n for _ in range(n)]
        for S in cleaned:
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    cov[S[i]][S[j]] += 1
                    cov[S[j]][S[i]] += 1
        
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] and cov[u][v] < 1:
                    cleaned.append([u, v])
        
        return cleaned

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > DEADLINE:
            ce = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for (u, v) in ce:
                cliques.append([int(u), int(v)])
            result.append(cliques)
        else:
            result.append(solve_instance(n, k, p, seed))

    # Ensure types
    final = []
    for partition in result:
        clean = []
        for clique in partition:
            clean.append([int(x) for x in clique])
        final.append(clean)
    return final
