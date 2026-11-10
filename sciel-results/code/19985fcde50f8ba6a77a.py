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

    def solve_instance(n, k, p, seed):
        cross_edges = instance(n, k, p, seed)
        
        # N_C[v] is set of neighbors in C for v in I
        # N_I[u] is set of neighbors in I for u in C
        N_C = {v: set() for v in range(k, n)}
        N_I = {u: set() for u in range(k)}
        for u, v in cross_edges:
            N_C[v].add(u)
            N_I[u].add(v)
            
        # Uncovered edges
        # Cross edges: set of (u, v)
        # C edges: set of (u, w)
        uncov_cross = set(cross_edges)
        uncov_C = set()
        for u in range(k):
            for w in range(u + 1, k):
                uncov_C.add((u, w))
                
        cliques = []
        
        # Greedy covering of cross edges
        while uncov_cross:
            if time.time() > deadline:
                # Fallback: add all remaining cross edges as 2-cliques and C edges as 2-cliques
                for u, v in uncov_cross:
                    cliques.append([int(u), int(v)])
                for u, w in uncov_C:
                    cliques.append([int(u), int(w)])
                break
                
            # Pick an edge (u, v) from uncov_cross
            # Heuristic: pick (u, v) such that |N_C[v]| + |N_I[u]| is large?
            # Or just pick the first one.
            u, v = next(iter(uncov_cross))
            
            # Initialize S and T
            S = {u}
            T = {v}
            
            # Expand S and T
            changed = True
            while changed:
                changed = False
                # Try to add w to S
                for w in range(k):
                    if w in S:
                        continue
                    # Check if w is connected to all t in T
                    can_add = True
                    for t in T:
                        if t not in N_I[w]:
                            can_add = False
                            break
                    
                    if can_add:
                        # Check if adding w covers any new edges
                        new_cross = 0
                        for t in T:
                            if (w, t) in uncov_cross:
                                new_cross += 1
                        new_C = 0
                        for s in S:
                            e = (min(w, s), max(w, s))
                            if e in uncov_C:
                                new_C += 1
                            
                        if new_cross > 0 or new_C > 0:
                            S.add(w)
                            changed = True
                            
                # Try to add z to T
                for z in range(k, n):
                    if z in T:
                        continue
                    can_add = True
                    has_uncovered = False
                    for w in S:
                        # Check if z is connected to w
                        if w not in N_C[z]:
                            can_add = False
                            break
                        # Check if (w, z) is uncovered
                        if (w, z) in uncov_cross:
                            has_uncovered = True
                    
                    if can_add and has_uncovered:
                        T.add(z)
                        changed = True
            
            # Create clique
            clique = list(S) + list(T)
            cliques.append(clique)
            
            # Remove covered edges
            # Cross edges between S and T
            for w in S:
                for t in T:
                    uncov_cross.discard((w, t))
            # C edges within S
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    e = (min(S[i], S[j]), max(S[i], S[j]))
                    uncov_C.discard(e)
                    
        # Cover remaining C edges
        while uncov_C:
            if time.time() > deadline:
                for u, w in uncov_C:
                    cliques.append([int(u), int(w)])
                break
                
            # Find a maximal clique in H
            verts = set()
            for u, w in uncov_C:
                verts.add(u)
                verts.add(w)
                
            S = set()
            cand = set(verts)
            
            while cand:
                # Pick vertex with max degree in cand
                best_v = None
                best_deg = -1
                for v in cand:
                    deg = 0
                    for w in cand:
                        if v == w: continue
                        e = (min(v, w), max(v, w))
                        if e in uncov_C:
                            deg += 1
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                        
                if best_v is None:
                    break
                    
                S.add(best_v)
                
                # Update cand: keep only neighbors of best_v in H
                new_cand = set()
                for w in cand:
                    if w == best_v: continue
                    e = (min(best_v, w), max(best_v, w))
                    if e in uncov_C:
                        new_cand.add(w)
                cand = new_cand
                
            if S:
                cliques.append(list(S))
                # Remove edges in S from uncov_C
                for i in range(len(S)):
                    for j in range(i + 1, len(S)):
                        e = (min(S[i], S[j]), max(S[i], S[j]))
                        uncov_C.discard(e)
            else:
                # Fallback
                u, w = next(iter(uncov_C))
                cliques.append([int(u), int(w)])
                uncov_C.discard((u, w))
                
        return cliques

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    result = []
    for (n, k, p, seed) in specs:
        if time.time() > deadline:
            # Fallback
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
