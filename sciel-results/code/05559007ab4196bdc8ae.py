"""Combined approach: Greedy set cover for cross-edges with internal edge tracking."""
import time
import sys
from itertools import combinations

def construct():
    _t0 = time.time()
    _DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_one(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Separate edges
        internal_edges = set()
        cross_edges = []
        for u, v in edges:
            if u < k and v < k:
                internal_edges.add((u, v))
            else:
                cross_edges.append((u, v))
        
        # Build adjacency for independent vertices
        # neighbors_in_k[v] = set of u in K such that (u, v) is an edge
        neighbors_in_k = {v: set() for v in range(k, n)}
        for u, v in cross_edges:
            if u < k:
                neighbors_in_k[v].add(u)
            else:
                neighbors_in_k[u].add(v)
        
        # Track used internal edges
        used_internal = set()
        
        # Track covered cross edges
        # We will construct the final list of cliques
        cliques = []
        
        # Process independent vertices
        # Sort by degree descending to prioritize large cliques
        indep_vertices = [v for v in range(k, n) if neighbors_in_k[v]]
        indep_vertices.sort(key=lambda v: len(neighbors_in_k[v]), reverse=True)
        
        # For each independent vertex, find largest subset S of N(v)
        # such that no internal edge in S is used.
        for v in indep_vertices:
            if time.time() - _t0 > _DEADLINE:
                break
                
            N_v = neighbors_in_k[v]
            if not N_v:
                continue
                
            # Build graph on N_v where edge (u, w) exists if (u, w) not in used_internal
            # Vertices are sorted for deterministic processing
            verts = sorted(N_v)
            m = len(verts)
            if m == 0:
                continue
                
            # Map vertex to index
            vert_to_idx = {vert: i for i, vert in enumerate(verts)}
            
            # Adjacency matrix for availability
            # avail[i][j] is True if edge (verts[i], verts[j]) is NOT in used_internal
            avail = [[True] * m for _ in range(m)]
            for i in range(m):
                for j in range(i + 1, m):
                    u, w = verts[i], verts[j]
                    if (u, w) in used_internal or (w, u) in used_internal:
                        avail[i][j] = False
                        avail[j][i] = False
            
            # Find largest clique in this availability graph
            # Use Bron-Kerbosch with pivot
            best_clique = []
            
            def bronk(R, P, X):
                nonlocal best_clique
                if not P and not X:
                    if len(R) > len(best_clique):
                        best_clique = R[:]
                    return
                if len(R) + len(P) <= len(best_clique):
                    return
                
                # Pivot selection: choose v in P U X with max degree in P
                pivot = None
                max_deg = -1
                for v in P:
                    deg = sum(1 for u in P if avail[v][u])
                    if deg > max_deg:
                        max_deg = deg
                        pivot = v
                if pivot is None:
                    for v in X:
                        deg = sum(1 for u in P if avail[v][u])
                        if deg > max_deg:
                            max_deg = deg
                            pivot = v
                
                if pivot is None:
                    # No candidates, but P is not empty? Should not happen if P non-empty
                    candidates = list(P)
                else:
                    # Candidates are vertices in P not adjacent to pivot
                    candidates = [v for v in P if not avail[v][pivot]]
                    
                for u in candidates:
                    P_new = [v for v in P if avail[u][v]]
                    X_new = [v for v in X if avail[u][v]]
                    R.append(u)
                    bronk(R, P_new, X_new)
                    R.pop()
                    P.remove(u)
                    X.append(u)
                    
            # Initial call
            P_init = list(range(m))
            bronk([], P_init, [])
            
            if not best_clique:
                # If no clique found (should not happen if m>0, at least size 1)
                # But Bron-Kerbosch might fail if logic is slightly off.
                # Fallback: take single largest available vertex?
                # Actually, if m>0, there is at least one vertex.
                # Let's just take the first vertex if best_clique is empty.
                if m > 0:
                    best_clique = [0]
            
            S_indices = best_clique
            S_verts = [verts[i] for i in S_indices]
            
            if len(S_verts) >= 1:
                # Form clique S_verts + [v]
                clique = S_verts + [v]
                cliques.append(clique)
                
                # Mark internal edges in S_verts as used
                for i in range(len(S_verts)):
                    for j in range(i + 1, len(S_verts)):
                        u, w = S_verts[i], S_verts[j]
                        used_internal.add((u, w))
            
            # Note: Cross edges covered are (u, v) for u in S_verts.
            # We don't explicitly track covered cross edges yet, 
            # but we know that for this v, edges to S_verts are covered.
            # Edges to N_v \ S_verts are NOT covered yet.

        # Now handle remaining cross edges
        # A cross edge (u, v) is covered if v was processed and u was in S_v.
        # We need to identify which cross edges are NOT covered.
        
        # Re-iterate to check
        # We can reconstruct which (u, v) pairs were covered.
        covered_cross = set()
        for c in cliques:
            # c is a list of vertices
            # Identify the independent vertex (>= k)
            indep_v = None
            clique_verts = []
            for vert in c:
                if vert >= k:
                    indep_v = vert
                else:
                    clique_verts.append(vert)
            
            if indep_v is not None:
                for u in clique_verts:
                    covered_cross.add((u, indep_v))
        
        # Find uncovered cross edges
        for u, v in cross_edges:
            if (u, v) not in covered_cross:
                # Add as 2-clique
                cliques.append([u, v])
        
        # Now handle remaining internal edges
        # Remaining internal edges are those in internal_edges but not in used_internal
        remaining_internal = internal_edges - used_internal
        
        if remaining_internal:
            # We need to cover these edges with cliques from K only.
            # Greedy: find largest clique in the graph of remaining_internal
            # Vertices involved are those in K that have remaining edges.
            
            # Build adjacency for remaining internal graph
            # Vertices are 0..k-1
            rem_adj = {i: set() for i in range(k)}
            for u, v in remaining_internal:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Get active vertices
            active_verts = [i for i in range(k) if rem_adj[i]]
            
            while remaining_internal:
                if time.time() - _t0 > _DEADLINE:
                    # Fallback: add remaining edges as pairs
                    for u, v in remaining_internal:
                        cliques.append([u, v])
                    break
                
                # Find largest clique in the subgraph induced by active_verts
                # using rem_adj
                best_c = []
                
                def bronk_rem(R, P, X):
                    nonlocal best_c
                    if not P and not X:
                        if len(R) > len(best_c):
                            best_c = R[:]
                        return
                    if len(R) + len(P) <= len(best_c):
                        return
                    
                    pivot = None
                    max_deg = -1
                    for v in P:
                        deg = len(rem_adj[v] & set(P))
                        if deg > max_deg:
                            max_deg = deg
                            pivot = v
                    if pivot is None:
                        for v in X:
                            deg = len(rem_adj[v] & set(P))
                            if deg > max_deg:
                                max_deg = deg
                                pivot = v
                    
                    if pivot is None:
                        candidates = list(P)
                    else:
                        candidates = [v for v in P if v not in rem_adj[pivot]]
                    
                    for u in candidates:
                        P_new = [v for v in P if v in rem_adj[u]]
                        X_new = [v for v in X if v in rem_adj[u]]
                        R.append(u)
                        bronk_rem(R, P_new, X_new)
                        R.pop()
                        P.remove(u)
                        X.append(u)
                        
                P_init = list(active_verts)
                bronk_rem([], P_init, [])
                
                if not best_c:
                    # Fallback: take any edge
                    u, v = next(iter(remaining_internal))
                    best_c = [u, v]
                else:
                    # Ensure best_c is a clique in rem_adj
                    # Bron-Kerbosch guarantees this.
                    pass
                
                if len(best_c) < 2:
                    # Should not happen if remaining_internal is not empty
                    break
                
                cliques.append(best_c)
                
                # Remove edges covered by best_c from remaining_internal
                for i in range(len(best_c)):
                    for j in range(i + 1, len(best_c)):
                        u, v = best_c[i], best_c[j]
                        if (u, v) in remaining_internal:
                            remaining_internal.remove((u, v))
                            rem_adj[u].remove(v)
                            rem_adj[v].remove(u)
                            if not rem_adj[u]:
                                active_verts.remove(u)
                            if not rem_adj[v]:
                                active_verts.remove(v)
                # Update active_verts list to only include those with edges
                active_verts = [i for i in range(k) if rem_adj[i]]
            
            # If we broke out of loop due to deadline, remaining_internal might not be empty
            # But the fallback inside the loop handles it.
            # If we finished the loop normally, remaining_internal should be empty.
            # However, if active_verts becomes empty but remaining_internal is not, we have a bug.
            # Let's add a safety check.
            if remaining_internal:
                for u, v in remaining_internal:
                    cliques.append([u, v])

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() - _t0 > _DEADLINE:
            # Fallback for remaining
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for (u, v) in edges])
        else:
            results.append(solve_one(*spec))

    return results
