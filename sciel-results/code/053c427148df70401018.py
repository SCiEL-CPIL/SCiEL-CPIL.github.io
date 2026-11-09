"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random
    import time
    from itertools import combinations

    start_time = time.time()
    deadline = 115.0 # 120s limit, keep margin

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges_cc = []
        for u in range(k):
            for v in range(u + 1, k):
                edges_cc.append((u, v))
        
        edges_ci = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges_ci.append((u, v))
                    
        return edges_cc, edges_ci

    def solve_split_graph(n, k, edges_cc, edges_ci):
        cliques = []
        
        # 1. Compute N(v) for v in I
        N = {}
        for v in range(k, n):
            N[v] = set(u for u, v_edge in edges_ci if v_edge == v)
        
        # 2. Initialize S_v = N(v)
        S = {v: set(neigh) for v, neigh in N.items()}
        
        # 3. Resolve conflicts: ensure |S_v intersection S_w| <= 1 for all v != w
        # Iterate until stable or time runs out
        # To be efficient, only check pairs that might conflict?
        # For n=90, I has 60 vertices. 60*60/2 = 1800 pairs.
        # Checking intersection size is fast.
        
        independent_vertices = list(range(k, n))
        
        # Optimization: If N(v) is empty, S(v) is empty. No clique generated.
        
        # Conflict resolution loop
        # We want to remove vertices from S sets to reduce intersections.
        # Heuristic: Remove from the set that has more conflicts or is larger?
        # Let's just pick a conflicting pair and remove a common vertex from the larger set.
        
        changed = True
        while changed and (time.time() - start_time) < deadline - 10:
            changed = False
            # Find a conflicting pair
            # We can randomize the order to escape local minima? 
            # Deterministic is safer for reproducibility if needed, but random is fine.
            # Let's iterate through all pairs to find the first conflict.
            for i in range(len(independent_vertices)):
                v1 = independent_vertices[i]
                if not S[v1]: continue
                for j in range(i + 1, len(independent_vertices)):
                    v2 = independent_vertices[j]
                    if not S[v2]: continue
                    
                    inter = S[v1] & S[v2]
                    if len(inter) >= 2:
                        # Conflict found
                        # Remove a vertex from inter from the larger set
                        # Or just remove one
                        to_remove = next(iter(inter))
                        
                        if len(S[v1]) >= len(S[v2]):
                            S[v1].remove(to_remove)
                        else:
                            S[v2].remove(to_remove)
                        
                        changed = True
                        # Break inner and outer loops to restart check?
                        # Or just continue? 
                        # Restarting is safer to ensure we don't miss new conflicts created?
                        # Removing a vertex doesn't create new conflicts, only reduces them.
                        # So we can just continue.
                        break
                if changed:
                    break
            
        # 4. Generate cross cliques
        for v in independent_vertices:
            if S[v]:
                cliques.append(list(S[v]) + [v])
        
        # 5. Identify covered internal edges
        covered_edges = set()
        for v in independent_vertices:
            s_v = S[v]
            if len(s_v) >= 2:
                for u1, u2 in combinations(s_v, 2):
                    covered_edges.add((min(u1, u2), max(u1, u2)))
        
        # 6. Identify remaining internal edges
        remaining_edges = set()
        for u, w in edges_cc:
            edge = (min(u, w), max(u, w))
            if edge not in covered_edges:
                remaining_edges.add(edge)
        
        # 7. Partition remaining internal edges into cliques
        # Graph G_rem on vertices 0..k-1 with edges in remaining_edges
        # Greedy Max Clique
        while remaining_edges:
            # Find a maximal clique in G_rem
            # Simple greedy: pick a vertex, expand to neighbors
            # Or just pick the edge with highest degree?
            # Let's use a simple heuristic:
            # Pick any edge (u, w). 
            # Find the largest clique containing (u, w).
            # We can do this by BFS/DFS or greedy intersection.
            
            if not remaining_edges: break
            
            # Pick an arbitrary edge to start the clique
            u, w = next(iter(remaining_edges))
            
            # Initialize clique candidates as neighbors of u and w
            # Clique must contain u and w.
            # Other vertices must be connected to both u and w in G_rem.
            
            # Get neighbors of u in G_rem
            neigh_u = set(v for e in remaining_edges if e[0] == u for v in [e[1]] + ([e[0]] if e[0] != u else [])) # This is messy
            # Better: build adjacency list for G_rem
            
        # Re-implementing G_rem adjacency for efficiency
        adj = {u: set() for u in range(k)}
        for u, w in remaining_edges:
            adj[u].add(w)
            adj[w].add(u)
            
        while remaining_edges:
            # Find max clique
            # Greedy:
            # 1. Pick vertex with max degree in remaining graph
            # 2. Form clique from its neighbors?
            # Actually, let's just find ANY clique. 
            # A single edge is a clique of size 2.
            # To minimize count, we want large cliques.
            
            # Heuristic: 
            # 1. Pick edge (u, v) with max (deg(u) + deg(v))
            # 2. Clique = {u, v}
            # 3. Try to add other vertices x connected to both u and v.
            
            if not remaining_edges: break
            u, w = next(iter(remaining_edges))
            
            # Common neighbors of u and w in G_rem
            common = adj[u] & adj[w]
            
            # Greedily build clique
            clique = {u, w}
            # We can add vertices from common one by one if they are connected to all in clique
            # Since common are connected to u and w, we just need them to be connected to each other?
            # No, we need them to be connected to ALL vertices in the clique.
            
            # Simple approach: Just take {u, w}.
            # This guarantees progress.
            # To be better, try to expand.
            # Let's try to find a maximal clique containing u and w.
            
            # Sort common neighbors by degree in subgraph induced by common?
            # For now, just use {u, w} if expansion is complex.
            # Given k=30, even size 2 cliques are okay if we have many.
            # But we want to minimize count.
            
            # Let's try a simple expansion:
            # Iterate through common neighbors, add if connected to all in clique.
            for x in list(common):
                if all(x in adj[y] for y in clique):
                    clique.add(x)
            
            # Remove edges of this clique from remaining_edges
            clique_edges = set()
            for e1, e2 in combinations(clique, 2):
                edge = (min(e1, e2), max(e1, e2))
                if edge in remaining_edges:
                    clique_edges.add(edge)
            
            remaining_edges -= clique_edges
            cliques.append(list(clique))
            
            # Update adj? 
            # If we don't update adj, we might pick vertices not in remaining_edges.
            # But we check `if edge in remaining_edges` when removing.
            # The clique construction used `adj` which is static.
            # This might create a clique with edges that were already removed?
            # No, we only remove edges in `clique_edges` which are in `remaining_edges`.
            # The clique is valid in the original graph.
            # Are the edges in the clique all present in `remaining_edges` at the start of the iteration?
            # Yes, because we picked u, w from remaining_edges, and x from common (neighbors in adj).
            # But adj is static. It might contain edges not in remaining_edges.
            # So the clique might contain an edge (x, y) that is NOT in remaining_edges.
            # That's fine, we just don't remove it.
            # But the clique must be a clique in the graph. It is.
            # The constraint is that every edge in the clique must be in remaining_edges to be covered by it?
            # No, the partition requires that the edges of the clique are exactly the edges we assign to it.
            # If the clique contains an edge already covered, it's invalid.
            # So we MUST ensure all edges in the clique are in remaining_edges.
            
            # Fix: Verify clique edges.
            valid_clique_edges = set()
            for e1, e2 in combinations(clique, 2):
                edge = (min(e1, e2), max(e1, e2))
                if edge in remaining_edges:
                    valid_clique_edges.add(edge)
                
            # If the clique is not a clique in the REMAINING graph, we can't use it as is.
            # But we constructed it using `adj` (static).
            # We should construct it using the dynamic remaining graph.
            
            # Let's restart the clique finding with a dynamic adjacency check.
            
        # The above loop was flawed. Let's implement a clean Max Clique on dynamic graph.
        
        # Reset remaining_edges if the flawed loop modified it? 
        # It modified it. We should have saved a copy or done it right.
        
        # Let's re-do step 7 correctly.
        
        # Re-initialize remaining_edges and adj for step 7 to be safe, 
        # or just trust that the flawed loop didn't run if I comment it out?
        # I will write the correct version now.
        
        return cliques

    # Correct Implementation of Step 7
    def partition_remaining(edges, k):
        cliques = []
        if not edges:
            return cliques
            
        adj = {u: set() for u in range(k)}
        for u, w in edges:
            adj[u].add(w)
            adj[w].add(u)
            
        rem_edges = set(edges)
        
        while rem_edges:
            # Pick a vertex with max degree
            # Or just pick any edge
            if not rem_edges: break
            u, w = next(iter(rem_edges))
            
            # Build clique greedily
            clique = {u, w}
            
            # Candidates: common neighbors of u and w in CURRENT rem graph
            # We need to check connectivity in rem_edges
            
            # Simple greedy: 
            # Try to add any vertex x such that (x, y) in rem_edges for all y in clique
            # To speed up, iterate through neighbors of u
            candidates = list(adj[u])
            # Sort by degree?
            # Just iterate
            for x in candidates:
                if x in clique: continue
                # Check if x is connected to all in clique in rem_edges
                if all((min(x, y), max(x, y)) in rem_edges for y in clique):
                    clique.add(x)
                    
            # Add clique
            cliques.append(list(clique))
            
            # Remove edges of clique
            for e1, e2 in combinations(clique, 2):
                edge = (min(e1, e2), max(e1, e2))
                if edge in rem_edges:
                    rem_edges.remove(edge)
                    
            # Update adj? Not strictly necessary if we check rem_edges, 
            # but candidates list becomes stale.
            # Rebuilding adj is O(E). E is small (<= 435).
            # Let's rebuild adj for correctness if we rely on it.
            # But the loop `for x in candidates` uses stale adj.
            # This might miss vertices that are connected in rem but not in stale adj?
            # No, adj is superset. So candidates are superset of true candidates.
            # We filter by `in rem_edges`. So it's correct.
            # We might just check more vertices than necessary.
            
        return cliques

    # Main logic
    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback
            n, k, p, seed = spec
            edges_cc, edges_ci = instance(n, k, p, seed)
            cliques = [list(range(k))]
            for u, v in edges_ci:
                cliques.append([u, v])
            results.append(cliques)
            continue
            
        n, k, p, seed = spec
        edges_cc, edges_ci = instance(n, k, p, seed)
        
        # Solve
        cliques = []
        
        # 1. N(v)
        N = {}
        for v in range(k, n):
            N[v] = set(u for u, v_edge in edges_ci if v_edge == v)
            
        # 2. S_v
        S = {v: set(neigh) for v, neigh in N.items()}
        independent_vertices = list(range(k, n))
        
        # 3. Resolve conflicts
        changed = True
        while changed and (time.time() - start_time) < deadline - 10:
            changed = False
            for i in range(len(independent_vertices)):
                v1 = independent_vertices[i]
                if not S[v1]: continue
                for j in range(i + 1, len(independent_vertices)):
                    v2 = independent_vertices[j]
                    if not S[v2]: continue
                    inter = S[v1] & S[v2]
                    if len(inter) >= 2:
                        to_remove = next(iter(inter))
                        if len(S[v1]) >= len(S[v2]):
                            S[v1].remove(to_remove)
                        else:
                            S[v2].remove(to_remove)
                        changed = True
                        break
                if changed:
                    break
        
        # 4. Cross cliques
        for v in independent_vertices:
            if S[v]:
                cliques.append(list(S[v]) + [v])
                
        # 5. Covered edges
        covered_edges = set()
        for v in independent_vertices:
            s_v = S[v]
            if len(s_v) >= 2:
                for u1, u2 in combinations(s_v, 2):
                    covered_edges.add((min(u1, u2), max(u1, u2)))
                    
        # 6. Remaining edges
        remaining_edges = set()
        for u, w in edges_cc:
            edge = (min(u, w), max(u, w))
            if edge not in covered_edges:
                remaining_edges.add(edge)
                
        # 7. Partition remaining
        cliques.extend(partition_remaining(remaining_edges, k))
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
