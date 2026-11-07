import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 150.0
    
    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for spec in specs:
        if time.time() > deadline - 5.0:
            # Fallback: return edges as cliques if no time
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        n, k, p, seed = spec
        edges = instance(*spec)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v))
            
        # Identify C and I
        C = set(range(k))
        I = set(range(k, n))
        
        # Initial cliques: {i} U N(i) for i in I
        # We represent cliques as frozensets of vertices
        cliques = []
        covered_edges = set()
        
        # Map edge to clique index
        edge_to_clique = {}
        
        def add_clique(vertices):
            if len(vertices) < 2:
                return
            v_set = frozenset(vertices)
            # Check if this exact clique already exists? 
            # We want distinct cliques in the list? 
            # "Every listed set must be a clique". 
            # It doesn't say distinct, but usually partition implies distinct sets of edges.
            # If two cliques are identical, they cover the same edges. 
            # If edges are covered exactly once, we can't have duplicate cliques covering same edges.
            # So cliques must be edge-disjoint.
            
            # Check validity: is it a clique?
            for u, v in combinations(v_set, 2):
                if (min(u,v), max(u,v)) not in edge_set:
                    return # Not a clique
            
            # Check edge overlap
            new_edges = set()
            for u, v in combinations(v_set, 2):
                e = (min(u,v), max(u,v))
                if e in covered_edges:
                    return # Conflict
                new_edges.add(e)
                
            if not new_edges:
                return
                
            cliques.append(list(v_set))
            for e in new_edges:
                covered_edges.add(e)
                edge_to_clique[e] = len(cliques) - 1
                
        # Step 1: Add initial large cliques
        for i in I:
            if time.time() > deadline - 10.0:
                break
            N_i = adj[i] & C
            if N_i:
                add_clique(list(N_i) + [i])
                
        # Step 2: Repair conflicts and cover missing edges
        # We need to ensure all edges are covered.
        # The add_clique function above rejects if there's a conflict.
        # So we might have rejected some initial cliques or parts of them.
        # Actually, add_clique adds the WHOLE clique or nothing.
        # If it rejects, the edges in N(i) are not covered by this clique.
        
        # Let's refine: 
        # We want to cover as much as possible with large cliques.
        # Then cover the rest with small ones.
        
        # Re-evaluate: 
        # The simple "add if valid" might leave many edges uncovered.
        # We should iterate.
        
        # Better approach:
        # 1. Try to add {i} U N(i).
        # 2. If conflict, try to remove vertices from N(i) to resolve conflict, then add.
        # 3. After all i processed, cover remaining edges with pairs.
        
        cliques = []
        covered_edges = set()
        
        def add_clique_safe(vertices):
            v_set = frozenset(vertices)
            if len(v_set) < 2:
                return False
            # Check clique property
            for u, v in combinations(v_set, 2):
                if (min(u,v), max(u,v)) not in edge_set:
                    return False
            # Check edge overlap
            for u, v in combinations(v_set, 2):
                e = (min(u,v), max(u,v))
                if e in covered_edges:
                    return False
            # Add
            cliques.append(list(v_set))
            for u, v in combinations(v_set, 2):
                e = (min(u,v), max(u,v))
                covered_edges.add(e)
            return True

        # Process I vertices
        for i in I:
            if time.time() > deadline - 5.0:
                break
            N_i = list(adj[i] & C)
            # Try to add full N_i
            if not add_clique_safe(N_i + [i]):
                # Try to reduce N_i
                # Greedy: remove vertices causing conflict
                # We need to find a subset S of N_i such that {i} U S is valid.
                # Start with S = N_i.
                S = set(N_i)
                changed = True
                while changed:
                    changed = False
                    # Find conflict
                    # Check edges in S
                    conflict_found = False
                    for u, v in combinations(list(S), 2):
                        e = (min(u,v), max(u,v))
                        if e in covered_edges:
                            # Conflict on (u,v)
                            # Remove v from S
                            S.remove(v)
                            changed = True
                            conflict_found = True
                            break
                    if conflict_found:
                        continue
                    # Also check if {i} U S is a clique (it is, since S subset C)
                    # But we must ensure no edge (i, s) is covered?
                    # Edges (i, s) are not in covered_edges yet if we haven't added any clique with i.
                    # Wait, if we added a previous clique with i? No, each i is processed once.
                    # So edges (i, s) are free.
                    # The only conflicts are internal edges of C.
                    
                if S:
                    add_clique_safe(list(S) + [i])
                else:
                    # Could not form a clique with i and any C vertex?
                    # Maybe N_i was empty or all conflicts?
                    pass

        # Cover remaining edges
        # Iterate over all edges in edge_set
        for e in edge_set:
            if time.time() > deadline - 1.0:
                break
            if e not in covered_edges:
                u, v = e
                # Add clique {u, v}
                add_clique_safe([u, v])

        # Final check: are all edges covered?
        # If not, the partition is invalid.
        # But we added cliques for all uncovered edges, so it should be valid.
        
        # Convert to required format
        # List of lists of ints
        # Ensure ints are Python ints
        final_cliques = []
        for c in cliques:
            final_cliques.append([int(x) for x in c])
            
        results.append(final_cliques)
        
    return results
