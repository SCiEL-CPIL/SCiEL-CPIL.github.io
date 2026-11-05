def construct():
    import time
    from random import Random
    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigs
    
    deadline = time.time() + 140  # 10s safety margin
    
    def generate_graph(n, p, seed):
        rng = Random(seed)
        edges = []
        adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if rng.random() < p:
                    edges.append((u, v))
                    adj[u][v] = True
                    adj[v][u] = True
        return edges, adj
    
    def find_max_clique(adj, n, start_vertices):
        """Find a maximum clique using Bron-Kerbosch with pivot, restricted to start_vertices."""
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if time.time() > deadline - 5:
                return
            if not P and not X:
                if len(R) > len(best_clique):
                    best_clique = list(R)
                return
            if not P:
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            U = list(P) + list(X)
            if not U:
                return
            # Pick vertex with maximum degree in P
            u = max(U, key=lambda v: sum(1 for w in P if adj[v][w]))
            
            # Only branch on vertices in P that are NOT adjacent to u
            candidates = [v for v in P if not adj[v][u]]
            
            for v in candidates:
                if v not in P:
                    continue
                new_R = list(R) + [v]
                new_P = [w for w in P if adj[v][w]]
                new_X = [w for w in X if adj[v][w]]
                bron_kerbosch(new_R, new_P, new_X)
                P.remove(v)
                X.append(v)
        
        # Start with all start_vertices as potential clique members
        initial_P = list(start_vertices)
        bron_kerbosch([], initial_P, [])
        return best_clique
    
    def solve_instance(edges, adj, n):
        if not edges:
            return []
        
        # Build complement graph adjacency (edges NOT in original graph)
        comp_adj = [[False]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    comp_adj[u][v] = True
                    comp_adj[v][u] = True
        
        # Find connected components in complement graph
        # Vertices in the same complement component have "structural similarity"
        visited = [False]*n
        complement_components = []
        
        for start in range(n):
            if not visited[start]:
                component = []
                stack = [start]
                visited[start] = True
                while stack:
                    v = stack.pop()
                    component.append(v)
                    for w in range(n):
                        if comp_adj[v][w] and not visited[w]:
                            visited[w] = True
                            stack.append(w)
                if len(component) > 1:
                    complement_components.append(component)
        
        # Sort components by size (larger first)
        complement_components.sort(key=len, reverse=True)
        
        # Remaining edges
        remaining = set(edges)
        cliques = []
        
        # For each complement component, try to find large cliques within it
        for comp in complement_components:
            if time.time() > deadline - 10:
                break
            comp_set = set(comp)
            
            # Find edges within this component
            comp_edges = [(u, v) for (u, v) in remaining if u in comp_set and v in comp_set]
            if not comp_edges:
                continue
            
            # Build local adjacency for this component
            local_adj = {u: set() for u in comp}
            for u, v in comp_edges:
                local_adj[u].add(v)
                local_adj[v].add(u)
            
            # Find vertices with high degree in local graph
            deg = {u: len(local_adj[u]) for u in comp}
            high_deg_vertices = sorted(comp, key=lambda v: deg[v], reverse=True)
            
            # Try to find cliques starting from high-degree vertices
            used_in_component = set()
            
            for start_v in high_deg_vertices:
                if time.time() > deadline - 10:
                    break
                if start_v in used_in_component:
                    continue
                if deg[start_v] == 0:
                    continue
                
                # Get neighbors in remaining graph
                candidates = [v for v in local_adj[start_v] if v not in used_in_component]
                if not candidates:
                    continue
                
                # Build candidate set including start
                candidate_set = set(candidates) | {start_v}
                
                # Find max clique within candidate_set
                clique = find_max_clique(adj, n, candidate_set)
                
                if len(clique) >= 2:
                    # Verify all edges in clique are in remaining
                    valid = True
                    for i in range(len(clique)):
                        for j in range(i+1, len(clique)):
                            pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                            if pair not in remaining:
                                valid = False
                                break
                        if not valid:
                            break
                    
                    if valid:
                        cliques.append(clique)
                        used_in_component.update(clique)
                        # Remove edges covered by this clique
                        for i in range(len(clique)):
                            for j in range(i+1, len(clique)):
                                pair = (min(clique[i], clique[j]), max(clique[i], clique[j]))
                                remaining.discard(pair)
                        # Update local_adj
                        for v in clique:
                            local_adj[v].clear()
                    else:
                        # Fallback: just use start_v with its best neighbor
                        best_neighbor = max(candidates, key=lambda v: deg[v])
                        pair = (min(start_v, best_neighbor), max(start_v, best_neighbor))
                        if pair in remaining:
                            cliques.append([start_v, best_neighbor])
                            remaining.discard(pair)
                            used_in_component.add(start_v)
                            used_in_component.add(best_neighbor)
                            local_adj[start_v].discard(best_neighbor)
                            local_adj[best_neighbor].discard(start_v)
                            deg[start_v] = max(0, deg[start_v] - 1)
                            deg[best_neighbor] = max(0, deg[best_neighbor] - 1)
        
        # Handle any remaining edges as 2-cliques
        for (u, v) in remaining:
            cliques.append([u, v])
        
        return cliques
    
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        edges, adj = generate_graph(n, p, seed)
        cliques = solve_instance(edges, adj, n)
        results.append(cliques)
    
    return results
