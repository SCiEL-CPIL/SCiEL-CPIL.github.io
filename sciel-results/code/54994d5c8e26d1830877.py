def construct():
    import random
    import sys

    def instance(n, p, seed):
        rng = random.Random(seed)
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u][v] = adj[v][u] = True
                    edges.append((u, v))
        return adj, edges

    def solve(n, adj, edges):
        # remaining_edges is a set of tuples (u, v) with u < v
        remaining_edges = set(tuple(sorted(e)) for e in edges)
        cliques = []
        
        # Helper to check if a set of vertices forms a clique in the remaining graph
        # and to find the maximum clique using greedy MIS on complement
        
        while remaining_edges:
            # Determine the set of vertices involved in remaining edges
            vertices = set()
            for u, v in remaining_edges:
                vertices.add(u)
                vertices.add(v)
                
            if not vertices:
                break

            # Build complement graph restricted to 'vertices'
            # We want to find a large clique in G[vertices], which is a large independent set in ~G[vertices]
            # Let's use a greedy heuristic for Max Independent Set on the complement graph
            
            # Degrees in the complement graph for each vertex in 'vertices'
            # deg_comp[v] = number of non-edges (u, v) in G where u, v in 'vertices'
            # Note: We only care about edges that are NOT in remaining_edges.
            # But wait, the graph for the next step is defined by remaining_edges.
            # A clique in the remaining graph must consist of vertices where ALL pairs are in remaining_edges.
            
            # So we are looking for a clique in the graph H where H has vertex set V_H = vertices
            # and edge set E_H = remaining_edges restricted to V_H.
            
            # Greedy approach:
            # 1. Start with candidate set C = vertices.
            # 2. While C is not empty:
            #    a. Calculate for each v in C, the number of neighbors in C within H.
            #    b. Pick v with the highest degree in H[C] (greedy for Max Clique).
            #    c. Add v to the current clique.
            #    d. Restrict C to the neighbors of v in H[C].
            
            # This is a standard greedy algorithm for finding a large clique.
            
            C = list(vertices)
            clique = []
            
            # To speed up, we can precompute adjacency lists for H
            # But H changes? No, H is fixed for this iteration (remaining_edges).
            # We just restrict the candidate set.
            
            # Let's use a simpler greedy:
            # Sort vertices by degree in H (descending).
            # Iterate through sorted vertices.
            # If vertex connects to all vertices in current_clique, add it.
            
            # Calculate degrees in H
            deg = {v: 0 for v in C}
            for u, v in remaining_edges:
                if u in deg and v in deg:
                    deg[u] += 1
                    deg[v] += 1
            
            # Sort candidates by degree descending
            C.sort(key=lambda x: deg[x], reverse=True)
            
            for v in C:
                # Check if v is connected to all vertices in 'clique' in H
                # Note: 'clique' contains vertices from C that were already added.
                # We need to check if (v, u) is in remaining_edges for all u in clique.
                can_add = True
                for u in clique:
                    pair = tuple(sorted((v, u)))
                    if pair not in remaining_edges:
                        can_add = False
                        break
                if can_add:
                    clique.append(v)
            
            # If we found a clique of size >= 2, add it
            if len(clique) >= 2:
                cliques.append(clique)
                # Remove all edges within clique from remaining_edges
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        pair = tuple(sorted((clique[i], clique[j])))
                        remaining_edges.discard(pair)
            else:
                # Should not happen if remaining_edges is not empty, 
                # but if it does, pick an arbitrary edge to avoid infinite loop
                edge = next(iter(remaining_edges))
                cliques.append(list(edge))
                remaining_edges.discard(edge)

        return cliques

    adj1, edges1 = instance(32, 0.6, 601)
    adj2, edges2 = instance(64, 0.55, 602)
    
    cliques1 = solve(32, adj1, edges1)
    cliques2 = solve(64, adj2, edges2)
    
    return [cliques1, cliques2]
