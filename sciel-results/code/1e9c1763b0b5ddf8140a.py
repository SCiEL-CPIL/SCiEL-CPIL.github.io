"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from itertools import combinations

def construct():
    from random import Random
    
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        edges += cross_edges
        return edges

    def solve_split(n, k, p, seed):
        rng = Random(seed)
        
        # Determine neighbors in C for each vertex in I
        neighbors = {}
        for v in range(k, n):
            nbrs = set()
            for u in range(k):
                if rng.random() < p:
                    nbrs.add(u)
            neighbors[v] = nbrs
        
        cliques = []
        
        # 1. Create cliques for I-vertices
        # For each v in I with neighbors, create clique {v} U N(v)
        # This covers all C-I edges incident to v, and all C-C edges within N(v)
        covered_cc_edges = set()
        
        for v in range(k, n):
            nbrs = neighbors[v]
            if not nbrs:
                continue
            
            clique = [v] + sorted(nbrs)
            cliques.append(clique)
            
            # Mark C-C edges within N(v) as covered
            nbr_list = list(nbrs)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    u1, u2 = nbr_list[i], nbr_list[j]
                    if u1 > u2:
                        u1, u2 = u2, u1
                    covered_cc_edges.add((u1, u2))
        
        # 2. Determine remaining C-C edges
        # All C-C edges are (u, v) for 0 <= u < v < k
        remaining_cc_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                if (u, v) not in covered_cc_edges:
                    remaining_cc_edges.add((u, v))
        
        # 3. Partition remaining C-C edges into cliques
        # Graph G_rem on vertices 0..k-1 with edges remaining_cc_edges
        # We want to cover all edges in G_rem with minimum number of cliques.
        
        if not remaining_cc_edges:
            pass
        else:
            # Build adjacency for G_rem
            adj = [set() for _ in range(k)]
            for u, v in remaining_cc_edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Greedy clique cover
            # Repeatedly find a maximal clique, remove its edges
            # To be better, we can try to find a clique that covers the most edges
            # Given k is small, we can enumerate all cliques? No, 2^30 is too big.
            # But we can use a heuristic.
            
            remaining_edges = set(remaining_cc_edges)
            
            # Heuristic: Pick vertex with max degree, find maximal clique containing it
            # Or simply: iterate through vertices, try to extend a clique
            # Let's try a simple greedy:
            # 1. Find a vertex with max degree.
            # 2. Find a maximal clique containing it.
            # 3. Remove edges of that clique.
            
            while remaining_edges:
                if time.time() > deadline:
                    break
                
                # Find max degree vertex in remaining graph
                deg = [0] * k
                for u, v in remaining_edges:
                    deg[u] += 1
                    deg[v] += 1
                
                # Pick vertex with max degree
                max_deg = -1
                start_v = -1
                for u in range(k):
                    if deg[u] > max_deg:
                        max_deg = deg[u]
                        start_v = u
                
                if start_v == -1:
                    break
                
                # Find a maximal clique containing start_v
                # Use Bron-Kerbosch or simple expansion
                # Simple expansion:
                # Candidates: neighbors of start_v in remaining graph
                # Intersect with neighbors of current clique
                
                clique = [start_v]
                candidates = set(adj[start_v])
                # Ensure candidates are valid in remaining graph
                # Actually, adj is static for G_rem. We need to check if edges exist in remaining_edges.
                
                # Let's refine:
                # Current clique C_clique
                # Common neighbors of all v in C_clique in G_rem
                common = set(adj[v] for v in clique)
                # Intersect all
                for v in clique:
                    common &= adj[v]
                
                # We want to extend the clique with vertices from common
                # To make it maximal, we should add as many as possible?
                # No, we want a large clique to cover many edges.
                # A maximal clique in the subgraph induced by common?
                # Let's just pick one vertex from common to extend?
                # Better: Find a maximal clique in the subgraph induced by common.
                # Since k is small, we can try to find a large clique.
                
                # Simple greedy extension:
                # While there is a vertex in common that is connected to all in clique, add it?
                # This just finds a maximal clique.
                
                # Let's try to find the largest clique in the subgraph induced by common.
                # This is NP-hard, but k is small.
                # Let's just use a heuristic: sort common by degree in remaining graph, add greedily.
                
                # Actually, just adding any vertex from common that is connected to all current clique members
                # will keep it a clique.
                # To maximize size, we should pick vertices that are connected to many others in common.
                
                new_clique = list(clique)
                available = list(common)
                # Sort by degree in remaining graph
                available.sort(key=lambda x: deg[x], reverse=True)
                
                for v in available:
                    if all((min(v, u), max(v, u)) in remaining_edges for u in new_clique):
                        new_clique.append(v)
                
                if len(new_clique) > 1:
                    cliques.append(new_clique)
                    # Remove edges of new_clique from remaining_edges
                    for i in range(len(new_clique)):
                        for j in range(i + 1, len(new_clique)):
                            u1, u2 = new_clique[i], new_clique[j]
                            e = (min(u1, u2), max(u1, u2))
                            if e in remaining_edges:
                                remaining_edges.remove(e)
                else:
                    # If we couldn't extend, just take the vertex?
                    # But a clique must have at least 2 vertices.
                    # If start_v has no neighbors in remaining_edges, it's an isolated vertex in G_rem.
                    # But G_rem edges are only those not covered.
                    # If deg[start_v] > 0, it must have neighbors.
                    # Why would new_clique be size 1?
                    # Because common was empty?
                    # adj[start_v] contains neighbors in G_rem.
                    # If deg[start_v] > 0, adj[start_v] is not empty.
                    # So common should not be empty?
                    # Wait, common is intersection of adj[v] for v in clique.
                    # For clique=[start_v], common = adj[start_v].
                    # So if deg[start_v] > 0, common is not empty.
                    # So new_clique should be at least size 2.
                    pass

        # Convert to lists of ints
        final_cliques = []
        for c in cliques:
            final_cliques.append(list(c))
            
        return final_cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        result.append(solve_split(*spec))
    
    return result
# EVOLVE-BLOCK-END
