import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill

    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(k):
            for v in range(u + 1, k):
                edges.append((u, v))
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.append((u, v))
        return sorted(edges)

    def solve_split(n, k, p, seed):
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # C vertices are 0..k-1, I vertices are k..n-1
        C = list(range(k))
        I = list(range(k, n))
        
        # Uncovered edges in C
        uncovered_C_edges = set()
        for u in C:
            for v in C:
                if u < v and (u, v) in edge_set:
                    uncovered_C_edges.add((u, v))
        
        # Uncovered edges between C and I
        uncovered_CI_edges = set()
        for u in C:
            for v in I:
                if (u, v) in edge_set:
                    uncovered_CI_edges.add((u, v))
                    
        # Neighbors of each v in I within C
        N_in_C = {v: set() for v in I}
        for u in C:
            for v in I:
                if (u, v) in edge_set:
                    N_in_C[v].add(u)
        
        # Sort I vertices by degree in C (descending)
        I_sorted = sorted(I, key=lambda v: len(N_in_C[v]), reverse=True)
        
        cliques = []
        
        # Helper to find max clique in a graph defined by a set of vertices and a set of edges
        def find_max_clique(vertices, edges_available):
            if not vertices:
                return []
            
            # Build adjacency for this subgraph
            adj = {v: set() for v in vertices}
            for u, v in edges_available:
                if u in adj and v in adj:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Greedy max clique
            remaining = set(vertices)
            clique = []
            
            while remaining:
                best_v = None
                max_deg = -1
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg > max_deg:
                        max_deg = deg
                        best_v = v
                
                if best_v is None:
                    break
                
                clique.append(best_v)
                remaining.remove(best_v)
                remaining = remaining & adj[best_v]
                
            return clique

        # Phase 1: Process each v in I
        for v in I_sorted:
            if time.time() > deadline:
                break
            
            # Keep processing v until all its cross edges are covered
            while True:
                if time.time() > deadline:
                    break
                
                # Find neighbors of v that have uncovered cross edges
                N_v_uncovered = set()
                for u in N_in_C[v]:
                    if (u, v) in uncovered_CI_edges:
                        N_v_uncovered.add(u)
                
                if not N_v_uncovered:
                    break
                
                # Find max clique in N_v_uncovered using uncovered_C_edges
                K = find_max_clique(list(N_v_uncovered), uncovered_C_edges)
                
                if len(K) >= 2:
                    new_clique = K + [v]
                    cliques.append(new_clique)
                    
                    # Remove edges within K from uncovered_C_edges
                    for i in range(len(K)):
                        for j in range(i + 1, len(K)):
                            u, w = K[i], K[j]
                            if u > w:
                                u, w = w, u
                            uncovered_C_edges.discard((u, w))
                            
                    # Remove edges between K and v from uncovered_CI_edges
                    for u in K:
                        uncovered_CI_edges.discard((u, v))
                else:
                    # No clique of size >= 2 found in N_v_uncovered
                    # Pick one neighbor to cover with a 2-clique
                    # Heuristic: pick the one with the most uncovered C-edges?
                    # Or just pick the first one.
                    # Let's pick the one with the highest degree in the remaining C-graph?
                    # Actually, since we don't cover C-edges, it doesn't matter for C-edges.
                    # But it might matter for other I vertices? No, I vertices are independent.
                    # So just pick one.
                    u = next(iter(N_v_uncovered))
                    cliques.append([u, v])
                    uncovered_CI_edges.discard((u, v))
        
        # Phase 2: Cover remaining uncovered edges in C
        while uncovered_C_edges:
            if time.time() > deadline:
                break
                
            u, v = next(iter(uncovered_C_edges))
            
            # Find neighbors of u and v in the remaining graph
            # To form a clique containing u and v, we need common neighbors
            # that are connected to both u and v in the remaining graph.
            
            # Get candidates: u, v, and common neighbors
            # Neighbors of u in uncovered_C_edges
            neigh_u = set()
            neigh_v = set()
            for a, b in uncovered_C_edges:
                if a == u:
                    neigh_u.add(b)
                elif b == u:
                    neigh_u.add(a)
                if a == v:
                    neigh_v.add(b)
                elif b == v:
                    neigh_v.add(a)
            
            common_neigh = neigh_u & neigh_v
            candidates = [u, v] + list(common_neigh)
            
            K = find_max_clique(candidates, uncovered_C_edges)
            
            # Ensure u and v are in K
            # The greedy might not include them if they have low degree?
            # But they are connected, so they should be.
            # If not, force them.
            if u not in K or v not in K:
                # Find max clique in common_neigh
                K_common = find_max_clique(list(common_neigh), uncovered_C_edges)
                K = [u, v] + K_common
            
            cliques.append(K)
            
            for i in range(len(K)):
                for j in range(i + 1, len(K)):
                    a, b = K[i], K[j]
                    if a > b:
                        a, b = b, a
                    uncovered_C_edges.discard((a, b))
                    
        # Phase 3: Cover remaining uncovered edges between C and I
        while uncovered_CI_edges:
            if time.time() > deadline:
                break
                
            u, v = next(iter(uncovered_CI_edges))
            # Just cover with a 2-clique
            cliques.append([u, v])
            uncovered_CI_edges.discard((u, v))
            
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            pass
        
        cliques = solve_split(*spec)
        result.append(cliques)
        
    return result
