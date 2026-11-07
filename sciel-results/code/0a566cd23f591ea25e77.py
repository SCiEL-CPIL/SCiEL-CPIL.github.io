import time
import random
from itertools import combinations

def construct():
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

    results = []
    start_time = time.time()

    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        
        # Reconstruct graph
        # C = 0..k-1, I = k..n-1
        # Edges in C: all pairs
        # Edges in I: none
        # Cross edges: random
        
        # We need to cover all edges exactly once.
        
        # 1. Handle cross edges and some C-edges using I-vertices
        covered_C_edges = set()
        cliques = []
        
        # Generate neighbors for I vertices
        rng = random.Random(seed)
        neighbors_I = [set() for _ in range(n)]
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    neighbors_I[v].add(u)
        
        # Process each v in I
        # To potentially improve, we could sort I vertices by degree in C (ascending)
        # to let "small" vertices take fewer C-edges, leaving more for "large" ones?
        # Or descending?
        # Let's try sorting by degree ascending.
        I_vertices = list(range(k, n))
        I_vertices.sort(key=lambda v: len(neighbors_I[v]))
        
        for v in I_vertices:
            U = set(neighbors_I[v])
            
            while U:
                if time.time() - start_time > 115:
                    # Emergency fallback: just add pairs
                    # This might break validity if not careful, but we must finish.
                    # However, k is small, so this loop is fast.
                    pass
                
                # Build forbidden graph on U
                # Vertices: U
                # Edge (u, w) if (min(u,w), max(u,w)) in covered_C_edges
                # We want a large independent set in this forbidden graph.
                
                # Greedy MIS heuristic
                # 1. Calculate degrees in forbidden graph
                # 2. Pick min degree vertex
                # 3. Remove it and its neighbors
                
                # To speed up, we can just iterate if U is small.
                # k <= 30, so U <= 30.
                
                # Let's use a simple greedy:
                # Sort U by degree in forbidden graph (ascending)
                # Pick first, remove it and its neighbors from consideration for THIS clique.
                
                # Build adjacency list for forbidden graph
                # adj[u] = set of w in U such that (u,w) is forbidden
                adj = {u: set() for u in U}
                for u in U:
                    for w in U:
                        if u == w:
                            continue
                        edge = (min(u, w), max(u, w))
                        if edge in covered_C_edges:
                            adj[u].add(w)
                
                # Sort U by degree
                sorted_U = sorted(U, key=lambda x: len(adj[x]))
                
                S = []
                available = set(U)
                
                for u in sorted_U:
                    if u not in available:
                        continue
                    # Check if u is still "available" (not removed by previous picks)
                    # Since we process in order, we just check if it's in available.
                    # When we pick u, we remove u and adj[u] from available.
                    S.append(u)
                    available.discard(u)
                    for w in adj[u]:
                        available.discard(w)
                
                if not S:
                    # Should not happen if U is not empty, but just in case
                    S = [min(U)]
                    available.discard(S[0])
                
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Mark C-edges within S as covered
                for u1, u2 in combinations(sorted(S), 2):
                    covered_C_edges.add((u1, u2))
                    
                U = U - set(S)
                
        # 2. Handle remaining C-edges
        remaining_C_edges = set()
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_C_edges:
                    remaining_C_edges.add((u, w))
        
        # Greedy cover remaining_C_edges with cliques
        while remaining_C_edges:
            if time.time() - start_time > 118:
                # Fallback: add edges as size-2 cliques
                # This is valid but suboptimal.
                for e in list(remaining_C_edges):
                    cliques.append([e[0], e[1]])
                break
            
            # Pick an edge
            u, w = min(remaining_C_edges)
            
            # Build maximal clique containing u, w in the graph of remaining_C_edges
            clique_set = {u, w}
            
            # Candidates are vertices in C
            # We want to add x if (x,y) is in remaining_C_edges for all y in clique_set
            # To make it deterministic, iterate x in sorted order
            for x in range(k):
                if x in clique_set:
                    continue
                can_add = True
                for y in clique_set:
                    edge = (min(x, y), max(x, y))
                    if edge not in remaining_C_edges:
                        can_add = False
                        break
                if can_add:
                    clique_set.add(x)
            
            clique_list = sorted(clique_set)
            cliques.append(clique_list)
            
            # Remove edges in clique_set from remaining_C_edges
            for u1, u2 in combinations(clique_list, 2):
                remaining_C_edges.discard((u1, u2))
                
        results.append(cliques)
        
    return results
