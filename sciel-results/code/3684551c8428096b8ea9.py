import time
import sys
from itertools import combinations

def construct():
    from random import Random
    
    # Keep the instance generator as is
    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        
        # Internal edges of C
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        
        # Used edges in C
        used_edges = set()
        
        partition = []
        
        # Sort I by degree in C (descending) to process "dense" connections first
        # This might leave a sparser graph for later vertices, or vice versa?
        # If we process high degree first, we use many edges.
        # If we process low degree first, we use few edges.
        # Let's try sorting by degree descending.
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)), reverse=True)
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
                
            # Build graph H on N_v using used_edges
            # Vertices are N_v
            # Edge (a,b) in H if (a,b) in used_edges
            
            # We need to color H.
            # Vertices of H: list(N_v)
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for u in h_vertices:
                for w in h_vertices:
                    if u == w: continue
                    e = (min(u,w), max(u,w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        
            # Greedy coloring of H
            # Order vertices by degree in H (descending) for better coloring
            h_vertices_sorted = sorted(h_vertices, key=lambda x: len(h_adj[x]), reverse=True)
            
            colors = {}
            color_count = 0
            for u in h_vertices_sorted:
                # Find smallest available color
                used_colors = {colors[w] for w in h_adj[u] if w in colors}
                c = 0
                while c in used_colors:
                    c += 1
                colors[u] = c
                color_count = max(color_count, c + 1)
                
            # Group vertices by color
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 2:
                    # If S has 1 vertex, clique is {v, u}. 
                    # This is a valid clique of size 2.
                    # If S has 0 vertices, skip.
                    pass
                    
                clique = S + [v]
                # Check if clique is valid (it should be by construction)
                # S is independent in H, so no edges in S are in used_edges.
                # So all edges in S are available.
                # v is connected to all in S.
                partition.append(sorted(clique))
                
                # Mark edges in S as used
                for u in S:
                    for w in S:
                        if u < w:
                            used_edges.add((u, w))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        
        # Partition remaining_edges into cliques
        # Greedy: pick an edge, extend to max clique
        rem_set = set(remaining_edges)
        while rem_set:
            # Pick an arbitrary edge
            u, v = next(iter(rem_set))
            
            # Find largest clique containing u and v in the graph defined by rem_set
            # Start with {u, v}
            current_clique = {u, v}
            
            # Candidate vertices: neighbors of u and v in rem_set
            # Actually, we can just check all other vertices in C
            # To be efficient, we can look for vertices connected to all in current_clique
            
            # Heuristic: try to add vertices one by one
            # Order candidates by degree in rem_set?
            candidates = [w for w in range(k) if w not in current_clique]
            # Sort candidates by number of connections to current_clique in rem_set
            def score(w):
                count = 0
                for x in current_clique:
                    e = (min(w,x), max(w,x))
                    if e in rem_set:
                        count += 1
                return count
            
            candidates.sort(key=score, reverse=True)
            
            for w in candidates:
                # Check if w can be added to current_clique
                can_add = True
                for x in current_clique:
                    e = (min(w,x), max(w,x))
                    if e not in rem_set:
                        can_add = False
                        break
                if can_add:
                    current_clique.add(w)
                    
            # Add this clique to partition
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            # Remove edges of this clique from rem_set
            for u_c in current_clique:
                for v_c in current_clique:
                    if u_c < v_c:
                        rem_set.discard((u_c, v_c))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        res = solve_instance(*spec)
        # Ensure types are correct
        # res is list of lists of ints
        results.append(res)
        
    return results
