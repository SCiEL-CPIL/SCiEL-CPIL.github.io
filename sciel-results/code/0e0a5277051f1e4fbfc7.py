"""Clique partitions of three fixed split (chordal) graphs."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    from random import Random

    t0 = time.time()
    DEADLINE = t0 + 110.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = set()
        # C-C edges (clique)
        for u in range(k):
            for v in range(u + 1, k):
                edges.add((u, v))
        # C-I edges
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def dsatur_coloring(vertices, adj):
        """
        Colors the graph (vertices, adj) using DSATUR heuristic.
        Returns a list of color classes (lists of vertices).
        adj is a dict of vertex -> set of neighbors.
        """
        if not vertices:
            return []
        
        uncolored = set(vertices)
        colors = {}
        color_sets = {v: set() for v in vertices}
        # Pre-calculate degrees
        deg = {v: len(adj[v]) for v in vertices}
        
        while uncolored:
            # Select vertex with max saturation degree, tie-break by degree
            best_v = None
            best_sat = -1
            best_deg = -1
            
            for v in uncolored:
                sat = len(color_sets[v])
                if sat > best_sat or (sat == best_sat and deg[v] > best_deg):
                    best_sat = sat
                    best_deg = deg[v]
                    best_v = v
                    
            # Assign smallest available color
            c = 0
            while c in color_sets[best_v]:
                c += 1
            colors[best_v] = c
            uncolored.remove(best_v)
            
            # Update neighbors
            for u in adj[best_v]:
                if u in uncolored:
                    if c not in color_sets[u]:
                        color_sets[u].add(c)
                        
        # Group by color
        max_color = max(colors.values()) if colors else 0
        color_classes = [[] for _ in range(max_color + 1)]
        for v, c in colors.items():
            color_classes[c].append(v)
            
        # Filter out empty classes
        return [cls for cls in color_classes if cls]

    def partition_cliques(n, k, p, seed):
        edges = instance(n, k, p, seed)
        cliques = []
        
        # Identify C and I
        C = list(range(k))
        I = list(range(k, n))
        
        # N_C[v] for v in I: set of neighbors in C
        N_C = {v: set() for v in I}
        for v in I:
            for u in C:
                if (u, v) in edges:
                    N_C[v].add(u)
                    
        # used_edges: set of edges in C that are already covered by a clique involving an I-vertex
        used_edges = set()
        
        # Sort I vertices by size of N_C (ascending)
        # Processing smaller neighborhoods first preserves the density of the
        # available C-C edge graph for the larger neighborhoods, which are
        # harder to partition into few cliques.
        I_sorted = sorted(I, key=lambda v: len(N_C[v]))
        
        for v in I_sorted:
            if time.time() > DEADLINE:
                break
                
            neighbors = N_C[v]
            if not neighbors:
                continue
                
            # We need to partition neighbors into sets S such that S is a clique
            # in the available graph (edges not in used_edges).
            # This is equivalent to finding independent sets in the graph H
            # induced by neighbors, where H has an edge (u, w) if (u, w) IS in used_edges.
            
            # Build H adjacency
            H_adj = {u: set() for u in neighbors}
            for u in neighbors:
                for w in neighbors:
                    if u == w: continue
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        H_adj[u].add(w)
                        H_adj[w].add(u)
                        
            # Color H to get independent sets (which are cliques in available graph)
            color_classes = dsatur_coloring(neighbors, H_adj)
            
            for S in color_classes:
                if not S: continue
                # S is a clique in the available graph
                # Create clique {v} U S
                clique = [v] + sorted(S)
                cliques.append(clique)
                
                # Mark edges in S as used
                S_sorted = sorted(S)
                for i in range(len(S_sorted)):
                    for j in range(i+1, len(S_sorted)):
                        e = (S_sorted[i], S_sorted[j])
                        used_edges.add(e)
                        
        # Now cover remaining edges in C
        # Remaining edges are those in E_C not in used_edges
        remaining_edges = set()
        for u in range(k):
            for w in range(u+1, k):
                e = (u, w)
                if e in edges and e not in used_edges:
                    remaining_edges.add(e)
                    
        if remaining_edges:
            # Vertices involved in remaining edges
            rem_vertices = set()
            for u, w in remaining_edges:
                rem_vertices.add(u)
                rem_vertices.add(w)
                
            # Build graph G_rem of remaining edges
            G_rem_adj = {v: set() for v in rem_vertices}
            for u, w in remaining_edges:
                G_rem_adj[u].add(w)
                G_rem_adj[w].add(u)
                
            # We need to cover all edges in G_rem with cliques.
            # This is equivalent to Coloring the complement of G_rem.
            
            comp_rem_adj = {v: set() for v in rem_vertices}
            for v in rem_vertices:
                for w in rem_vertices:
                    if v == w: continue
                    if w not in G_rem_adj[v]:
                        comp_rem_adj[v].add(w)
                        comp_rem_adj[w].add(v)
                        
            color_classes = dsatur_coloring(rem_vertices, comp_rem_adj)
            
            for S in color_classes:
                if len(S) >= 2:
                    cliques.append(sorted(S))
                elif len(S) == 1:
                    # Isolated vertex in G_rem?
                    # This means it has no remaining edges.
                    # But rem_vertices was built from remaining_edges, so every vertex has degree >= 1.
                    # So this shouldn't happen.
                    pass
                    
        return cliques

    result = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > DEADLINE:
            break
        cliques = partition_cliques(*spec)
        cleaned = []
        for c in cliques:
            cl = [int(x) for x in c]
            if len(cl) >= 2 and len(set(cl)) == len(cl):
                cleaned.append(cl)
        result.append(cleaned)

    return result
# EVOLVE-BLOCK-END
