"""Rewritten program with optimized split-graph clique partition."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def solve_split(n, k, p, seed):
        rng = random.Random(seed)
        
        # Generate cross edges and adjacency
        # C is 0..k-1, I is k..n-1
        # adj_C[v] will be a set of neighbors in C for v in I
        adj_I_to_C = {}
        for v in range(k, n):
            neighbors = []
            for u in range(k):
                if rng.random() < p:
                    neighbors.append(u)
            adj_I_to_C[v] = neighbors
        
        cliques = []
        
        # Step 1: Cover cross edges with cliques {v} U N(v)
        # Track which edges in C are covered
        # covered_edges is a set of frozensets or tuples (u, w) with u < w
        covered_edges = set()
        
        for v in range(k, n):
            neighbors = adj_I_to_C[v]
            if len(neighbors) > 0:
                # Create clique
                clique = [v] + neighbors
                cliques.append(clique)
                
                # Mark all pairs in neighbors as covered
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        u = neighbors[i]
                        w = neighbors[j]
                        if u > w:
                            u, w = w, u
                        covered_edges.add((u, w))
        
        # Step 2: Find residual edges in C
        residual_edges = []
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) not in covered_edges:
                    residual_edges.append((u, w))
        
        # If no residual edges, we are done
        if not residual_edges:
            return cliques
        
        # Step 3: Cover residual edges with cliques within C
        # This is equivalent to coloring the complement of the residual graph
        # Let H be the graph (C, residual_edges)
        # We want to cover edges of H with cliques.
        # A clique in H is an independent set in H_bar (complement).
        # So we need to color H_bar.
        
        # Build adjacency for H_bar
        # H_bar has edge (u, w) if (u, w) is NOT in residual_edges
        # Note: (u, w) is in H_bar if it was covered in Step 1 OR if it doesn't exist?
        # Wait. The vertices are 0..k-1.
        # In H, edge (u, w) exists if it is NOT covered.
        # In H_bar, edge (u, w) exists if it IS covered.
        # Actually, for coloring H_bar, we just need the adjacency list of H_bar.
        
        adj_bar = [set() for _ in range(k)]
        for u in range(k):
            for w in range(u + 1, k):
                if (u, w) in covered_edges:
                    adj_bar[u].add(w)
                    adj_bar[w].add(u)
        
        # Greedy coloring of H_bar
        # We want to minimize the number of colors.
        # Try multiple orderings.
        
        def greedy_color(adj, order):
            colors = {}
            for v in order:
                used_colors = set()
                for u in adj[v]:
                    if u in colors:
                        used_colors.add(colors[u])
                c = 0
                while c in used_colors:
                    c += 1
                colors[v] = c
            return colors
        
        best_colors = None
        best_num_colors = float('inf')
        
        # Try a few random orderings and degree ordering
        vertices = list(range(k))
        
        # 1. Degree ordering (descending)
        deg_order = sorted(vertices, key=lambda x: len(adj_bar[x]), reverse=True)
        colors = greedy_color(adj_bar, deg_order)
        num_colors = max(colors.values()) + 1 if colors else 0
        if num_colors < best_num_colors:
            best_num_colors = num_colors
            best_colors = colors
            
        # 2. Random orderings
        rng_color = random.Random(42)
        for _ in range(100):
            rand_order = vertices[:]
            rng_color.shuffle(rand_order)
            colors = greedy_color(adj_bar, rand_order)
            num_colors = max(colors.values()) + 1 if colors else 0
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = colors
                
        # Extract cliques from best coloring
        # Each color class is a clique in H (and thus a valid clique in the original graph)
        if best_colors:
            color_classes = {}
            for v, c in best_colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(v)
            
            for c, verts in color_classes.items():
                if len(verts) >= 2:
                    cliques.append(sorted(verts))
                    
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        cliques = solve_split(n, k, p, seed)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
