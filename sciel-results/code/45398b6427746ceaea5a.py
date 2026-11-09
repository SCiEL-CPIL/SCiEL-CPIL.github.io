import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    from random import Random

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
        
        # Sort I by degree in C (ascending) - process smaller neighborhoods first
        # This leaves more C-edges available for larger neighborhoods
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))
        
        # Group independent vertices by their neighborhood in C
        # Vertices with the same neighborhood can be grouped together
        neighborhood_groups = defaultdict(list)
        for v in I_sorted:
            N_v = frozenset(adj[v] & set(C))
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        # Process groups: for each unique neighborhood, handle all vertices at once
        for N_v_frozen, group_vertices in neighborhood_groups.items():
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 1:
                # Single vertex in neighborhood, each vertex forms a clique with it
                u = next(iter(N_v))
                for v in group_vertices:
                    partition.append(sorted([v, u]))
                continue
                
            # Build graph H on N_v using used_edges
            h_vertices = list(N_v)
            h_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e in used_edges:
                        h_adj[u].add(w)
                        h_adj[w].add(u)
                        
            # Try multiple colorings with different vertex orderings and pick best
            best_coloring = None
            best_color_count = float('inf')
            
            # Ordering 1: DSATUR (saturation degree, then degree)
            colors = {}
            sat_degree = {u: 0 for u in h_vertices}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                best = max(uncolored, key=lambda x: (sat_degree[x], len(h_adj[x])))
                used_colors = color_set[best]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best] = c
                color_set[best].add(c)
                uncolored.discard(best)
                for w in h_adj[best]:
                    if w in uncolored:
                        if c not in color_set[w]:
                            color_set[w].add(c)
                            sat_degree[w] = len(color_set[w])
            
            color_count = max(colors.values()) + 1 if colors else 0
            if color_count < best_color_count:
                best_color_count = color_count
                best_coloring = dict(colors)
            
            # Ordering 2: Largest degree first
            colors = {}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                best = max(uncolored, key=lambda x: len(h_adj[x]))
                used_colors = color_set[best]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best] = c
                color_set[best].add(c)
                uncolored.discard(best)
                for w in h_adj[best]:
                    if w in uncolored:
                        if c not in color_set[w]:
                            color_set[w].add(c)
            
            color_count = max(colors.values()) + 1 if colors else 0
            if color_count < best_color_count:
                best_color_count = color_count
                best_coloring = dict(colors)
            
            # Ordering 3: Smallest degree first
            colors = {}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                best = min(uncolored, key=lambda x: len(h_adj[x]))
                used_colors = color_set[best]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best] = c
                color_set[best].add(c)
                uncolored.discard(best)
                for w in h_adj[best]:
                    if w in uncolored:
                        if c not in color_set[w]:
                            color_set[w].add(c)
            
            color_count = max(colors.values()) + 1 if colors else 0
            if color_count < best_color_count:
                best_color_count = color_count
                best_coloring = dict(colors)
            
            # Use the best coloring found
            colors = best_coloring
            color_count = best_color_count
            
            # Group vertices by color
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            # For each color class, create one clique containing all group vertices + color class vertices
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + group_vertices
                partition.append(sorted(clique))
                
                # Mark edges in S as used
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
        
        # Partition remaining_edges into cliques using maximal clique extraction
        rem_set = set(remaining_edges)
        
        while rem_set:
            # Pick edge with highest combined degree to grow large cliques
            best_edge = None
            best_score = -1
            for e in rem_set:
                u, v = e
                common = 0
                for w in range(k):
                    if w == u or w == v:
                        continue
                    if (min(u,w), max(u,w)) in rem_set and (min(v,w), max(v,w)) in rem_set:
                        common += 1
                score = common
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(rem_set))
                
            u, v = best_edge
            current_clique = {u, v}
            
            changed = True
            while changed:
                changed = False
                for w in range(k):
                    if w in current_clique:
                        continue
                    can_add = True
                    for x in current_clique:
                        e = (min(w, x), max(w, x))
                        if e not in rem_set:
                            can_add = False
                            break
                    if can_add:
                        current_clique.add(w)
                        changed = True
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            clique_list_sorted = sorted(current_clique)
            for i in range(len(clique_list_sorted)):
                for j in range(i+1, len(clique_list_sorted)):
                    a, b = clique_list_sorted[i], clique_list_sorted[j]
                    rem_set.discard((min(a, b), max(a, b)))
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
