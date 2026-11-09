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
        
        for v in I_sorted:
            N_v = adj[v] & set(C)
            if not N_v:
                continue
            
            n_v = len(N_v)
            if n_v == 1:
                # Single vertex, clique is {v, u}
                u = N_v.pop()
                partition.append(sorted([v, u]))
                # No C-edges used
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
                        
            # DSATUR-style greedy coloring of H
            # Track saturation degree (number of distinct colors in neighborhood)
            colors = {}
            sat_degree = {u: 0 for u in h_vertices}
            color_set = {u: set() for u in h_vertices}
            uncolored = set(h_vertices)
            
            while uncolored:
                # Pick vertex with highest saturation degree, break ties by degree in H
                best = max(uncolored, key=lambda x: (sat_degree[x], len(h_adj[x])))
                # Find smallest available color
                used_colors = color_set[best]
                c = 0
                while c in used_colors:
                    c += 1
                colors[best] = c
                color_set[best].add(c)
                uncolored.discard(best)
                # Update saturation of neighbors
                for w in h_adj[best]:
                    if w in uncolored:
                        if c not in color_set[w]:
                            color_set[w].add(c)
                            sat_degree[w] = len(color_set[w])
                
            color_count = max(colors.values()) + 1 if colors else 0
            
            # Group vertices by color
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                    
                clique = S + [v]
                partition.append(sorted(clique))
                
                # Mark edges in S as used
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C - use improved greedy with proper clique finding
        remaining_edges = internal_edges - used_edges
        rem_set = set(remaining_edges)
        
        if rem_set:
            # Build adjacency for the remaining edge graph
            rem_adj = [set() for _ in range(k)]
            for u, v in rem_set:
                rem_adj[u].add(v)
                rem_adj[v].add(u)
            
            # Greedy: repeatedly find a large clique in the remaining edge graph
            from random import Random as _Random
            rng2 = _Random(seed + 9999)
            
            best_partition = None
            best_count = float('inf')
            
            for attempt in range(5):
                if time.time() > deadline - 5:
                    break
                    
                rem_set_copy = set(rem_set)
                rem_adj_copy = [set(s) for s in rem_adj]
                temp_partition = []
                
                active_vertices = set()
                for u, v in rem_set_copy:
                    active_vertices.add(u)
                    active_vertices.add(v)
                
                while rem_set_copy:
                    if not active_vertices:
                        break
                    
                    if attempt == 0:
                        best_v = max(active_vertices, key=lambda x: len(rem_adj_copy[x]))
                    else:
                        sorted_verts = sorted(active_vertices, key=lambda x: -len(rem_adj_copy[x]))
                        top_n = min(3, len(sorted_verts))
                        best_v = rng2.choice(sorted_verts[:top_n])
                    
                    current_clique = {best_v}
                    candidates = set(rem_adj_copy[best_v]) & active_vertices
                    
                    while candidates:
                        best_cand = None
                        best_score = -1
                        for w in candidates:
                            common = len(rem_adj_copy[w] & current_clique)
                            if common > best_score:
                                best_score = common
                                best_cand = w
                        if best_cand is None:
                            break
                        current_clique.add(best_cand)
                        new_candidates = set()
                        for w in candidates:
                            if w == best_cand:
                                continue
                            if all(x in rem_adj_copy[w] for x in current_clique):
                                new_candidates.add(w)
                        candidates = new_candidates
                    
                    clique_list = sorted(current_clique)
                    if len(clique_list) >= 2:
                        temp_partition.append(clique_list)
                    
                    for i in range(len(clique_list)):
                        for j in range(i+1, len(clique_list)):
                            a, b = clique_list[i], clique_list[j]
                            e = (min(a, b), max(a, b))
                            rem_set_copy.discard(e)
                            if b in rem_adj_copy[a]:
                                rem_adj_copy[a].discard(b)
                            if a in rem_adj_copy[b]:
                                rem_adj_copy[b].discard(a)
                    
                    for v in clique_list:
                        if not rem_adj_copy[v]:
                            active_vertices.discard(v)
                
                if len(temp_partition) < best_count:
                    best_count = len(temp_partition)
                    best_partition = temp_partition
                
                if best_count == 0:
                    break
            
            partition.extend(best_partition)
                        
        return partition

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    for spec in specs:
        if time.time() > deadline:
            # Return what we have so far with empty lists for remaining
            while len(results) < 3:
                results.append([])
            return results
        res = solve_instance(*spec)
        # Ensure types are correct: list of lists of Python ints
        cleaned = []
        for clique in res:
            cleaned.append([int(x) for x in clique])
        results.append(cleaned)
        
    return results
