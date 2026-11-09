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
        I_sorted = sorted(I, key=lambda v: len(adj[v] & set(C)))
        
        # Group independent vertices by their neighborhood in C
        neighborhood_groups = defaultdict(list)
        for v in I_sorted:
            N_v = frozenset(adj[v] & set(C))
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        # Process groups
        for N_v_frozen, group_vertices in neighborhood_groups.items():
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 1:
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
                        
            # Try multiple colorings and pick best
            best_coloring = None
            best_color_count = float('inf')
            
            # Ordering 1: DSATUR
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
            
            # Ordering 4: Randomized restarts (up to 5)
            import random as _rnd
            for trial in range(5):
                colors = {}
                color_set = {u: set() for u in h_vertices}
                uncolored = list(h_vertices)
                _rnd.shuffle(uncolored)
                while uncolored:
                    # Pick from uncolored with some randomness favoring higher saturation
                    best = max(uncolored, key=lambda x: (sat_degree[x] if x in sat_degree else 0, len(h_adj[x]), _rnd.random()))
                    used_colors = color_set[best]
                    c = 0
                    while c in used_colors:
                        c += 1
                    colors[best] = c
                    color_set[best].add(c)
                    uncolored.remove(best)
                    for w in h_adj[best]:
                        if w in uncolored:
                            if c not in color_set[w]:
                                color_set[w].add(c)
                color_count = max(colors.values()) + 1 if colors else 0
                if color_count < best_color_count:
                    best_color_count = color_count
                    best_coloring = dict(colors)
            
            colors = best_coloring
            color_count = best_color_count
            
            color_classes = {}
            for u, c in colors.items():
                if c not in color_classes:
                    color_classes[c] = []
                color_classes[c].append(u)
                
            for c in range(color_count):
                S = color_classes.get(c, [])
                if len(S) < 1:
                    continue
                clique = S + group_vertices
                partition.append(sorted(clique))
                for i in range(len(S)):
                    for j in range(i+1, len(S)):
                        used_edges.add((min(S[i], S[j]), max(S[i], S[j])))
                            
        # Remaining edges in C - use bitmask for speed
        remaining_edges = internal_edges - used_edges
        rem_set = set(remaining_edges)
        
        # Build bitmask adjacency for remaining edges
        rem_adj_mask = [0] * k
        for u, v in rem_set:
            rem_adj_mask[u] |= (1 << v)
            rem_adj_mask[v] |= (1 << u)
        
        # Track which vertices are still "active" (have at least one remaining edge)
        active_mask = 0
        for u, v in rem_set:
            active_mask |= (1 << u) | (1 << v)
        
        while rem_set:
            # Pick edge with highest common neighbor count using bitmask
            best_edge = None
            best_score = -1
            for e in rem_set:
                u, v = e
                # Common neighbors: vertices adjacent to both u and v in rem_set
                common = rem_adj_mask[u] & rem_adj_mask[v]
                # Exclude u and v themselves
                common &= ~(1 << u) & ~(1 << v)
                score = bin(common).count('1')
                if score > best_score:
                    best_score = score
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(rem_set))
                
            u, v = best_edge
            current_clique = {u, v}
            
            # Greedily add vertices that are adjacent to all in current_clique
            changed = True
            while changed:
                changed = False
                # Find candidates: adjacent to all current clique members
                candidates = rem_adj_mask[u] & rem_adj_mask[v]
                for x in current_clique:
                    candidates &= rem_adj_mask[x]
                candidates &= ~(1 << u) & ~(1 << v)
                # Remove already in clique
                for x in current_clique:
                    candidates &= ~(1 << x)
                
                # Pick the candidate with highest degree in remaining graph
                best_w = -1
                best_deg = -1
                w = 0
                while w < k:
                    if (candidates >> w) & 1:
                        deg = bin(rem_adj_mask[w]).count('1')
                        if deg > best_deg:
                            best_deg = deg
                            best_w = w
                    w += 1
                
                if best_w >= 0:
                    current_clique.add(best_w)
                    changed = True
                    # Update rem_adj_mask for removed edges
                    for x in current_clique:
                        if x != best_w:
                            rem_adj_mask[x] &= ~(1 << best_w)
                            rem_adj_mask[best_w] &= ~(1 << x)
            
            clique_list = sorted(list(current_clique))
            if len(clique_list) >= 2:
                partition.append(clique_list)
                
            # Remove all edges in clique from rem_set and update masks
            clique_sorted = sorted(current_clique)
            for i in range(len(clique_sorted)):
                for j in range(i+1, len(clique_sorted)):
                    a, b = clique_sorted[i], clique_sorted[j]
                    rem_set.discard((min(a, b), max(a, b)))
            # Update active_mask
            for x in current_clique:
                active_mask &= ~(1 << x)
                        
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
