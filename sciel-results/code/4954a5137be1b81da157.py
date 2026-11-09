import time
import sys
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def dsatur_color(adj_dict, vertices):
        if not vertices:
            return {}
        colors = {}
        color_set = {u: set() for u in vertices}
        uncolored = set(vertices)
        sat_degree = {u: 0 for u in vertices}
        
        while uncolored:
            best = max(uncolored, key=lambda x: (sat_degree[x], len(adj_dict.get(x, set()))))
            used_colors = color_set[best]
            c = 0
            while c in used_colors:
                c += 1
            colors[best] = c
            color_set[best].add(c)
            uncolored.discard(best)
            for w in adj_dict.get(best, set()):
                if w in uncolored:
                    if c not in color_set[w]:
                        color_set[w].add(c)
                        sat_degree[w] = len(color_set[w])
        return colors

    def solve_instance(n, k, p, seed):
        edges = instance(n, k, p, seed)
        
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        C = list(range(k))
        I = list(range(k, n))
        C_set = set(C)
        
        internal_edges = set((u, v) for u in range(k) for v in range(u+1, k))
        used_edges = set()
        partition = []
        
        # Group independent vertices by their neighborhood in C
        neighborhood_groups = defaultdict(list)
        for v in I:
            N_v = frozenset(adj[v] & C_set)
            if N_v:
                neighborhood_groups[N_v].append(v)
        
        # Sort groups by neighborhood size (ascending)
        sorted_groups = sorted(neighborhood_groups.items(), key=lambda item: len(item[0]))
        
        for N_v_frozen, group_vertices in sorted_groups:
            N_v = set(N_v_frozen)
            n_v = len(N_v)
            
            if n_v == 0:
                continue
                
            if n_v == 1:
                u = next(iter(N_v))
                for v in group_vertices:
                    partition.append(sorted([v, u]))
                continue
                
            # Build complement graph: edge exists if NOT in used_edges
            h_vertices = list(N_v)
            comp_adj = {u: set() for u in h_vertices}
            
            for i in range(len(h_vertices)):
                u = h_vertices[i]
                for j in range(i+1, len(h_vertices)):
                    w = h_vertices[j]
                    e = (min(u, w), max(u, w))
                    if e not in used_edges:
                        comp_adj[u].add(w)
                        comp_adj[w].add(u)
                        
            # Color comp_adj to get clique cover of unused-edge graph
            colors = dsatur_color(comp_adj, h_vertices)
            
            color_count = max(colors.values()) + 1 if colors else 0
            
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
                            
        # Remaining edges in C
        remaining_edges = internal_edges - used_edges
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
            
            # Greedily extend clique
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
                
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    rem_set.discard((min(a, b), max(a, b)))
                        
        # Refinement: try to merge cliques that share vertices
        # This is a simple local improvement
        changed = True
        max_iterations = 50
        iter_count = 0
        while changed and iter_count < max_iterations:
            changed = False
            iter_count += 1
            if time.time() > deadline - 10:
                break
            
            new_partition = []
            used_cliques = set()
            
            for i in range(len(partition)):
                if i in used_cliques:
                    continue
                for j in range(i+1, len(partition)):
                    if j in used_cliques:
                        continue
                    # Try to merge partition[i] and partition[j]
                    merged = set(partition[i]) | set(partition[j])
                    # Check if merged is a clique
                    is_clique = True
                    merged_list = sorted(merged)
                    for a_idx in range(len(merged_list)):
                        for b_idx in range(a_idx+1, len(merged_list)):
                            a, b = merged_list[a_idx], merged_list[b_idx]
                            if (a, b) not in adj[a]:
                                is_clique = False
                                break
                        if not is_clique:
                            break
                    if is_clique:
                        new_partition.append(merged_list)
                        used_cliques.add(i)
                        used_cliques.add(j)
                        changed = True
                        break
                else:
                    new_partition.append(partition[i])
                    used_cliques.add(i)
            
            if changed:
                partition = new_partition
        
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
