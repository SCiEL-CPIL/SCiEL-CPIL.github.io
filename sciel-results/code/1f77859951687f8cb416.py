"""Strong edge coloring via greedy set-cover over candidate induced matchings."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_fast(edges, adj):
        m = len(edges)
        vert_to_edges = [[] for _ in range(max(adj.keys()) + 1 if adj else 1)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        conflict = [set() for _ in range(m)]
        
        for u in range(len(vert_to_edges)):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    ei = e_list[i]
                    ej = e_list[j]
                    conflict[ei].add(ej)
                    conflict[ej].add(ei)
        
        for i, (u, v) in enumerate(edges):
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            
            for j in range(i + 1, m):
                if j in conflict[i]:
                    continue
                
                w, x = edges[j]
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict[i].add(j)
                    conflict[j].add(i)
        
        return conflict

    def greedy_coloring(edges, conflict, seed=0):
        m = len(edges)
        if m == 0:
            return []
        
        rng = random.Random(seed)
        order = list(range(m))
        rng.shuffle(order)
        
        assign = [-1] * m
        for i in order:
            used_colors = set(assign[j] for j in conflict[i] if assign[j] != -1)
            c = 0
            while c in used_colors:
                c += 1
            assign[i] = c
        
        return assign

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict = build_conflict_fast(edges, adj)
        
        # Generate candidate induced matchings
        candidates = []
        rng = random.Random(seed)
        
        # 1. Greedy random matchings
        for _ in range(100):
            used_vertices = set()
            matching = []
            # Shuffle edge indices
            idxs = list(range(m))
            rng.shuffle(idxs)
            
            for i in idxs:
                u, v = edges[i]
                if u not in used_vertices and v not in used_vertices:
                    # Check if adding this edge violates induced matching property
                    # with respect to already selected edges in this matching
                    valid = True
                    for j in matching:
                        w, x = edges[j]
                        # They don't share vertices (guaranteed by used_vertices check)
                        # Check if joined by an edge
                        if w in adj.get(u, set()) or w in adj.get(v, set()) or \
                           x in adj.get(u, set()) or x in adj.get(v, set()):
                            valid = False
                            break
                    if valid:
                        matching.append(i)
                        used_vertices.add(u)
                        used_vertices.add(v)
            
            if matching:
                candidates.append(frozenset(matching))
        
        # 2. Star-based matchings (pick a vertex, take non-adjacent edges)
        # Actually, induced matching requires no edges between the endpoints.
        # So if we pick edges incident to u, they all share u, so they can't be in the same matching.
        # So star-based doesn't directly work for induced matching unless we pick one edge per star?
        # Let's stick to random greedy which is more flexible.
        
        # 3. Try to extend small matchings
        # For now, rely on the 100 random greedy matchings.
        
        # Remove duplicates
        unique_candidates = list(set(candidates))
        
        if not unique_candidates:
            # Fallback if no candidates found (should not happen)
            return greedy_coloring(edges, conflict, seed)
        
        # Set Cover Greedy
        uncovered = set(range(m))
        chosen_matchings = []
        
        while uncovered:
            best_match = None
            best_score = 0
            for match in unique_candidates:
                score = len(match & uncovered)
                if score > best_score:
                    best_score = score
                    best_match = match
            
            if best_match is None or best_score == 0:
                # No candidate covers any remaining edge? 
                # This can happen if candidates are limited.
                # Fallback: assign remaining edges individually? 
                # Or break and handle leftovers.
                # For now, break and handle leftovers with individual colors.
                break
            
            chosen_matchings.append(best_match)
            uncovered -= best_match
        
        # Assign colors
        assign = [-1] * m
        for c, match in enumerate(chosen_matchings):
            for i in match:
                assign[i] = c
        
        # Handle leftovers (edges not covered by any chosen matching)
        # This should ideally not happen if we have enough candidates, 
        # but if we broke out of the loop, we have leftovers.
        # We need to assign them valid colors.
        # Since we might not have a matching for them, we just assign new colors 
        # ensuring no conflict with existing assignments.
        
        # Actually, the greedy loop `while uncovered` should continue until uncovered is empty 
        # unless no candidate helps. If no candidate helps, it means for every remaining edge,
        # it is not in any candidate? That's impossible since candidates are subsets of edges.
        # Wait, if a candidate has no intersection with uncovered, it's useless.
        # If ALL candidates have no intersection, then we are stuck.
        # This implies the remaining edges are not part of any generated induced matching.
        # We can just assign them one by one with new colors.
        
        for i in uncovered:
            used_colors = set(assign[j] for j in conflict[i] if assign[j] != -1)
            c = 0
            while c in used_colors:
                c += 1
            assign[i] = c
        
        return assign
    
    out = []
    deadline = time.time() + 115 # 5s safety margin
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: Fast Greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict = build_conflict_fast(edges, adj)
            
            assign = [0] * m
            for i in range(m):
                used = set(assign[j] for j in conflict[i])
                c = 0
                while c in used:
                    c += 1
                assign[i] = c
            out.append(assign)
        else:
            # Allocate remaining time, reserving 5s for next
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))
    
    return out
# EVOLVE-BLOCK-END
