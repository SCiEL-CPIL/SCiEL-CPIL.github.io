import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin from 150s
    
    def get_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def min_vertex_cover(n, edges):
        if not edges:
            return set()
            
        # Build adjacency sets for faster degree calculation
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Greedy: Pick vertex covering most uncovered edges
        uncovered = set(edges)
        cover = set()
        
        while uncovered:
            if time.time() > DEADLINE:
                break
            
            # Count how many uncovered edges each vertex covers
            edge_count = {i: 0 for i in range(n)}
            for u, v in uncovered:
                edge_count[u] += 1
                edge_count[v] += 1
            
            # Pick vertex covering most uncovered edges
            best_v = max(range(n), key=lambda x: edge_count[x])
            if edge_count[best_v] == 0:
                break
                
            cover.add(best_v)
            
            # Remove all edges incident to best_v from uncovered
            uncovered = {(u, v) for u, v in uncovered if u != best_v and v != best_v}
            
        # Local Search to improve
        # Helper to check if a set covers all edges
        def is_valid_cover(c):
            for u, v in edges:
                if u not in c and v not in c:
                    return False
            return True
            
        # Precompute for each vertex, which edges it covers
        vertex_edges = {i: set() for i in range(n)}
        for u, v in edges:
            vertex_edges[u].add((u, v))
            vertex_edges[v].add((u, v))
            
        # Track which edges are covered by current solution
        def get_uncovered(c):
            covered = set()
            for v in c:
                covered.update(vertex_edges[v])
            return edges_set - covered
            
        edges_set = set(edges)
        
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and time.time() < DEADLINE and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # 1. Try removing vertices (multiple passes)
            removed_any = False
            for u in sorted(list(cover)):
                if time.time() > DEADLINE:
                    break
                if u not in cover:
                    continue
                new_cover = cover - {u}
                if is_valid_cover(new_cover):
                    cover = new_cover
                    improved = True
                    removed_any = True
            
            # 2. Try 1-swap moves
            if not improved and time.time() < DEADLINE:
                found_swap = False
                for u in sorted(list(cover)):
                    if time.time() > DEADLINE:
                        break
                    if u not in cover:
                        continue
                    for v in range(n):
                        if v in cover:
                            continue
                        new_cover = (cover - {u}) | {v}
                        if is_valid_cover(new_cover):
                            cover = new_cover
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
            
            # 3. Try 2-swap moves (remove 2, add 2) - only if still room
            if not improved and time.time() < DEADLINE and len(cover) >= 2:
                found_2swap = False
                cover_list = sorted(cover)
                for i in range(len(cover_list)):
                    if time.time() > DEADLINE:
                        break
                    for j in range(i + 1, len(cover_list)):
                        if time.time() > DEADLINE:
                            break
                        u1, u2 = cover_list[i], cover_list[j]
                        base_cover = cover - {u1, u2}
                        # Try adding pairs of vertices not in cover
                        for v1 in range(n):
                            if time.time() > DEADLINE:
                                break
                            if v1 in cover:
                                continue
                            for v2 in range(n):
                                if time.time() > DEADLINE:
                                    break
                                if v2 in cover or v2 <= v1:
                                    continue
                                new_cover = base_cover | {v1, v2}
                                if is_valid_cover(new_cover):
                                    cover = new_cover
                                    found_2swap = True
                                    improved = True
                                    break
                            if found_2swap:
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
            
        return sorted(cover)

    results = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > DEADLINE:
            # If we run out of time, return empty sets? No, must be valid.
            # Return the greedy result if we have it, or all vertices if not.
            # For safety, return all non-isolated vertices as fallback
            edges = get_graph(n, p, seed)
            touched = {x for e in edges for x in e}
            results.append(sorted(touched))
            continue
            
        edges = get_graph(n, p, seed)
        cover = min_vertex_cover(n, edges)
        results.append(cover)

    return results
