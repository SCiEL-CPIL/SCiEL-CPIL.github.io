import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def beam_search_path(adj, remaining_edges, start, beam_width=5, max_len=100):
        """
        Use beam search to find a long simple path starting from `start`.
        Returns the best path found (list of vertices).
        """
        if start is None or not adj.get(start):
            return [start] if start is not None else []
        
        # Each beam entry: (path_list, set_of_visited_vertices, current_vertex)
        beams = [([start], {start}, start)]
        
        for _ in range(max_len):
            if not beams:
                break
            new_beams = []
            for path, visited, curr in beams:
                # Try all neighbors of curr
                for neighbor in sorted(adj.get(curr, [])):
                    edge = (min(curr, neighbor), max(curr, neighbor))
                    if edge not in remaining_edges:
                        continue
                    if neighbor in visited:
                        continue
                    new_path = path + [neighbor]
                    new_visited = visited | {neighbor}
                    new_beams.append((new_path, new_visited, neighbor))
            
            if not new_beams:
                break
            
            # Score beams by length (primary) and endpoint degree (secondary, lower is better for dead-end)
            def score_beam(b):
                path, visited, curr = b
                deg = len(adj.get(curr, set()))
                # Primary: longer is better; Secondary: lower degree endpoint is better (more likely to terminate)
                return (len(path), -deg)
            
            new_beams.sort(key=score_beam, reverse=True)
            beams = new_beams[:beam_width]
            
            if len(beams[0][0]) == len(new_beams[0][0]) and len(new_beams) == len(beams):
                # No improvement possible in this layer
                break
        
        # Return the longest path from the final beams
        best_path = max((b[0] for b in beams), key=len) if beams else [start]
        return best_path

    def greedy_cover_with_beam(edges, beam_width=5):
        """
        Greedy edge-disjoint path cover using beam search for path construction.
        """
        if not edges:
            return []
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        while remaining_edges:
            # Find a start vertex. Prefer degree 1, then degree 2, etc.
            candidates = []
            for u in adj:
                deg = len(adj[u])
                if deg > 0:
                    candidates.append((deg, u))
            
            if not candidates:
                break
            
            candidates.sort()
            start = candidates[0][1]
            
            # Use beam search to find a long path from start
            path = beam_search_path(adj, remaining_edges, start, beam_width=beam_width)
            
            if len(path) > 1:
                paths.append(path)
                
            # Remove edges used in the path
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
            
            # Clean up isolated vertices
            for u in list(adj.keys()):
                if not adj[u]:
                    del adj[u]
                    
            if time.time() > deadline:
                break
                
        return paths

    def try_merge_paths(paths, edges_set):
        """
        Try to merge two paths into one by finding a common edge.
        Returns (new_paths, success) where new_paths is the updated list if merged.
        """
        n_paths = len(paths)
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                p1 = paths[i]
                p2 = paths[j]
                
                # Check if we can merge p1 and p2
                # Case 1: p1 ends at a vertex that is an endpoint of p2
                # Case 2: p1 starts at a vertex that is an endpoint of p2
                # etc.
                
                # Build edge sets for quick lookup
                e1 = set()
                for k in range(len(p1) - 1):
                    e1.add((min(p1[k], p1[k+1]), max(p1[k], p1[k+1])))
                e2 = set()
                for k in range(len(p2) - 1):
                    e2.add((min(p2[k], p2[k+1]), max(p2[k], p2[k+1])))
                
                # Try all 4 combinations of endpoints
                # p1: [a, ..., b], p2: [c, ..., d]
                # Merge if b==c or b==d or a==c or a==d, and the merged path is simple
                
                a, b = p1[0], p1[-1]
                c, d = p2[0], p2[-1]
                
                merged = None
                # Try merging b with c
                if b == c:
                    candidate = p1 + p2[1:]
                    if len(set(candidate)) == len(candidate):
                        merged = candidate
                # Try merging b with d
                if merged is None and b == d:
                    candidate = p1 + list(reversed(p2))[1:]
                    if len(set(candidate)) == len(candidate):
                        merged = candidate
                # Try merging a with c
                if merged is None and a == c:
                    candidate = list(reversed(p1)) + p2[1:]
                    if len(set(candidate)) == len(candidate):
                        merged = candidate
                # Try merging a with d
                if merged is None and a == d:
                    candidate = list(reversed(p1)) + p2
                    if len(set(candidate)) == len(candidate):
                        merged = candidate
                
                if merged is not None:
                    # Verify all edges in merged path are in edges_set
                    valid = True
                    for k in range(len(merged) - 1):
                        e = (min(merged[k], merged[k+1]), max(merged[k], merged[k+1]))
                        if e not in edges_set:
                            valid = False
                            break
                    if valid:
                        new_paths = paths[:i] + paths[i+1:j] + paths[j+1:] + [merged]
                        return new_paths, True
        
        return paths, False

    def local_search_merge(paths, edges_set, max_iters=100):
        """
        Repeatedly try to merge pairs of paths to reduce the total count.
        """
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            new_paths, success = try_merge_paths(paths, edges_set)
            if success:
                paths = new_paths
                improved = True
            if time.time() > deadline:
                break
        return paths

    def solve_graph(edges):
        if not edges:
            return []
        
        edges_set = set((u, v) if u < v else (v, u) for u, v in edges)
        
        # Try multiple beam widths and pick the best
        best_paths = None
        best_count = float('inf')
        
        for bw in [3, 5, 8, 12]:
            if time.time() > deadline - 5:
                break
            paths = greedy_cover_with_beam(edges, beam_width=bw)
            if not paths:
                continue
            
            # Apply local search to merge paths
            paths = local_search_merge(paths, edges_set, max_iters=50)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
        
        if best_paths is None:
            # Fallback: each edge as a path
            best_paths = [[list(e)] for e in edges]
        
        return best_paths

    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        edges = instance(*spec)
        cover = solve_graph(edges)
        if not cover:
            cover = [[list(e)] for e in edges]
        results.append(cover)
        
    return results
