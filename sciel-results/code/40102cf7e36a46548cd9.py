import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose_paths(edges):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        # Build adjacency with edge sets for fast removal
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            edge_key = (u, v) if u < v else (v, u)
            adj[u].add((v, edge_key))
            adj[v].add((u, edge_key))
            
        remaining_edges = set((u, v) if u < v else (v, u) for u, v in edges)
        paths = []
        
        # Helper to get available neighbors from a vertex
        def get_available_neighbors(v, visited):
            candidates = []
            for w, e_key in adj[v]:
                if w in visited:
                    continue
                if e_key in remaining_edges:
                    # Count available neighbors of w (Warnsdorff)
                    avail_count = 0
                    for x, _ in adj[w]:
                        if x not in visited:
                            ex = (w, x) if w < x else (x, w)
                            if ex in remaining_edges:
                                avail_count += 1
                    candidates.append((avail_count, w, e_key))
            return candidates
        
        # Greedy path construction with backtracking
        def build_path(start, max_backtrack=3):
            best_path = None
            best_len = 0
            
            # Try different strategies for start vertex selection
            def try_path(start_v, backtrack_depth=0):
                nonlocal best_path, best_len
                if time.time() > deadline - 5:
                    return
                
                path = [start_v]
                visited = {start_v}
                current = start_v
                removed_edges = []
                
                # Recursive backtracking
                def backtrack(current_v, path, visited, removed_edges, depth):
                    nonlocal best_path, best_len
                    if time.time() > deadline - 5:
                        return
                    
                    candidates = get_available_neighbors(current_v, visited)
                    if not candidates:
                        # End of path
                        if len(path) > best_len:
                            best_len = len(path)
                            best_path = list(path)
                        return
                    
                    # Sort by Warnsdorff rule
                    candidates.sort(key=lambda x: x[0])
                    
                    # Limit branching factor
                    max_branch = 3 if depth < backtrack_depth else 1
                    
                    for i, (_, next_v, e_key) in enumerate(candidates[:max_branch]):
                        remaining_edges.remove(e_key)
                        removed_edges.append(e_key)
                        path.append(next_v)
                        visited.add(next_v)
                        
                        backtrack(next_v, path, visited, removed_edges, depth + 1)
                        
                        # Undo
                        path.pop()
                        visited.discard(next_v)
                        removed_edges.pop()
                        remaining_edges.add(e_key)
                
                backtrack(start_v, path, visited, removed_edges, 0)
            
            try_path(start)
            
            # Restore edges if we found a path
            if best_path:
                # Remove edges from remaining
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    e_key = (u, v) if u < v else (v, u)
                    remaining_edges.discard(e_key)
                return best_path
            else:
                # Fallback: just take one edge
                if remaining_edges:
                    e = next(iter(remaining_edges))
                    remaining_edges.remove(e)
                    return list(e)
                return []
        
        while remaining_edges:
            # Calculate degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            # Find start vertex: Max degree, tie-break by neighbor degree sum
            start_candidates = [i for i in range(n) if deg[i] > 0]
            if not start_candidates:
                break
            
            def start_score(v):
                # Primary: degree, Secondary: sum of neighbor degrees
                neighbor_deg_sum = sum(deg[w] for w, _ in adj[v]
                                       if (v, w) in remaining_edges or (w, v) in remaining_edges)
                return (deg[v], neighbor_deg_sum)
            
            start = max(start_candidates, key=start_score)
            
            path = build_path(start)
            if path:
                paths.append(path)
        
        # Post-processing: try to merge paths
        # If two paths have endpoints that are adjacent via an unused edge, merge them
        # But we already used all edges, so this is just for validation
        return paths
    
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        if time.time() > deadline:
            # Fallback: return seed-like result (each edge a path) for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
            
        edges = instance(*spec)
        
        paths = decompose_paths(edges)
        
        results.append(paths)

    return results
