import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_greedy(edges, n, rng, max_paths=None):
        """Greedy: repeatedly find a long simple path, remove its edges."""
        edge_set = set(edges)
        if not edge_set:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if max_paths is not None and len(paths) >= max_paths:
                break
            
            # Find a long simple path using DFS with heuristic
            # Try multiple starting points, pick the longest path found
            best_path = None
            best_len = 0
            
            # Choose starting vertex: prefer high degree
            candidates = [v for v in adj if adj[v]]
            if not candidates:
                break
            
            rng.shuffle(candidates)
            
            for start in candidates[:min(10, len(candidates))]:  # try up to 10 starts
                # DFS to find a long simple path
                visited = set()
                path = []
                
                def dfs(v, current_path, used_edges):
                    nonlocal best_path, best_len
                    current_path.append(v)
                    visited.add(v)
                    
                    if len(current_path) > best_len:
                        best_len = len(current_path)
                        best_path = list(current_path)
                    
                    # Try neighbors in order of degree (highest first)
                    neighbors = list(adj[v])
                    neighbors.sort(key=lambda x: len(adj[x]), reverse=True)
                    
                    for nb in neighbors:
                        edge = tuple(sorted((v, nb)))
                        if edge in remaining_edges and nb not in visited:
                            used_edges.add(edge)
                            dfs(nb, current_path, used_edges)
                            used_edges.discard(edge)
                    
                    current_path.pop()
                    visited.discard(v)
                
                dfs(start, [], set())
            
            if not best_path:
                # Fallback: just take any edge
                edge = next(iter(remaining_edges))
                best_path = list(edge)
            
            # Extract the path edges
            path_edges = set()
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                path_edges.add(e)
            
            # Remove these edges from remaining
            remaining_edges -= path_edges
            for u, v in path_edges:
                adj[u].discard(v)
                adj[v].discard(u)
            
            paths.append(best_path)
        
        return paths

    def validate_and_fix(paths, edges, n):
        """Validate paths and ensure they cover all edges exactly once."""
        edge_set = set(edges)
        used_edges = set()
        valid_paths = []
        
        for path in paths:
            path_valid = True
            path_edge_set = set()
            seen_vertices = set()
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                e = tuple(sorted((u, v)))
                if e not in edge_set or e in used_edges or e in path_edge_set:
                    path_valid = False
                    break
                if u in seen_vertices or v in seen_vertices:
                    path_valid = False
                    break
                seen_vertices.add(u)
                seen_vertices.add(v)
                path_edge_set.add(e)
            if path_valid:
                valid_paths.append(path)
                used_edges |= path_edge_set
        
        # Add any uncovered edges as single-edge paths
        uncovered = edge_set - used_edges
        for e in uncovered:
            valid_paths.append(list(e))
        
        return valid_paths

    def solve_instance(edges, n, rng, time_budget):
        """Solve a single instance with multiple restarts."""
        best_paths = None
        best_count = float('inf')
        
        # Lower bound
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        odd_count = sum(1 for d in deg.values() if d % 2 == 1)
        lb = max(1, (odd_count + 1) // 2) if edges else 0
        
        # Try multiple restarts
        num_restarts = 20
        for restart in range(num_restarts):
            if time.time() - start_time > time_budget:
                break
            
            r = random.Random(restart * 12345 + hash(tuple(edges)) % 100000)
            paths = decompose_greedy(edges, n, r)
            paths = validate_and_fix(paths, edges, n)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = paths
                
                # If we hit the lower bound, stop
                if best_count <= lb:
                    break
        
        if best_paths is None:
            # Fallback: every edge as its own path
            best_paths = [list(e) for e in edges]
        
        return best_paths

    results = []
    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    
    for idx, (n, p, seed) in enumerate(specs):
        if time.time() - start_time > DEADLINE:
            # Fallback for remaining instances
            edges = instance(n, p, seed)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(n, p, seed)
        rng = random.Random(seed + 1000)
        
        # Allocate time budget per instance
        remaining_time = DEADLINE - (time.time() - start_time)
        time_budget = remaining_time * (1 / (len(specs) - idx)) - 2  # reserve 2s per instance
        
        paths = solve_instance(edges, n, rng, time_budget)
        results.append(paths)
    
    return results
