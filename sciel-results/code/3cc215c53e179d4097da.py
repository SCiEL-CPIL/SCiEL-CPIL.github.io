"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_path_decomposition(edges, max_time=None):
        """
        Greedy approach: repeatedly find the longest simple path in the remaining graph,
        remove its edges, and continue until no edges remain.
        """
        if max_time is None:
            max_time = 30
        
        # Build adjacency from edges
        edge_set = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        remaining_edges = set(edge_set)
        
        while remaining_edges:
            if max_time and (time.time() - start_time) > max_time:
                break
            
            # Find a longest path in the current graph using DFS with backtracking
            # But for efficiency, use a heuristic: start from a leaf or low-degree node
            # and greedily extend
            
            # Find a good starting vertex: prefer vertices with lower degree
            degrees = {v: len(adj[v] & set(remaining_edges_neighbors)) for v in adj}
            # Actually, let's just pick a vertex that's in an edge
            start_candidates = []
            for e in remaining_edges:
                start_candidates.append(e[0])
                start_candidates.append(e[1])
            
            if not start_candidates:
                break
            
            # Try multiple starting points and pick the best path
            best_path = None
            best_len = 0
            
            # Limit the number of starting points we try
            random.shuffle(start_candidates)
            num_starts = min(20, len(start_candidates))
            
            for start in start_candidates[:num_starts]:
                path = _find_longest_path_from(start, remaining_edges, adj)
                if path and len(path) > best_len:
                    best_len = len(path)
                    best_path = path
                
                if max_time and (time.time() - start_time) > max_time:
                    break
            
            if best_path is None:
                # Fallback: just take any single edge
                e = next(iter(remaining_edges))
                best_path = list(e)
            
            paths.append(best_path)
            
            # Remove edges from remaining
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining_edges.discard(e)
            
            # Update adj (optional, but good for correctness)
            # We'll rely on remaining_edges for edge availability
        
        # If we still have remaining edges, add them as single-edge paths
        if remaining_edges:
            for e in remaining_edges:
                paths.append(list(e))
        
        return paths

    def _find_longest_path_from(start, remaining_edges, adj, max_depth=50):
        """
        Find a long simple path starting from 'start' using DFS with backtracking.
        Returns the path as a list of vertices, or None if no path found.
        """
        best_path = [start]
        
        def dfs(current, visited, path):
            nonlocal best_path
            if len(path) > len(best_path):
                best_path = list(path)
            
            # Try to extend
            for neighbor in adj[current]:
                if neighbor not in visited:
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining_edges:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs(neighbor, visited, path)
                        path.pop()
                        visited.discard(neighbor)
        
        dfs(start, {start}, [start])
        return best_path

    def improved_decomposition(edges, max_time=None):
        """
        Better approach: Use the fact that in sparse graphs, we can find longer paths.
        Use a combination of greedy matching and path extension.
        """
        if max_time is None:
            max_time = 30
        
        edge_set = set(edges)
        remaining = set(edge_set)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        while remaining:
            if max_time and (time.time() - start_time) > max_time:
                break
            
            # Strategy: find a maximal path by starting from a vertex of degree 1 or 2
            # in the remaining graph, then greedily extend
            
            # Find vertices with their degrees in remaining graph
            deg = defaultdict(int)
            for u, v in remaining:
                deg[u] += 1
                deg[v] += 1
            
            # Prefer starting from low-degree vertices
            candidates = [v for v in deg if deg[v] <= 2]
            if not candidates:
                candidates = list(deg.keys())
            
            random.shuffle(candidates)
            
            best_path = None
            best_len = 0
            
            for start in candidates[:15]:
                path = _greedy_extend(start, remaining, adj)
                if path and len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if best_path is None:
                e = next(iter(remaining))
                best_path = list(e)
            
            paths.append(best_path)
            
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
        
        if remaining:
            for e in remaining:
                paths.append(list(e))
        
        return paths

    def _greedy_extend(start, remaining, adj):
        """
        Greedily extend a path from start, always choosing the neighbor that
        leads to the longest possible continuation.
        """
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Find unvisited neighbors connected by remaining edges
            candidates = []
            for neighbor in adj[current]:
                if neighbor not in visited:
                    edge = tuple(sorted((current, neighbor)))
                    if edge in remaining:
                        candidates.append(neighbor)
            
            if not candidates:
                break
            
            # Choose the candidate with the fewest remaining edges (to avoid dead ends)
            # or the one that leads to the longest path
            best_next = None
            best_score = -1
            
            for cand in candidates:
                # Score: number of unvisited neighbors of cand
                score = sum(1 for nb in adj[cand] if nb not in visited)
                if score > best_score:
                    best_score = score
                    best_next = cand
            
            if best_next is None:
                break
            
            path.append(best_next)
            visited.add(best_next)
            current = best_next
        
        return path

    # Process each graph
    results = []
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    for spec in specs:
        edges = instance(*spec)
        
        # Try the improved decomposition
        if time.time() < deadline - 5:
            paths = improved_decomposition(edges, max_time=deadline - 5)
        else:
            paths = [[list(e)] for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
