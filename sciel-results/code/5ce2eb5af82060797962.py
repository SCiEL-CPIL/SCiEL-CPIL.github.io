# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_path(edges):
        """Find a long path in the graph defined by edges using greedy DFS."""
        if not edges:
            return []
        
        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        # Start from a vertex with highest degree
        start = max(adj.keys(), key=lambda x: len(adj[x]))
        
        # Greedy DFS to find a long path
        path = [start]
        visited_vertices = {start}
        used_edges = set()
        
        while True:
            current = path[-1]
            neighbors = adj.get(current, set())
            # Find unused edges from current
            available = []
            for v in neighbors:
                e = tuple(sorted((current, v)))
                if e not in used_edges:
                    available.append(v)
            
            if not available:
                break
            
            # Choose neighbor that leads to longest continuation
            best_next = None
            best_len = -1
            for v in available:
                if v not in visited_vertices:
                    # Prefer unvisited vertices
                    deg = len(adj.get(v, set()))
                    if deg > best_len:
                        best_len = deg
                        best_next = v
            
            if best_next is None:
                # All neighbors are visited, pick any available
                best_next = available[0]
            
            e = tuple(sorted((current, best_next)))
            used_edges.add(e)
            path.append(best_next)
            visited_vertices.add(best_next)
        
        return path

    def edge_path_cover(edges):
        """Decompose edges into edge-disjoint paths covering all edges."""
        if not edges:
            return []
        
        remaining = set(tuple(sorted(e)) for e in edges)
        paths = []
        
        while remaining and (time.time() - start_time) < DEADLINE:
            # Convert to list for path finding
            edge_list = list(remaining)
            path = find_longest_path(edge_list)
            
            if len(path) < 2:
                # Single edge or no path found, just take one edge
                e = edge_list[0]
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            # Extract edges from path
            path_edges = set()
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                if e in remaining:
                    path_edges.add(e)
                else:
                    # Edge not in remaining, break
                    path = path[:i + 1]
                    break
            
            if not path_edges:
                e = edge_list[0]
                paths.append([e[0], e[1]])
                remaining.discard(e)
                continue
            
            paths.append(path)
            remaining -= path_edges
        
        # Handle any remaining edges as individual paths
        for e in remaining:
            paths.append([e[0], e[1]])
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if (time.time() - start_time) >= DEADLINE:
            # Fallback: use single-edge paths
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        
        edges = instance(*spec)
        paths = edge_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
