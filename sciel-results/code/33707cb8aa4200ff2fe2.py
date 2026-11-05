"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from collections import deque

def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        remaining = set(edges)
        paths = []
        
        while remaining:
            # Find odd-degree vertices in remaining graph
            deg = {}
            for u, v in remaining:
                deg[u] = deg.get(u, 0) + 1
                deg[v] = deg.get(v, 0) + 1
            
            odd_verts = [v for v in deg if deg[v] % 2 == 1]
            
            if odd_verts:
                # Pick two odd vertices, find simple path between them
                u, v = odd_verts[0], odd_verts[1]
                path = bfs_path(u, v, adj, remaining)
                if path is None:
                    # Fallback: pick any edge
                    u, v = next(iter(remaining))
                    path = [u, v]
            else:
                # All even degrees, pick any edge and extend to a path
                u, v = next(iter(remaining))
                path = [u, v]
                # Try to extend
                path = extend_path(path, adj, remaining)
            
            # Remove path edges
            for i in range(len(path) - 1):
                a, b = path[i], path[i+1]
                e = tuple(sorted((a, b)))
                remaining.discard(e)
            
            paths.append(path)
        
        return paths
    
    def bfs_path(start, end, adj, remaining_edges):
        """Find a simple path from start to end using only remaining edges."""
        if start == end:
            return [start]
        
        # Build adjacency from remaining edges only
        rem_adj = {}
        for u, v in remaining_edges:
            rem_adj.setdefault(u, set()).add(v)
            rem_adj.setdefault(v, set()).add(u)
        
        # BFS
        from collections import deque
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in rem_adj.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def extend_path(path, adj, remaining_edges):
        """Try to extend the path by adding more edges."""
        path = list(path)
        changed = True
        while changed:
            changed = False
            # Try to extend from the end
            end_node = path[-1]
            rem_adj = {}
            for u, v in remaining_edges:
                rem_adj.setdefault(u, set()).add(v)
                rem_adj.setdefault(v, set()).add(u)
            
            for neighbor in rem_adj.get(end_node, []):
                if neighbor not in path:
                    path.append(neighbor)
                    e = tuple(sorted((end_node, neighbor)))
                    remaining_edges.discard(e)
                    changed = True
                    break
            
            # Try to extend from the start
            if not changed:
                start_node = path[0]
                for neighbor in rem_adj.get(start_node, []):
                    if neighbor not in path:
                        path.insert(0, neighbor)
                        e = tuple(sorted((start_node, neighbor)))
                        remaining_edges.discard(e)
                        changed = True
                        break
        
        return path
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
