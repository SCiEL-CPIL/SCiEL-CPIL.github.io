# EVOLVE-BLOCK-START
import sys
from random import Random
from collections import deque

def construct():
    # Generate the instance graphs using the same method as the verifier
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def min_path_cover(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = {i: [] for i in range(max(edges) + 1 if edges else 1)}
        # Actually, let's use a dict of sets for edges and adjacency
        edge_set = set()
        adj = {}
        degree = {}
        for u, v in edges:
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            edge_set.add((u, v))
        
        remaining_edges = set(edge_set)
        paths = []
        
        def get_odd_vertices():
            rem_degree = {}
            for u, v in remaining_edges:
                rem_degree[u] = rem_degree.get(u, 0) + 1
                rem_degree[v] = rem_degree.get(v, 0) + 1
            return [v for v, d in rem_degree.items() if d % 2 == 1]
        
        def find_shortest_path(start, end, rem_edges):
            """Find shortest path in graph defined by rem_edges."""
            if start == end:
                return [start]
            
            # BFS
            parent = {start: None}
            queue = deque([start])
            
            while queue:
                curr = queue.popleft()
                if curr == end:
                    # Reconstruct path
                    path = []
                    node = end
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                
                # Get neighbors via rem_edges
                neighbors = []
                for (u, v) in rem_edges:
                    if u == curr:
                        neighbors.append(v)
                    elif v == curr:
                        neighbors.append(u)
                
                for nb in neighbors:
                    if nb not in parent:
                        parent[nb] = curr
                        queue.append(nb)
                        
            return None # No path found
        
        while remaining_edges:
            odd_verts = get_odd_vertices()
            
            if odd_verts:
                # Pair up odd vertices greedily by shortest path distance
                # To be efficient, we just pick the first two? Or the closest pair?
                # Let's find the closest pair among all odd vertices.
                # N is small, so O(N^2) is fine.
                best_pair = None
                best_dist = float('inf')
                best_path = None
                
                # Optimization: We only need to find *a* valid path.
                # Finding the *closest* pair might be a good heuristic.
                for i in range(len(odd_verts)):
                    for j in range(i + 1, len(odd_verts)):
                        u, v = odd_verts[i], odd_verts[j]
                        # Check if path exists
                        p = find_shortest_path(u, v, remaining_edges)
                        if p:
                            dist = len(p) - 1
                            if dist < best_dist:
                                best_dist = dist
                                best_pair = (u, v)
                                best_path = p
                
                if best_path is None:
                    # Fallback: should not happen if graph is connected
                    # But if it does, just pick any path
                    start = odd_verts[0]
                    # Find any path starting from start
                    # Use BFS to find any reachable node
                    parent = {start: None}
                    queue = deque([start])
                    found = False
                    while queue:
                        curr = queue.popleft()
                        for (u, v) in remaining_edges:
                            if u == curr:
                                nb = v
                            elif v == curr:
                                nb = u
                            else:
                                continue
                            if nb not in parent:
                                parent[nb] = curr
                                queue.append(nb)
                                if nb != start:
                                    # Reconstruct
                                    path = []
                                    node = nb
                                    while node is not None:
                                        path.append(node)
                                        node = parent[node]
                                    path.reverse()
                                    best_path = path
                                    found = True
                                    break
                        if found: break
                    if not best_path:
                        break # Stuck
                    
                    u, v = start, best_path[-1]
            else:
                # No odd vertices. Graph is Eulerian.
                # We need to pick a path.
                # Pick an arbitrary edge (u, v).
                # Find a simple path starting at u.
                # To minimize paths, we want a long path.
                # But we must ensure we don't disconnect the graph in a way that requires more paths.
                # Simple heuristic: Pick the longest simple path?
                # Or just pick a path from u to any other node.
                # Let's pick an arbitrary edge and extend it.
                u, v = next(iter(remaining_edges))
                # Find a path starting at u
                # We can try to find a path that ends at a node that allows the rest to be Eulerian?
                # If we pick a path u->w, u and w become odd.
                # Then we proceed with the odd-vertex logic.
                # So we just need *any* simple path starting at u.
                # Let's use BFS to find the *farthest* reachable node from u to get a long path?
                # Or just the first one found.
                
                # Let's find a simple path from u to some v'
                # We can use a DFS to find a long path, but BFS is safer for connectivity.
                # Let's just pick the neighbor v and extend greedily?
                # Let's use the same "find path" logic but target is "any node".
                # We'll find a path from u to a node that is "good".
                # For now, let's just find *a* path from u to v (the neighbor).
                # This creates 2 odd vertices u and v.
                # Then the next iteration will handle them.
                
                # Actually, if we just pick the edge (u, v), the path is [u, v].
                # This is a valid simple path.
                best_path = [u, v]
                u, v = u, v # For removal logic below

            # Remove edges of best_path from remaining_edges
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i+1]
                if (a, b) in remaining_edges:
                    remaining_edges.remove((a, b))
                elif (b, a) in remaining_edges:
                    remaining_edges.remove((b, a))
            
            paths.append(best_path)
            
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = min_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
