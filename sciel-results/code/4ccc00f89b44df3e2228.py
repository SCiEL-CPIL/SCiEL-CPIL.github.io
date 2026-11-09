import random
import time
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list and edge set
        # adj[u] = set of neighbors
        # edge_set = set of tuples (min, max)
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((u, v) if u < v else (v, u))
            
        remaining_edges = set(edge_set)
        paths = []
        
        def get_degrees():
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            return deg

        def find_eulerian_path(start=None):
            """
            Finds an Eulerian path in the graph defined by remaining_edges.
            Assumes 0 or 2 odd vertices.
            Returns list of vertices.
            """
            if not remaining_edges:
                return []
            
            # Get degrees to find start node
            deg = get_degrees()
            if start is None:
                # If 0 odd, start at any node with degree > 0
                # If 2 odd, start at one of the odd nodes
                odds = [v for v, d in deg.items() if d % 2 == 1 and d > 0]
                if len(odds) == 2:
                    start = odds[0]
                else:
                    # Pick any node with degree > 0
                    for v, d in deg.items():
                        if d > 0:
                            start = v
                            break
                    if start is None:
                        return []
            
            # Hierholzer's Algorithm
            # We need a mutable adjacency structure for the remaining edges
            # To avoid copying the whole set, we can use a local dict of sets
            local_adj = defaultdict(set)
            for u, v in remaining_edges:
                local_adj[u].add(v)
                local_adj[v].add(u)
                
            stack = [start]
            path = []
            
            while stack:
                u = stack[-1]
                if local_adj[u]:
                    v = local_adj[u].pop()
                    stack.append(v)
                    # Remove edge from local_adj
                    local_adj[u].discard(v)
                    local_adj[v].discard(u)
                else:
                    path.append(stack.pop())
                    
            path.reverse()
            return path

        def find_simple_path(u, v):
            """
            Finds a simple path from u to v in remaining_edges using BFS.
            Returns list of vertices or None.
            """
            if u == v:
                return [u]
                
            # BFS
            from collections import deque
            queue = deque([(u, [u])])
            visited = {u}
            
            while queue:
                curr, path = queue.popleft()
                if curr == v:
                    return path
                
                for neighbor in adj[curr]:
                    # Check if edge (curr, neighbor) is in remaining_edges
                    e = (curr, neighbor) if curr < neighbor else (neighbor, curr)
                    if e in remaining_edges and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
                        
            return None

        while remaining_edges:
            if time.time() - start_time > time_limit:
                # Fallback: add remaining edges as individual paths
                for e in remaining_edges:
                    paths.append(list(e))
                break
            
            deg = get_degrees()
            odds = [v for v, d in deg.items() if d % 2 == 1]
            
            if len(odds) <= 2:
                # Case 0 or 2 odd vertices: Eulerian path
                path = find_eulerian_path()
                if path:
                    paths.append(path)
                    # Remove edges of this path from remaining_edges
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        e = (u, v) if u < v else (v, u)
                        remaining_edges.discard(e)
                else:
                    # Should not happen if edges exist, but safeguard
                    u, v = next(iter(remaining_edges))
                    paths.append([u, v])
                    remaining_edges.discard((u, v) if u < v else (v, u))
            else:
                # Case > 2 odd vertices: Pair two odds and remove path between them
                # Pick first two odds
                u, v = odds[0], odds[1]
                path = find_simple_path(u, v)
                
                if path:
                    paths.append(path)
                    for i in range(len(path) - 1):
                        a, b = path[i], path[i+1]
                        e = (a, b) if a < b else (b, a)
                        remaining_edges.discard(e)
                else:
                    # If no path found (disconnected?), take any edge
                    a, b = next(iter(remaining_edges))
                    paths.append([a, b])
                    remaining_edges.discard((a, b) if a < b else (b, a))

        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
