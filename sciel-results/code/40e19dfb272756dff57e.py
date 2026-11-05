"""Program for path decomposition of fixed connected graphs (Gallai's conjecture)."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_adjacency(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def eulerian_circuit(adj, start=None):
        """Find an Eulerian circuit in a connected graph where all vertices have even degree."""
        if start is None:
            # Find a vertex with edges
            for v in adj:
                if adj[v]:
                    start = v
                    break
            if start is None:
                return []
        
        # Use Hierholzer's algorithm
        # Make a mutable copy of adjacency
        adj_copy = {v: set(neigh) for v, neigh in adj.items()}
        
        circuit = []
        stack = [start]
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                # Pick next vertex
                u = next(iter(adj_copy[v]))
                adj_copy[v].discard(u)
                adj_copy[u].discard(v)
                stack.append(u)
            else:
                circuit.append(stack.pop())
        
        circuit.reverse()
        return circuit

    def eulerian_path(adj, start, end):
        """Find an Eulerian path from start to end."""
        # Make a mutable copy
        adj_copy = {v: set(neigh) for v, neigh in adj.items()}
        
        path = []
        stack = [start]
        
        while stack:
            v = stack[-1]
            if adj_copy[v]:
                u = next(iter(adj_copy[v]))
                adj_copy[v].discard(u)
                adj_copy[u].discard(v)
                stack.append(u)
            else:
                path.append(stack.pop())
        
        path.reverse()
        
        # Verify it ends at end
        if path[-1] != end:
            return None
        return path

    def find_path_bfs(s, t, adj, n):
        """Find a simple path from s to t using BFS."""
        if s == t:
            return [s]
        
        visited = {s}
        parent = {s: None}
        queue = deque([s])
        
        while queue:
            u = queue.popleft()
            if u == t:
                path = []
                cur = t
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path
            
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)
        
        return None

    def greedy_extend_path(start, adj, n, max_len=None):
        """
        Greedily extend a path from start, always taking the least-used edge first.
        Returns a simple path.
        """
        if max_len is None:
            max_len = n - 1
        
        path = [start]
        visited = {start}
        current = start
        
        for _ in range(max_len):
            # Find neighbor with fewest remaining edges
            best_neighbor = None
            best_degree = float('inf')
            
            for v in adj[current]:
                if v not in visited:
                    deg = len(adj[v])
                    if deg < best_degree:
                        best_degree = deg
                        best_neighbor = v
            
            if best_neighbor is None:
                break
            
            path.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        
        return path

    def decompose_with_euler(edges, n, time_budget=5):
        """
        Decompose using Eulerian paths/circuits.
        The minimum number of paths is max(1, odd_count // 2).
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Compute degrees
        degree = [0] * n
        for u, v in edge_set:
            degree[u] += 1
            degree[v] += 1
        
        # Find odd-degree vertices
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        
        # If all even, we can use one Eulerian circuit
        if not odd_verts:
            adj = build_adjacency(edge_set, n)
            # Find start vertex
            start = None
            for v in range(n):
                if adj[v]:
                    start = v
                    break
            if start is None:
                return []
            
            circuit = eulerian_circuit(adj, start)
            if circuit:
                return [circuit]
            else:
                # Fallback
                return [[u, v] for u, v in edge_set]
        
        # We need to pair up odd vertices
        # Try to find a good pairing using BFS distances
        # First, build adjacency
        adj = build_adjacency(edge_set, n)
        
        # Compute all-pairs shortest paths for odd vertices
        # For small graphs, this is fine
        odd_count = len(odd_verts)
        
        # Try several random pairings
        best_paths = None
        best_count = float('inf')
        
        rng = random.Random(42)
        max_attempts = min(50, int(time_budget * 10))
        
        for attempt in range(max_attempts):
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Create a random pairing
            verts = odd_verts[:]
            rng.shuffle(verts)
            pairs = []
            for i in range(0, len(verts), 2):
                if i + 1 < len(verts):
                    pairs.append((verts[i], verts[i + 1]))
            
            # Try to find edge-disjoint paths for this pairing
            paths = []
            remaining = set(edge_set)
            success = True
            
            # Sort pairs by distance (heuristic: shorter paths first)
            pair_dists = []
            for s, t in pairs:
                # Quick BFS distance
                dist = 0
                if s == t:
                    dist = 0
                else:
                    # Simple BFS for distance
                    visited = {s}
                    queue = deque([(s, 0)])
                    found = False
                    while queue:
                        u, d = queue.popleft()
                        if u == t:
                            dist = d
                            found = True
                            break
                        for v in adj[u]:
                            if v not in visited:
                                visited.add(v)
                                queue.append((v, d + 1))
                    if not found:
                        dist = n  # Large value
                pair_dists.append((dist, s, t))
            
            pair_dists.sort()
            pairs = [(s, t) for _, s, t in pair_dists]
            
            for s, t in pairs:
                path = find_path_bfs(s, t, {v: adj[v] for v in adj}, n)
                if path is None:
                    success = False
                    break
                paths.append(path)
                for i in range(len(path) - 1):
                    e = tuple(sorted((path[i], path[i + 1])))
                    remaining.discard(e)
            
            if success and not remaining:
                count = len(paths)
                if count < best_count:
                    best_count = count
                    best_paths = paths[:]
                    min_possible = odd_count // 2
                    if count == min_possible:
                        break
        
        if best_paths is not None:
            return best_paths
        
        # Fallback: greedy decomposition
        return greedy_decompose(edge_set, n)

    def greedy_decompose(edge_set, n):
        """
        Greedy decomposition: repeatedly find a long path, remove its edges.
        """
        if not edge_set:
            return []
        
        remaining = set(edge_set)
        paths = []
        
        # Build adjacency dynamically
        adj = build_adjacency(remaining, n)
        
        while remaining:
            # Find a vertex with highest degree to start from
            start = max(adj.keys(), key=lambda v: len(adj[v])) if adj else None
            if start is None or not adj[start]:
                # Take a single edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.remove(e)
                adj = build_adjacency(remaining, n)
                continue
            
            # Greedily extend
            path = greedy_extend_path(start, adj, n)
            
            if len(path) < 2:
                # Take a single edge
                e = next(iter(remaining))
                paths.append(list(e))
                remaining.remove(e)
                adj = build_adjacency(remaining, n)
                continue
            
            # Remove edges used in this path
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i + 1])))
                if e in remaining:
                    remaining.remove(e)
            
            paths.append(path)
            adj = build_adjacency(remaining, n)
        
        return paths

    def smart_decompose(edges, n, time_budget=10):
        """
        Combined approach: use Eulerian decomposition if possible, else greedy.
        """
        if not edges:
            return []
        
        edge_set = set(tuple(sorted(e)) for e in edges)
        
        # Compute degrees
        degree = [0] * n
        for u, v in edge_set:
            degree[u] += 1
            degree[v] += 1
        
        odd_verts = [v for v in range(n) if degree[v] % 2 == 1]
        min_paths = max(1, len(odd_verts) // 2)
        
        # Try Eulerian-based decomposition
        result = decompose_with_euler(edges, n, time_budget)
        
        # If result is good (close to minimum), return it
        if result and len(result) <= min_paths + 2:
            return result
        
        # Otherwise, try greedy
        greedy_result = greedy_decompose(edge_set, n)
        
        # Return the better one
        if len(greedy_result) < len(result):
            return greedy_result
        return result

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() - start_time > DEADLINE:
            # Fallback: return remaining as single edges
            n, p, seed = spec
            edges = instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        
        # Allocate time budget per instance
        elapsed = time.time() - start_time
        remaining_time = DEADLINE - elapsed
        per_instance_time = min(remaining_time / 3, 20)
        
        # Try smart decomposition
        paths = smart_decompose(edges, n, time_budget=per_instance_time)
        
        results.append(paths)
    
    return results

# EVOLVE-BLOCK-END
