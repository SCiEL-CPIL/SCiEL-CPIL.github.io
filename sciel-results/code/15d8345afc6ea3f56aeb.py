import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    time_limit = 110.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def get_odd_vertices(adj, n):
        odd = []
        for i in range(n):
            if len(adj[i]) % 2 == 1:
                odd.append(i)
        return odd

    def find_simple_path(start, end, adj, remaining_edges, visited_nodes=None):
        # DFS to find a simple path from start to end using only remaining_edges
        if visited_nodes is None:
            visited_nodes = set()
        
        if start == end:
            return [start]
            
        visited_nodes.add(start)
        
        # Get neighbors in remaining graph
        neighbors = []
        for e in remaining_edges:
            if start in e:
                other = e[0] if e[1] == start else e[1]
                if other not in visited_nodes:
                    neighbors.append(other)
        
        # Heuristic: sort neighbors by degree or distance?
        # For now, just iterate
        for nb in neighbors:
            # Remove edge start-nb
            edge = tuple(sorted((start, nb)))
            if edge not in remaining_edges:
                continue
            remaining_edges.remove(edge)
            
            path = find_simple_path(nb, end, adj, remaining_edges, visited_nodes)
            if path:
                return [start] + path
            else:
                # Backtrack
                remaining_edges.add(edge)
        
        visited_nodes.remove(start)
        return None

    def decompose_with_matching(edges, n, odd_verts):
        if not odd_verts:
            return None # Handle separately
            
        k = len(odd_verts) // 2
        # Random pairing
        odds = odd_verts[:]
        random.shuffle(odds)
        pairs = [(odds[i], odds[i + k]) for i in range(k)]
        
        # We need to find edge-disjoint paths for these pairs
        # This is a constraint satisfaction problem.
        # We will use backtracking on the pairs.
        
        # To make it faster, we can try to find paths sequentially.
        # But if we pick a path that blocks a later pair, we fail.
        # So we need to backtrack on path choices too?
        # That might be too slow.
        
        # Alternative: Just try to find *any* simple path for each pair in sequence.
        # If it fails, try a different pairing.
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining = set(edges)
        paths = []
        
        # Shuffle pairs to try different orders?
        # Actually, the order of pairs doesn't matter for the final set, but matters for the greedy search.
        
        for u, v in pairs:
            path = find_simple_path(u, v, adj, remaining)
            if path is None:
                return None # Failed
            paths.append(path)
            
        return paths

    def decompose_graph(edges):
        n = max(max(e) for e in edges) + 1 if edges else 0
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        odd_verts = get_odd_vertices(adj, n)
        
        # Try matching approach
        # Limit attempts to avoid timeout
        max_attempts = 1000
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            paths = decompose_with_matching(edges, n, odd_verts)
            if paths:
                return paths
                
        # Fallback: Greedy Longest Path
        remaining = set(edges)
        paths = []
        while remaining:
            if time.time() - start_time > time_limit:
                # Add remaining as single edges
                for e in remaining:
                    paths.append(list(e))
                break
            
            # Find a start node
            # Pick node with highest degree in remaining
            degs = {v: 0 for v in range(n)}
            for u, v in remaining:
                degs[u] += 1
                degs[v] += 1
            
            start = max(degs.keys(), key=lambda x: degs[x])
            
            # Greedy walk
            path = [start]
            visited = {start}
            while True:
                cur = path[-1]
                neighbors = []
                for nb in adj[cur]:
                    e = tuple(sorted((cur, nb)))
                    if e in remaining and nb not in visited:
                        neighbors.append(nb)
                        
                if not neighbors:
                    break
                
                # Pick neighbor with highest degree
                neighbors.sort(key=lambda x: degs[x], reverse=True)
                next_node = neighbors[0]
                
                path.append(next_node)
                visited.add(next_node)
                e = tuple(sorted((cur, next_node)))
                remaining.remove(e)
                
            if len(path) > 1:
                paths.append(path)
            else:
                # Should not happen if remaining is not empty and connected component has edges
                # But if start is isolated? No, degs[start] > 0.
                # If all neighbors are visited, we break.
                # So path length is at least 2?
                # If start has a neighbor, we take it.
                # So len(path) >= 2.
                pass
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
    return results
