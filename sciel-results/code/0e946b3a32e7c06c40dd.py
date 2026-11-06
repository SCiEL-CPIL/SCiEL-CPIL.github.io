import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        # We need to handle dynamic removal, so we use a dict of sets
        adj = {}
        for u, v in edges:
            if u not in adj: adj[u] = set()
            if v not in adj: adj[v] = set()
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        # Helper to get current degree
        def deg(v):
            return len(adj.get(v, set()))
            
        # Helper to remove a path's edges
        def remove_path(path):
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                adj[u].remove(v)
                adj[v].remove(u)
                # Optional cleanup if empty
                if not adj[u]: del adj[u]
                if not adj[v]: del adj[v]

        # Greedy path extension
        # Heuristic: Move to neighbor with lowest degree (to save hubs)
        def extend_path(start):
            if start not in adj or not adj[start]:
                return [start]
            path = [start]
            visited = {start}
            current = start
            while True:
                neighbors = [n for n in adj.get(current, set()) if n not in visited]
                if not neighbors:
                    break
                # Pick neighbor with lowest degree
                # If tie, pick random or arbitrary
                next_v = min(neighbors, key=lambda x: deg(x))
                path.append(next_v)
                visited.add(next_v)
                current = next_v
            return path

        while adj:
            # Check deadline
            if time.time() > deadline:
                # Fallback: return whatever we have plus remaining edges as singletons?
                # To be safe and valid, we should complete the decomposition if possible.
                # But if we run out of time, we might return an invalid solution.
                # Better to ensure the loop finishes. With N=30, it should be very fast.
                pass

            # Identify candidates for start
            # Prefer odd degree vertices, then high degree
            candidates = list(adj.keys())
            if not candidates:
                break
                
            # Sort candidates: odd degree first, then by degree descending
            candidates.sort(key=lambda v: (deg(v) % 2 == 0, -deg(v)))
            
            best_path = None
            best_len = -1
            
            # Try a subset of candidates to save time, or all if N is small
            # N is <= 30, so trying all is fine
            for start in candidates:
                if start not in adj: continue
                p = extend_path(start)
                if len(p) > best_len:
                    best_len = len(p)
                    best_path = p
                    
            if best_path is None or len(best_path) < 2:
                # If no path of length > 1 found, it means we have isolated vertices?
                # No, adj only contains vertices with degree > 0.
                # If best_path length is 1, it means start had no neighbors?
                # But adj[start] is not empty.
                # This case shouldn't happen if adj is built correctly.
                # However, if extend_path returns [start], it means no unvisited neighbors.
                # This implies start is isolated in the *remaining* graph?
                # But it's in adj.
                # Maybe the path was length 1 because all neighbors were visited?
                # No, visited starts with start.
                # If adj[start] is not empty, there is at least one neighbor.
                # So path length should be >= 2.
                # If this happens, force pick a single edge?
                # Let's just pick the first candidate and its first neighbor
                if best_path is None:
                     best_path = [candidates[0]]
                     if adj.get(best_path[0]):
                         best_path.append(list(adj[best_path[0]])[0])
                # If best_path is still length 1, something is wrong, break to avoid infinite loop
                if len(best_path) < 2:
                     # Force an edge
                     u = candidates[0]
                     v = list(adj[u])[0]
                     best_path = [u, v]
            
            paths.append(best_path)
            remove_path(best_path)
            
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
