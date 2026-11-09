"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict, deque

def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        from random import Random
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def greedy_path_cover(edges, time_limit):
        """Repeatedly extract the longest simple path from remaining edges."""
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        remaining = set(edges)
        paths = []
        
        while remaining:
            if time.time() > time_limit:
                # Fallback: each remaining edge as its own path
                for u, v in remaining:
                    paths.append([u, v])
                break
            # Find longest simple path via DFS/BFS heuristic
            # Try from each vertex, do DFS to find longest path
            best_path = None
            # Sort vertices by degree for heuristic
            verts = list(adj.keys())
            verts.sort(key=lambda v: len(adj[v]), reverse=True)
            
            for start in verts:
                if not any((start, v) in remaining or (v, start) in remaining for v in adj[start]):
                    continue
                # DFS to find longest simple path starting from start
                best = _dfs_longest(start, remaining, adj, time_limit)
                if best and (best_path is None or len(best) > len(best_path)):
                    best_path = best
                # Early stop if we found a very long path
                if best_path and len(best_path) > 2 * len(remaining) / max(1, len(verts)):
                    break
            
            if best_path is None or len(best_path) < 2:
                # Fallback
                for u, v in remaining:
                    paths.append([u, v])
                break
            
            paths.append(best_path)
            # Remove edges of this path
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                e = tuple(sorted((u, v)))
                remaining.discard(e)
                adj[u].discard(v)
                adj[v].discard(u)
        
        return paths

    def _dfs_longest(start, remaining, adj, time_limit):
        """DFS to find longest simple path from start using only remaining edges."""
        best = [start]
        
        def dfs(node, path, used_edges):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(path) > len(best):
                best = path[:]
            for nbr in list(adj[node]):
                e = tuple(sorted((node, nbr)))
                if e in remaining and nbr not in path:
                    path.append(nbr)
                    dfs(nbr, path, used_edges)
                    path.pop()
                    if time.time() > time_limit:
                        return
        
        dfs(start, [start], set())
        return best

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        # Per-graph time budget
        per_graph = min(DEADLINE - (time.time() - t0), 40.0)
        paths = greedy_path_cover(edges, time.time() + per_graph)
        results.append(paths)
        if time.time() > DEADLINE - 5:
            # Not enough time for remaining graphs; use trivial
            for spec2 in ((25, 0.3, 2), (30, 0.25, 3)):
                if spec2 != spec and not any(p == spec2 for p in []):
                    pass
            break
    
    # Fill in any missing graphs with trivial solution
    while len(results) < 3:
        spec = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))[len(results)]
        edges = instance(*spec)
        results.append([[u, v] for u, v in edges])
    
    return results[:3]
# EVOLVE-BLOCK-END
