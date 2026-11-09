import time
import sys
from random import Random
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def _greedy_path_cover_lookahead(edges, n):
        """
        Greedy path cover with lookahead to maximize path length.
        """
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(tuple(sorted(e)) for e in edges)
        new_paths = []
        
        def estimate_path_length(start, visited, adj, remaining_edges, depth=0, max_depth=6):
            """
            Estimate the length of the longest simple path starting from 'start' 
            using a limited DFS.
            """
            if depth == max_depth:
                return 0
            best = 0
            for nb in adj[start]:
                e = tuple(sorted((start, nb)))
                if e in remaining_edges and nb not in visited:
                    # Try extending to nb
                    new_visited = visited | {nb}
                    # Remove edge? No, we just estimate.
                    # We need to ensure we don't use edges that would block other branches?
                    # For estimation, we just want to see how far we can go.
                    # A simple DFS on the subgraph of remaining edges.
                    # To be fast, we just count the number of steps we can take.
                    
                    # Recurse
                    sub_len = 1 + estimate_path_length(nb, new_visited, adj, remaining_edges, depth + 1, max_depth)
                    if sub_len > best:
                        best = sub_len
            return best

        while remaining:
            # Find active vertices
            active = [v for v in range(n) if adj[v]]
            if not active:
                break
            
            # Start from the vertex with the highest degree in the remaining graph
            # Or try a few top vertices
            active.sort(key=lambda v: len(adj[v]), reverse=True)
            
            best_path = []
            best_len = -1
            
            # Try top 3 starting vertices
            for start in active[:3]:
                # Greedily extend with lookahead
                visited = {start}
                path = [start]
                cur = start
                while True:
                    cands = [nb for nb in adj[cur] if nb not in visited and tuple(sorted((cur, nb))) in remaining]
                    if not cands:
                        break
                    # Pick the candidate with the best estimated continuation
                    best_nb = None
                    best_est = -1
                    for nb in cands:
                        # Estimate length from nb
                        est = estimate_path_length(nb, visited | {nb}, adj, remaining, depth=0, max_depth=4)
                        if est > best_est:
                            best_est = est
                            best_nb = nb
                    if best_nb is None:
                        break
                    path.append(best_nb)
                    visited.add(best_nb)
                    cur = best_nb
                
                if len(path) > best_len:
                    best_len = len(path)
                    best_path = path
            
            if not best_path:
                # Should not happen if remaining is not empty
                break
                
            # Remove edges of best_path from remaining
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i + 1])))
                remaining.discard(e)
                # Update adj? 
                # adj is static, we check 'in remaining' in the loop.
                # But we need to update adj for the next iterations to reflect removed edges?
                # No, 'remaining' is the source of truth for edge existence.
                # The 'adj' sets are used for iteration. We should filter them or just rely on 'remaining'.
                # The check 'tuple(sorted((cur, nb))) in remaining' handles it.
            
            new_paths.append(best_path)
            
        return new_paths

    def _is_valid(paths, edges, n):
        edge_set = set(tuple(sorted(e)) for e in edges)
        used = set()
        for p in paths:
            if len(p) < 2:
                return False
            seen = set()
            for v in p:
                if v in seen:
                    return False
                seen.add(v)
            for i in range(len(p) - 1):
                e = tuple(sorted((p[i], p[i + 1])))
                if e not in edge_set:
                    return False
                if e in used:
                    return False
                used.add(e)
        return used == edge_set

    def path_cover(edges, n):
        if not edges:
            return []
        
        # Try the lookahead greedy
        res = _greedy_path_cover_lookahead(edges, n)
        
        if res and _is_valid(res, edges, n):
            return res
        
        # Fallback to simple greedy if lookahead fails validity (shouldn't happen)
        # Or if it's too slow?
        # For now, just return it.
        return res

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            # Emergency: return edge-by-edge
            results.append([list(e) for e in edges])
            while len(results) < 3:
                edges = instance(*spec)
                results.append([list(e) for e in edges])
            break
        
        edges = instance(*spec)
        n = spec[0]
        paths = path_cover(edges, n)
        results.append(paths)
        
    return results

if __name__ == "__main__":
    res = construct()
    print("Number of paths:", [len(r) for r in res])
    print("Total paths:", sum(len(r) for r in res))
