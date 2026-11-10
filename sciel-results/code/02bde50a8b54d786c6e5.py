"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_paths(edges):
        """
        Find an edge-disjoint path decomposition minimizing the number of paths.
        Strategy: repeatedly find a maximum matching in the remaining graph,
        then extend matched edges into maximal paths. This uses at most
        ceil(m / max_match_size) iterations but typically much fewer because
        each path covers multiple edges.
        
        Better strategy: Use a greedy approach where we keep finding the longest
        possible path in the remaining graph, removing its edges each time.
        """
        if not edges:
            return []
        
        # Build adjacency
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        
        paths = []
        
        while remaining:
            if time.time() > DEADLINE:
                # Fallback: each remaining edge is its own path
                for e in sorted(remaining):
                    paths.append(list(e))
                return paths
            
            # Find the longest simple path in the remaining graph using DFS with pruning
            # For small graphs (n<=30, ~100 edges), this is feasible
            # Try to find a long path starting from each vertex
            best_path = None
            best_len = 0
            
            # Get vertices that still have edges
            active_vertices = set()
            for u, v in remaining:
                active_vertices.add(u)
                active_vertices.add(v)
            
            # Try DFS from each active vertex to find longest path
            for start in sorted(active_vertices):
                # DFS to find longest simple path
                stack = [(start, [start], {start}, set())]
                while stack:
                    if time.time() > DEADLINE:
                        break
                    node, path, visited, used_edges = stack.pop()
                    if len(path) > best_len:
                        best_len = len(path)
                        best_path = list(path)
                    
                    for neighbor in sorted(adj.get(node, [])):
                        edge = tuple(sorted((node, neighbor)))
                        if edge in used_edges or edge not in remaining:
                            continue
                        if neighbor in visited:
                            continue
                        new_path = path + [neighbor]
                        new_visited = visited | {neighbor}
                        new_used = used_edges | {edge}
                        stack.append((neighbor, new_path, new_visited, new_used))
            
            if best_path is None:
                # Shouldn't happen, but fallback
                for e in sorted(remaining):
                    paths.append(list(e))
                return paths
            
            # Remove edges of best_path from remaining
            for i in range(len(best_path) - 1):
                e = tuple(sorted((best_path[i], best_path[i+1])))
                remaining.discard(e)
                adj[best_path[i]].discard(best_path[i+1])
                adj[best_path[i+1]].discard(best_path[i])
            
            paths.append(best_path)
        
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        if time.time() > DEADLINE:
            # Fallback for remaining instances
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        paths = decompose_paths(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
