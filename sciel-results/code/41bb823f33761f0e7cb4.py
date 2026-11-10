"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

def construct():
    # Deadline guard: must finish well before 150s wall-clock.
    start_time = time.time()
    DEADLINE = 130.0  # safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges_list):
        """
        Decompose a graph (given as list of (u,v) edges) into edge-disjoint
        cycles and single edges, minimizing the number of pieces.
        
        Strategy:
        1. Repeatedly find a cycle in the remaining edge set and extract it.
           Use DFS-based cycle finding, preferring longer cycles when possible.
        2. After no more cycles can be found, each remaining edge is a piece.
        
        To find longer cycles, we use a heuristic: at each step, try to find
        a cycle that uses as many edges as possible. We do this by:
        - Building adjacency lists
        - Using a DFS that tries to close a cycle back to the start
        - Among multiple possible cycles, prefer the longest one found
        """
        remaining = set(edges_list)
        if not remaining:
            return []
        
        pieces = []
        
        while remaining:
            if time.time() > DEADLINE:
                break
            
            # Build adjacency structure
            adj = {}
            for u, v in remaining:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
            
            # Find a cycle using DFS
            cycle = _find_longest_cycle(adj, remaining)
            
            if cycle:
                # Remove cycle edges from remaining
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining:
                        remaining.discard(e)
                pieces.append(cycle)
            else:
                # No more cycles; take one edge as a piece
                e = next(iter(remaining))
                remaining.discard(e)
                pieces.append([e[0], e[1]])
        
        return pieces

    def _find_longest_cycle(adj, remaining_edges):
        """
        Find a cycle in the graph. Uses a heuristic to find a reasonably long cycle.
        Returns a list of vertices forming the cycle, or None.
        """
        # Get all vertices that have at least one edge
        vertices = set()
        for e in remaining_edges:
            vertices.add(e[0])
            vertices.add(e[1])
        
        if not vertices:
            return None
        
        # Try to find a cycle starting from each vertex
        best_cycle = None
        best_len = 2  # need at least 3
        
        for start in sorted(vertices):
            if time.time() > DEADLINE:
                break
            # DFS to find a cycle back to start
            cycle = _dfs_find_cycle(adj, remaining_edges, start)
            if cycle and len(cycle) > best_len:
                best_cycle = cycle
                best_len = len(cycle)
                if best_len >= 8:  # Good enough, stop searching
                    break
        
        return best_cycle

    def _dfs_find_cycle(adj, remaining_edges, start):
        """
        DFS from start trying to find a path back to start (a cycle).
        Uses a greedy heuristic to extend paths.
        """
        # We'll do a DFS with backtracking, but limit depth for performance
        # For small graphs (n <= 30), this is fine.
        
        # Build a set of edges for quick lookup
        edge_set = remaining_edges
        
        # DFS: path from start, trying to return to start
        # Use iterative DFS with stack for efficiency
        # State: (current_vertex, path, visited_edges)
        
        best = None
        best_len = 2
        
        def dfs(current, path, used_edges):
            nonlocal best, best_len
            if time.time() > DEADLINE:
                return
            
            for neighbor in adj.get(current, []):
                e = tuple(sorted((current, neighbor)))
                if e not in used_edges and e in edge_set:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best = list(path)
                            best_len = len(path)
                        return  # Don't extend further from this branch
                    elif neighbor not in path:
                        # Extend the path
                        path.append(neighbor)
                        used_edges.add(e)
                        dfs(neighbor, path, used_edges)
                        path.pop()
                        used_edges.discard(e)
        
        dfs(start, [start], set())
        return best

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
