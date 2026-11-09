"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    """
    Decompose three fixed random graphs into few cycles and edges.
    Returns three lists of pieces, one per graph.
    A piece is either [u, v] (a single edge) or [v1, v2, ..., vk] with k>=3 (a cycle).
    """
    deadline = time.time() + 110  # safety margin before 120s limit

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add((min(u, v), max(u, v)))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def decompose(edges):
        """
        Decompose a graph into cycles and edges, minimizing the number of pieces.
        Strategy: repeatedly find and remove cycles (longer is better for minimizing count,
        but any cycle replaces k edges with 1 piece). We use a greedy approach:
        1. Find cycles using DFS-based cycle detection, preferring longer cycles.
        2. Remove those edges.
        3. Remaining edges are single-edge pieces.
        
        To maximize savings, we want to find as many cycles as possible, especially longer ones.
        """
        if not edges:
            return []
        
        # Build adjacency structure
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        remaining_edges = set(edges)
        
        # We'll iteratively find cycles and remove them
        # Use a modified DFS to find cycles, preferring to find longer cycles
        
        # First, let's try to find many cycles by repeatedly doing DFS
        # For efficiency, we'll limit the number of iterations
        
        max_iterations = 5000
        iter_count = 0
        
        while remaining_edges and iter_count < max_iterations:
            if time.time() > deadline - 5:
                break
            iter_count += 1
            
            # Find a cycle in the remaining graph
            cycle = _find_cycle(adj, remaining_edges)
            if cycle is None:
                break
            
            # Remove the cycle's edges from remaining_edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = (min(u, v), max(u, v))
                remaining_edges.discard(e)
            
            # Update adjacency
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                adj[u].discard(v)
                adj[v].discard(u)
            
            pieces.append(cycle)
        
        # Remaining edges are single-edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces

    def _find_cycle(adj, remaining_edges):
        """
        Find a cycle in the graph. We use a DFS-based approach.
        Returns a list of vertices forming the cycle, or None if no cycle exists.
        """
        # Find all vertices that still have edges
        active_vertices = set()
        for u, v in remaining_edges:
            active_vertices.add(u)
            active_vertices.add(v)
        
        if not active_vertices:
            return None
        
        # Try to find a cycle using DFS
        # We'll do a BFS/DFS from each unvisited vertex
        
        visited = set()
        parent = {}
        
        for start in sorted(active_vertices):
            if start in visited:
                continue
            
            # DFS
            stack = [(start, -1, [start])]
            visited.add(start)
            
            while stack:
                node, par, path = stack.pop()
                
                # Check neighbors
                neighbors = sorted(adj[node], reverse=True)  # Try higher-degree first for longer cycles
                for nb in neighbors:
                    edge = (min(node, nb), max(node, nb))
                    if edge not in remaining_edges:
                        continue
                    
                    if nb == par:
                        continue
                    
                    if nb in visited:
                        # Found a cycle: from nb to node in the current path, plus the edge node-nb
                        # Extract the cycle from path
                        idx = path.index(nb)
                        cycle = path[idx:] + [nb]
                        # Verify it's a valid cycle with distinct vertices (except first/last)
                        if len(cycle) >= 3:
                            # Check all vertices are distinct except the closing
                            inner = cycle[:-1]
                            if len(set(inner)) == len(inner):
                                return cycle
                    else:
                        visited.add(nb)
                        parent[nb] = node
                        stack.append((nb, node, path + [nb]))
        
        return None

    # Generate the three instances and decompose them
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([[u, v] for u, v in edges])
            continue
        
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
