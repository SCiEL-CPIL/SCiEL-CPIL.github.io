"""Decompose three fixed Erdős-Gallai-style graphs into few cycles + edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, n, time_limit=12.0):
    """
    Fast decomposition: 
    1. Repeatedly find a cycle in the remaining graph using DFS.
    2. Remove the cycle edges.
    3. When no cycles remain, take all remaining edges as single pieces.
    """
    start_time = time.time()
    remaining = set(edges)
    pieces = []
    
    if not remaining:
        return pieces

    # Helper to find a cycle in the current remaining graph
    def find_cycle():
        """
        Finds any cycle in the graph defined by 'remaining'.
        Returns list of vertices in the cycle, or None.
        Uses DFS. To speed up, we stop as soon as we find one.
        """
        # Build adjacency from remaining edges
        adj = defaultdict(list)
        for u, v in remaining:
            adj[u].append(v)
            adj[v].append(u)
        
        visited = [False] * n
        parent = [-1] * n
        cycle = []
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    parent[v] = u
                    if dfs(v):
                        return True
                elif v != parent[u]:
                    # Found a back edge to an ancestor
                    curr = u
                    cycle.append(v)
                    while curr != v:
                        cycle.append(curr)
                        curr = parent[curr]
                    return True
            return False

        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    return cycle
        return None

    while remaining:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a cycle
        cycle = find_cycle()
        
        if cycle:
            # Remove edges of this cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = tuple(sorted((u, v)))
                remaining.discard(edge)
            pieces.append(cycle)
        else:
            # No cycle exists, so the remaining graph is a forest.
            # All remaining edges must be single pieces.
            for e in remaining:
                pieces.append(list(e))
            break
            
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for n, p, seed in specs:
        edges = _instance(n, p, seed)
        pieces = _decompose(edges, n, time_limit=38.0)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
