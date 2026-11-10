import sys
from random import Random
import time

def construct():
    """
    Decomposes three fixed graphs into few cycles and edges.
    Uses a greedy approach: repeatedly find the longest cycle in the remaining graph,
    remove its edges, and add it to the decomposition. Remaining edges are added as single pieces.
    """
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n):
        """
        Finds the longest cycle in the graph represented by adj (list of sets).
        Uses a heuristic DFS from each node.
        Returns a list of vertices in the cycle, or None if no cycle exists.
        """
        best_cycle = None
        best_len = 0
        
        # To speed up and find longer cycles, we can try starting from high-degree nodes
        # or just iterate all. n is small.
        for start in range(n):
            if not adj[start]:
                continue
            
            # DFS state
            # path: list of nodes in current path
            # on_path: set of nodes in current path for O(1) check
            # visited: set of nodes visited in this DFS branch? 
            # Actually, standard DFS for cycles:
            # We need to avoid re-visiting nodes in the *current* path to form a simple cycle.
            # We don't need to avoid visiting nodes visited in *other* branches of the DFS tree 
            # if we are looking for *any* cycle, but for *longest* cycle, 
            # simple DFS might miss some. However, for random graphs, it's usually good enough.
            
            # Let's use a recursive DFS that tracks the current path.
            # To prevent exponential blowup, we limit the depth? No, n=30 is small.
            # But worst case is exponential. 
            # Given the sparse nature, it should be fine.
            
            def dfs(u, path, on_path):
                nonlocal best_cycle, best_len
                for v in adj[u]:
                    if v in on_path:
                        # Found a cycle
                        # The cycle is path[path.index(v):] + [v]
                        idx = path.index(v)
                        cycle = path[idx:]
                        length = len(cycle)
                        if length > best_len:
                            best_len = length
                            best_cycle = list(cycle)
                    else:
                        # Explore v
                        on_path.add(v)
                        path.append(v)
                        dfs(v, path, on_path)
                        path.pop()
                        on_path.remove(v)
            
            dfs(start, [start], {start})
            
        return best_cycle

    def decompose(edges, n):
        """
        Greedily decomposes the graph into cycles and edges.
        """
        # Build initial adjacency
        # adj[u] is a set of neighbors
        adj = [set() for _ in range(n)]
        remaining_edges = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            remaining_edges.add((u, v))
        
        pieces = []
        
        while remaining_edges:
            # Find longest cycle in the subgraph of remaining_edges
            # We need to pass the current adjacency to find_longest_cycle
            # But find_longest_cycle uses adj which is modified?
            # Let's make a copy of adj for the current state or pass remaining_edges.
            
            # It's more efficient to maintain adj and remove edges.
            # However, find_longest_cycle as written above takes adj.
            # We need to ensure it only uses edges in remaining_edges.
            # Since we remove edges from adj as we go, adj reflects the remaining graph.
            
            cycle = find_longest_cycle(adj, n)
            
            if cycle and len(cycle) >= 3:
                # Add cycle to pieces
                pieces.append(cycle)
                
                # Remove edges of this cycle from remaining_edges and adj
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                # No more cycles. Add all remaining edges as single pieces.
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
        
    return results

# EVOLVE-BLOCK-START
def construct():
    """
    Decomposes three fixed graphs into few cycles and edges.
    Uses a greedy approach: repeatedly find the longest cycle in the remaining graph,
    remove its edges, and add it to the decomposition. Remaining edges are added as single pieces.
    """
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_longest_cycle(adj, n):
        """
        Finds the longest cycle in the graph represented by adj (list of sets).
        Uses a heuristic DFS from each node.
        Returns a list of vertices in the cycle, or None if no cycle exists.
        """
        best_cycle = None
        best_len = 0
        
        for start in range(n):
            if not adj[start]:
                continue
            
            def dfs(u, path, on_path):
                nonlocal best_cycle, best_len
                for v in adj[u]:
                    if v in on_path:
                        # Found a cycle
                        idx = path.index(v)
                        cycle = path[idx:]
                        length = len(cycle)
                        if length > best_len:
                            best_len = length
                            best_cycle = list(cycle)
                    else:
                        on_path.add(v)
                        path.append(v)
                        dfs(v, path, on_path)
                        path.pop()
                        on_path.remove(v)
            
            dfs(start, [start], {start})
            
        return best_cycle

    def decompose(edges, n):
        adj = [set() for _ in range(n)]
        remaining_edges = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            remaining_edges.add((u, v))
        
        pieces = []
        
        while remaining_edges:
            cycle = find_longest_cycle(adj, n)
            
            if cycle and len(cycle) >= 3:
                pieces.append(cycle)
                
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining_edges:
                        remaining_edges.remove(edge)
                        adj[u].remove(v)
                        adj[v].remove(u)
            else:
                for u, v in remaining_edges:
                    pieces.append([u, v])
                break
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        n, p, seed = spec
        edges = instance(n, p, seed)
        pieces = decompose(edges, n)
        results.append(pieces)
        
    return results
# EVOLVE-BLOCK-END
