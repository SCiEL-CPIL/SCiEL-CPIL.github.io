import random
import time
from collections import defaultdict

def instance_graph(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_graph(edges, max_restarts=100, time_limit=10):
    """
    Decomposes a graph given by a list of edges into cycles and single edges.
    Returns a list of pieces. A piece is a list of vertices.
    If len(piece) == 2, it's an edge. If len(piece) >= 3, it's a cycle.
    """
    best_pieces = None
    best_count = float('inf')
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Deep copy edges for this run
        current_edges = set(edges)
        pieces = []
        
        while current_edges:
            # Pick a random edge
            u, v = random.choice(list(current_edges))
            current_edges.remove((u, v))
            
            # Try to find a path from u to v in the remaining graph
            # Build adjacency list for remaining edges
            adj = defaultdict(set)
            for a, b in current_edges:
                adj[a].add(b)
                adj[b].add(a)
                
            # DFS to find path from u to v
            # We want to find *a* path. To vary the length, we shuffle neighbors.
            path = find_path_randomized(u, v, adj)
            
            if path:
                # path is [u, ..., v]
                # The cycle is path + [u] (implied closing edge (u,v) was removed)
                # But wait, the edge (u,v) was removed. The cycle is formed by the path edges and the edge (u,v).
                # The piece is the sequence of vertices in the cycle.
                cycle_vertices = path
                # Ensure distinct vertices? The path from DFS is simple, so vertices are distinct.
                # Add the cycle as a piece
                pieces.append(cycle_vertices)
                
                # Remove all edges in the path from current_edges
                for i in range(len(cycle_vertices) - 1):
                    e1 = cycle_vertices[i]
                    e2 = cycle_vertices[i+1]
                    edge_to_remove = tuple(sorted((e1, e2)))
                    if edge_to_remove in current_edges:
                        current_edges.remove(edge_to_remove)
            else:
                # No path found, (u, v) is a bridge in the current context
                # Add as single edge piece
                pieces.append([u, v])
                
        # Evaluate this decomposition
        # Count pieces
        count = len(pieces)
        if count < best_count:
            best_count = count
            best_pieces = pieces
            
    return best_pieces

def find_path_randomized(start, end, adj):
    """
    Finds a path from start to end in the graph defined by adj.
    Uses randomized DFS to vary the path found.
    Returns list of nodes [start, ..., end] or None.
    """
    if start == end:
        return [start]
        
    visited = {start}
    stack = [[start]]
    
    # To make it more random, we can shuffle the order of nodes in the stack or neighbors.
    # Standard DFS with shuffled neighbors:
    
    def dfs(node, path):
        if node == end:
            return path[:]
        neighbors = list(adj[node])
        random.shuffle(neighbors)
        for nei in neighbors:
            if nei not in visited:
                visited.add(nei)
                path.append(nei)
                res = dfs(nei, path)
                if res:
                    return res
                path.pop()
                visited.remove(nei)
        return None
        
    return dfs(start, [start])

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    global_start = time.time()
    
    for spec in specs:
        edges = instance_graph(*spec)
        # Allocate time budget per graph
        # Total 150s limit. Let's use 40s per graph to be safe.
        pieces = decompose_graph(edges, max_restarts=200, time_limit=40)
        results.append(pieces)
        
    return results
