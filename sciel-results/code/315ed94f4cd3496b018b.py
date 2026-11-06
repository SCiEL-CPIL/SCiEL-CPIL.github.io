import random
import time

def construct():
    """
    Constructs a C6-free graph on 40 vertices with a high number of edges.
    Uses a randomized greedy algorithm with multiple restarts.
    """
    N = 40
    deadline = time.time() + 110  # 10s safety margin
    
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while time.time() < deadline:
        random.shuffle(all_edges)
        
        adj = [set() for _ in range(N)]
        edge_set = set()
        
        for u, v in all_edges:
            # Check if adding (u, v) creates a C6
            creates_cycle = False
            
            # A C6 involving (u,v) must have the form u-a-x-y-b-v
            # where a in N(u), b in N(v), x in N(a), y in N(b), and (x,y) is an edge.
            # We must ensure all vertices are distinct.
            
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b == u or b == a:
                        continue
                    # Check for path a-x-y-b of length 3
                    # x must be neighbor of a, y neighbor of b
                    # x, y distinct from u, v, a, b and each other
                    for x in adj[a]:
                        if x in (u, v, a, b):
                            continue
                        for y in adj[b]:
                            if y in (u, v, a, b, x):
                                continue
                            # Check if (x, y) is an edge
                            if (x, y) in edge_set or (y, x) in edge_set:
                                creates_cycle = True
                                break
                        if creates_cycle:
                            break
                    if creates_cycle:
                        break
                if creates_cycle:
                    break
            
            if not creates_cycle:
                adj[u].add(v)
                adj[v].add(u)
                edge_set.add((u, v))
        
        current_count = len(edge_set)
        if current_count > best_count:
            best_count = current_count
            best_edges = list(edge_set)
            
    # Return in the required format: list of [u, v]
    return [[u, v] for u, v in best_edges]
