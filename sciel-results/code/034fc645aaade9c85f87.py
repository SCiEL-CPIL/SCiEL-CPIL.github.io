import random
import time

def construct():
    deadline = time.time() + 110  # 10s margin for safety
    n = 20
    num_colors = 20
    
    best_edges = None
    best_count = 0
    
    # Pre-compute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    def greedy_build(seed):
        rng = random.Random(seed)
        shuffled_edges = all_edges[:]
        rng.shuffle(shuffled_edges)
        
        # adj[u][v] = color if edge exists, -1 otherwise
        adj = [[-1] * n for _ in range(n)]
        # neighbors[u] = set of v such that adj[u][v] != -1
        neighbors = [set() for _ in range(n)]
        edges = []
        
        for (u, v) in shuffled_edges:
            # Try each color
            placed = False
            # Randomize color order to avoid bias?
            colors = list(range(num_colors))
            rng.shuffle(colors)
            
            for c in colors:
                # Check if adding (u, v, c) creates any non-rainbow C4
                ok = True
                # A 4-cycle involving (u,v) must be u-a-b-v-u
                # where (u,a), (a,b), (b,v) are existing edges.
                # a is in neighbors[u], b is in neighbors[v]
                
                # Optimization: Iterate over smaller neighbor set?
                # But we need pairs (a,b).
                
                for a in neighbors[u]:
                    if a == v:
                        continue
                    # If (u,a) is edge, color is adj[u][a]
                    ca = adj[u][a]
                    for b in neighbors[v]:
                        if b == u or b == a:
                            continue
                        # Check if (a,b) is an edge
                        if adj[a][b] != -1:
                            cb = adj[a][b]
                            cv = adj[b][v]
                            # Colors: ca (u-a), cb (a-b), cv (b-v), c (v-u)
                            if ca == c or cb == c or cv == c or ca == cb or ca == cv or cb == cv:
                                # We need 4 distinct: {ca, cb, cv, c}
                                # If any pair is equal, it's not distinct.
                                # Actually, len(set) < 4 is the condition.
                                # Checking pairs is faster than set creation?
                                # Let's use set for clarity and speed in Python 3
                                if len({ca, cb, cv, c}) < 4:
                                    ok = False
                                    break
                    if not ok:
                        break
                
                if ok:
                    # Place edge
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                    placed = True
                    break
            
            # If no color works, skip edge
        
        return edges

    # Multiple restarts
    num_restarts = 1000
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            
            # Optional: Verify best result occasionally?
            # The greedy check is strict, so it should be valid.
            # But if there's a bug, we might return invalid.
            # Let's assume the check is correct.

    if best_edges is None:
        # Fallback: Star
        best_edges = [[0, v, 0] for v in range(1, 20)]
    
    return best_edges
