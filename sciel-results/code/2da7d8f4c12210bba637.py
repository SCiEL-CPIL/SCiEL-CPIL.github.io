import random
import time
import itertools

def construct():
    n = 20
    start_time = time.time()
    deadline = 110.0  # 10 seconds margin
    
    best_edges = 0
    best_adj = None
    
    # We will run multiple restarts
    while time.time() < deadline:
        # Initialize with complete graph
        adj = [[True] * n for _ in range(n)]
        for i in range(n):
            adj[i][i] = False
        
        # Bitmasks for fast intersection
        masks = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    masks[i] |= (1 << j)
        
        # Greedy removal
        # We iterate until no violations or time runs out
        # To avoid infinite loops if stuck (shouldn't happen as we remove edges),
        # we just check time.
        
        while time.time() < deadline:
            # Find a violation
            violation_found = False
            # Randomize the order of triples to escape local minima
            triples = list(itertools.combinations(range(n), 3))
            random.shuffle(triples)
            
            for u, v, w in triples:
                common = masks[u] & masks[v] & masks[w]
                # Check if common has at least 3 bits set
                # Python 3.10+ has int.bit_count(), but let's be safe or use bin(x).count('1')
                # bin(x).count('1') is fast enough for 20 bits.
                if bin(common).count('1') >= 3:
                    violation_found = True
                    # Get the list of common neighbors
                    common_neigh = [i for i in range(n) if (common >> i) & 1]
                    # Pick 3 from common_neigh
                    # random.sample is good
                    x, y, z = random.sample(common_neigh, 3)
                    
                    # The edges to break are between {u,v,w} and {x,y,z}
                    # There are 9 edges.
                    # Pick one to remove.
                    # Heuristic: Remove edge incident to vertex with highest degree?
                    # Or just random.
                    # Let's try random first.
                    edge_to_remove = random.choice([
                        (u, x), (u, y), (u, z),
                        (v, x), (v, y), (v, z),
                        (w, x), (w, y), (w, z)
                    ])
                    
                    a, b = edge_to_remove
                    if adj[a][b]:
                        adj[a][b] = False
                        adj[b][a] = False
                        masks[a] &= ~(1 << b)
                        masks[b] &= ~(1 << a)
                    break # Restart the triple scan after a removal
            
            if not violation_found:
                # Graph is valid
                edge_count = sum(1 for i in range(n) for j in range(i+1, n) if adj[i][j])
                if edge_count > best_edges:
                    best_edges = edge_count
                    best_adj = [row[:] for row in adj]
                break # Move to next restart
        
        # If we broke out of inner loop due to time, check if current is valid
        if time.time() >= deadline:
            break

    # Construct result from best_adj
    if best_adj is None:
        # Fallback to seed
        return [[a, b] for a in range(2) for b in range(2, 20)]
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
