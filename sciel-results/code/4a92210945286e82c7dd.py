import random
import time
from itertools import combinations

# Pre-compute all 8-vertex subsets as tuples
SUBSETS_8 = list(combinations(range(16), 8))

def is_q3_fast(vertices, adj):
    """
    vertices: tuple of 8 ints
    adj: list of lists of bools (16x16)
    """
    # Count edges
    ec = 0
    # Unpack for speed?
    v0, v1, v2, v3, v4, v5, v6, v7 = vertices
    
    # Adjacency checks
    # We need to check 28 pairs
    # Pairs:
    # (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7)
    # (1,2), (1,3), (1,4), (1,5), (1,6), (1,7)
    # ...
    
    # Instead of hardcoding, loop is fine if we break early.
    for i in range(8):
        ui = vertices[i]
        for j in range(i+1, 8):
            uj = vertices[j]
            if adj[ui][uj]:
                ec += 1
                if ec > 12:
                    return False
    
    if ec != 12:
        return False
        
    # Check degrees
    # For each vertex, count neighbors in the subset
    for i in range(8):
        ui = vertices[i]
        deg = 0
        for j in range(8):
            if i == j: continue
            uj = vertices[j]
            if adj[ui][uj]:
                deg += 1
        if deg != 3:
            return False
            
    # Check bipartite
    # BFS
    color = [-1] * 16 # Reuse? No, local scope.
    # Map vertex index in tuple to color?
    # Simpler: use a dict or array indexed by vertex id.
    # Since vertices are 0..15, we can use an array of size 16.
    # But we only care about these 8.
    # Let's use a local dict for clarity, or array.
    col = [0] * 16 # 0: unvisited, 1: red, 2: blue
    # Actually, let's just use a set-based BFS on the induced subgraph.
    
    # Find a start vertex
    start = vertices[0]
    col[start] = 1
    queue = [start]
    while queue:
        u = queue.pop(0)
        cu = col[u]
        # Neighbors of u in the subset
        for v in vertices:
            if u == v: continue
            if adj[u][v]:
                if col[v] == 0:
                    col[v] = 3 - cu
                    queue.append(v)
                elif col[v] == cu:
                    return False
    return True

def find_q3(adj):
    for verts in SUBSETS_8:
        if is_q3_fast(verts, adj):
            # Return edges
            edges = []
            for i in range(8):
                for j in range(i+1, 8):
                    if adj[verts[i]][verts[j]]:
                        edges.append((verts[i], verts[j]))
            return edges
    return None

def construct():
    random.seed(42)
    adj = [[False]*16 for _ in range(16)]
    edge_set = set()
    
    # Initialize with p=0.9
    for i in range(16):
        for j in range(i+1, 16):
            if random.random() < 0.9:
                adj[i][j] = True
                adj[j][i] = True
                edge_set.add((i, j))
                
    # Remove edges to make Q3-free
    # Limit iterations to avoid infinite loop if bug
    for _ in range(1000):
        q3_edges = find_q3(adj)
        if q3_edges is None:
            break
        e = random.choice(q3_edges)
        u, v = e
        adj[u][v] = False
        adj[v][u] = False
        edge_set.discard((u, v))
        
    # Local search: add edges
    start_time = time.time()
    timeout = 110.0
    
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    missing_edges = [e for e in all_edges if e not in edge_set]
    
    while time.time() - start_time < timeout:
        if not missing_edges:
            break
        
        idx = random.randint(0, len(missing_edges) - 1)
        u, v = missing_edges[idx]
        
        adj[u][v] = True
        adj[v][u] = True
        
        q3_edges = find_q3(adj)
        if q3_edges is not None:
            adj[u][v] = False
            adj[v][u] = False
        else:
            missing_edges.pop(idx)
            edge_set.add((u, v))
            
    return [list(e) for e in edge_set]
