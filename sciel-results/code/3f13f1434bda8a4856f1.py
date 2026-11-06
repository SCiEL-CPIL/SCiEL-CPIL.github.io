import time
import numpy as np

def _build_polarity_graph():
    q = 5
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero becomes 1
                vec = [x, y, z]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], q - 2, q)
                        vec = [(v * inv) % q for v in vec]
                        break
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    n = len(points)
    assert n == 31
    
    # Adjacency matrix
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            u = np.array(points[i])
            v = np.array(points[j])
            # Dot product mod 5
            if (u @ v) % q == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                edges.append([i, j])
                
    return edges, adj

def _count_c4(adj):
    # Fast count
    # common neighbors
    common = adj @ adj
    # subtract diagonal (self-loops in common count? no, adj has 0 diag)
    # common[i,j] is number of common neighbors
    # C4s = sum_{i<j} C(common[i,j], 2)
    c = common - adj # remove edges that are common neighbors? No.
    # Actually, C(common[i,j], 2) counts pairs of common neighbors.
    # Each pair of common neighbors (a,b) with i,j forms a C4 i-a-j-b-i.
    # This counts each C4 exactly once if we sum over i<j?
    # Yes.
    iu = np.triu_indices(n, k=1)
    vals = common[iu]
    return int(np.sum(vals * (vals - 1) // 2))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    edges_90, adj = _build_polarity_graph()
    
    # Verify
    if len(edges_90) != 90:
        # Fallback: Random graph?
        pass
        
    c4_base = _count_c4(adj)
    # Should be 0
    
    # Find best edge to add
    best_edge = None
    min_common = float('inf')
    
    for i in range(31):
        for j in range(i + 1, 31):
            if adj[i, j] == 0:
                common = int(np.dot(adj[i], adj[j]))
                if common < min_common:
                    min_common = common
                    best_edge = [i, j]
                    
    if best_edge is None:
        # Should not happen
        return edges_90[:91] if len(edges_90) >= 91 else edges_90 + [[0,1]]
        
    # Add the edge
    i, j = best_edge
    adj[i, j] = 1
    adj[j, i] = 1
    edges_91 = edges_90 + [best_edge]
    
    # Optional: Verify C4 count
    # c4_final = _count_c4(adj)
    # print(f"Added {best_edge}, Common Neighbors: {min_common}, Total C4s: {c4_final}")
    
    return edges_91
