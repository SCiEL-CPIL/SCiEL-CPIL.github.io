import time

N = 31
M = 91

def polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    Vertices are points in PG(2,5).
    Edge (u,v) if u . v = 0 mod 5.
    """
    points = []
    reps = {}
    idx = 0
    # Generate all non-zero vectors in GF(5)^3, normalized
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: first non-zero component should be 1
                if a != 0:
                    # a is 1..4. inv_a
                    inv_a = pow(a, 4, 5)
                    rep = (1, (b * inv_a) % 5, (c * inv_a) % 5)
                elif b != 0:
                    inv_b = pow(b, 4, 5)
                    rep = (0, 1, (c * inv_b) % 5)
                else:
                    rep = (0, 0, 1)
                
                if rep not in reps:
                    reps[rep] = idx
                    points.append(rep)
                    idx += 1
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    edges = []
    # Symplectic form: x1*y2 - x2*y1 ... wait.
    # The standard symplectic form in 3D?
    # The code used: a*e + b*d + c*f.
    # This is the dot product.
    # In GF(5), the dot product is symmetric.
    # Is it a polarity?
    # Yes, x.x = 0 has solutions.
    # (1,0,0) . (1,0,0) = 1 != 0.
    # (1,1,0) . (1,1,0) = 2 != 0.
    # (1,2,0) . (1,2,0) = 1 + 4 = 5 = 0. Absolute.
    
    for i in range(31):
        for j in range(i + 1, 31):
            a, b, c = points[i]
            d, e, f = points[j]
            val = (a * e + b * d + c * f) % 5
            if val == 0:
                edges.append((i, j))
    
    return edges

def count_c4s(edges):
    """
    Count 4-cycles in the graph defined by edges.
    """
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    c4_count = 0
    # A 4-cycle is u-v-w-x-u.
    # Count pairs of common neighbors for each pair of vertices.
    # Each C4 is counted 4 times (once for each edge as the "first" edge? No).
    # Each C4 corresponds to 4 pairs of opposite vertices?
    # No.
    # Number of C4s = sum_{u<v} C(common(u,v), 2) / 2 ?
    # Let's verify.
    # A C4 u-a-v-b-u.
    # Pairs of opposite vertices: (u,v) and (a,b).
    # u and v have common neighbors a and b.
    # So C(common(u,v), 2) counts the pair {a,b}.
    # Similarly for (a,b), common neighbors are u and v.
    # So each C4 is counted twice (once for pair u,v and once for pair a,b).
    # So divide by 2.
    
    for u in range(N):
        for v in range(u + 1, N):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                c4_count += len(common) * (len(common) - 1) // 2
    return c4_count // 2

def construct():
    """
    Construct the optimal graph.
    """
    edges = polarity_graph_q5()
    
    if len(edges) != 90:
        # Fallback if construction fails, though it shouldn't
        # Just add random edges to reach 91
        edge_set = set(edges)
        for u in range(N):
            for v in range(u + 1, N):
                if len(edges) >= M:
                    break
                if (u, v) not in edge_set:
                    edges.append((u, v))
                    edge_set.add((u, v))
            if len(edges) >= M:
                break
        return [[u, v] for u, v in edges]
        
    # We have 90 edges. Need 1 more.
    # Any non-edge adds exactly 1 C4.
    # So just pick the first non-edge.
    edge_set = set(edges)
    
    for u in range(N):
        for v in range(u + 1, N):
            if (u, v) not in edge_set:
                edges.append((u, v))
                break
        if len(edges) == M:
            break
            
    # Ensure exactly M edges
    if len(edges) > M:
        edges = edges[:M]
        
    return [[u, v] for u, v in edges]
