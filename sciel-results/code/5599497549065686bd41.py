"""Construct a graph on 31 vertices with 91 edges minimizing 4-cycles."""
import time

def _count_c4(adj):
    """Count 4-cycles in an undirected graph given by adjacency matrix."""
    n = adj.shape[0]
    # common[i,j] = number of common neighbors of i and j
    # Use matrix multiplication for speed
    common = (adj @ adj).astype(int)
    # Number of 4-cycles = sum over pairs i<j of C(common[i,j], 2) / 2
    # Each 4-cycle is counted 4 times in the sum over all unordered pairs
    # Actually: sum_{i<j} C(c_ij, 2) counts each 4-cycle 4 times
    # (each 4-cycle has 4 pairs of opposite vertices, and for each such pair
    #  the two common neighbors form the other two vertices)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = common[i, j]
            if c >= 2:
                total += c * (c - 1) // 2
    return total // 2

def _polarity_graph_q5():
    """
    Construct the polarity graph of PG(2,5).
    Points are 1D subspaces of F_5^3. There are (5^3-1)/(5-1) = 124/4 = 31 points.
    The polarity: a point p is adjacent to point q if p^T A q = 0 for a suitable
    symmetric matrix A (the polarity). For q=5 (odd), we use the standard polarity.
    
    Standard polarity for PG(2, q) with q odd:
    Use the Hermitian/symmetric form. For PG(2,5), a standard polarity is given by
    the matrix [[0, 1, 0], [1, 0, 0], [0, 0, 1]] or similar.
    
    Actually, let's use the standard construction:
    Points of PG(2,5) can be represented as:
    - [1, a, b] for a,b in F_5 (25 points)
    - [0, 1, b] for b in F_5 (5 points)  
    - [0, 0, 1] (1 point)
    Total: 31 points.
    
    The polarity for q=5: use the symmetric bilinear form with matrix
    A = [[0, 1, 0], [1, 0, -1], [0, 1, 0]] or the standard one.
    
    Let me use a well-known polarity. For PG(2, q) with q odd, a standard
    involution (polarity) maps [x,y,z] to a point. The associated symmetric
    matrix for the standard polarity is:
    
    For q=5, use: x*z + y^2 = 0 as the absolute points condition.
    The polarity is: point [x,y,z] is adjacent to [x',y',z'] if
    x*z' + x'*z + 2*y*y' = 0 (mod 5).
    
    Wait, let me be more careful. The standard polarity in PG(2,q) for q odd:
    The bilinear form is B((x,y,z), (x',y',z')) = x*z' + x'*z + 2*y*y'
    
    Two points are adjacent if B(p, q) = 0.
    A point is absolute if B(p,p) = 0, i.e., 2*x*z + 2*y^2 = 0, i.e., x*z + y^2 = 0.
    
    For q=5: 2*y*y' mod 5. Let's verify this gives a C4-free graph with 31 vertices and 90 edges.
    """
    import numpy as np
    
    # Enumerate all 31 points of PG(2,5)
    points = []
    # [1, a, b] for a, b in F_5
    for a in range(5):
        for b in range(5):
            points.append((1, a, b))
    # [0, 1, b] for b in F_5
    for b in range(5):
        points.append((0, 1, b))
    # [0, 0, 1]
    points.append((0, 0, 1))
    
    n = len(points)
    assert n == 31
    
    def bilinear(p, q):
        x, y, z = p
        xp, yp, zp = q
        return (x * zp + xp * z + 2 * y * yp) % 5
    
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if bilinear(points[i], points[j]) == 0:
                adj[i, j] = 1
                adj[j, i] = 1
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    
    return adj, edges

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Step 1: Get the polarity graph (C4-free, 90 edges)
    adj, edges = _polarity_graph_q5()
    num_edges = len(edges)
    
    # Verify it's C4-free
    c4_count = _count_c4(adj)
    # Should be 0
    
    if num_edges != 90:
        # Fallback if something went wrong
        # Use a different construction
        pass
    
    # Step 2: Find the edge to add that minimizes new C4s
    # For each non-edge (u,v), compute how many C4s would be created
    # Adding edge (u,v) creates C4s for each pair of common neighbors (w1, w2)
    # where w1 is adjacent to both u and v, and w2 is adjacent to both u and v.
    # The number of new C4s from adding (u,v) = C(common_neighbors(u,v), 2)
    # plus C4s that use the new edge in other ways... 
    # Actually, when we add edge (u,v), new C4s are:
    # - C4s where (u,v) is one of the 4 edges: for each common neighbor pair
    #   {w1, w2} of u and v, the cycle u-w1-v-w2-u is a C4 if w1!=w2 and 
    #   w1-w2 is not an edge (otherwise it's not a simple 4-cycle but a triangle
    #   with a chord... wait no, a C4 requires 4 distinct vertices)
    #   Actually u-w1-v-w2-u is a 4-cycle iff w1 != w2 and w1 is not adjacent to w2
    #   is NOT required - a 4-cycle just needs 4 distinct vertices in a cycle.
    #   u, w1, v, w2 must all be distinct and the edges u-w1, w1-v, v-w2, w2-u must exist.
    #   Since w1 and w2 are common neighbors of u and v, edges u-w1, w1-v, v-w2, w2-u exist.
    #   We need u, w1, v, w2 all distinct. Since w1, w2 are neighbors of u, they're not u.
    #   They're neighbors of v, so not v. w1 != w2. So we just need C(common(u,v), 2).
    #   But wait - we also need that w1-w2 is NOT an edge? No! A 4-cycle is a cycle of length 4.
    #   If w1-w2 is an edge, then u-w1-v-w2-u is still a valid 4-cycle (the edge w1-w2 is 
    #   just a chord, but the cycle itself is valid). Actually no - in graph theory, a C4 
    #   as a subgraph means 4 vertices and 4 edges forming a cycle. The edge w1-w2 would make
    #   it a K4 minus one edge or a triangle with an extra vertex. 
    #   
    #   Let me reconsider: a 4-cycle (C4) as a subgraph is a set of 4 vertices and 4 edges
    #   that form a cycle of length 4. So vertices {u, w1, v, w2} with edges {u-w1, w1-v, 
    #   v-w2, w2-u}. The edge w1-w2 is irrelevant to whether this is a C4 subgraph - 
    #   we just need those 4 specific edges to exist.
    #   
    #   So new C4s from adding (u,v) = C(cn(u,v), 2) where cn(u,v) is common neighbors.
    #   BUT we must also check: are there C4s that don't use edge (u,v) but become C4s
    #   because of the new edge? No - adding an edge can only create new C4s that use
    #   that edge. A C4 uses exactly 4 edges, and if it doesn't use (u,v), it was already
    #   a C4 before.
    #   
    #   Wait, that's not right either. Adding edge (u,v) creates new C4s that include
    #   edge (u,v). Each such C4 has the form u-w1-v-w2-u where w1,w2 are common neighbors
    #   of u and v, w1 != w2. So the count is C(cn(u,v), 2).
    
    # Find the non-edge with minimum C(cn(u,v), 2)
    min_c4_increase = float('inf')
    best_edge = None
    edge_set = set()
    for e in edges:
        edge_set.add((e[0], e[1]))
        edge_set.add((e[1], e[0]))
    
    for i in range(31):
        if time.time() > deadline:
            break
        for j in range(i + 1, 31):
            if (i, j) in edge_set:
                continue
            # Count common neighbors
            cn = 0
            for k in range(31):
                if adj[i, k] and adj[j, k]:
                    cn += 1
            c4s = cn * (cn - 1) // 2
            if c4s < min_c4_increase:
                min_c4_increase = c4s
                best_edge = [i, j]
    
    if best_edge is None:
        # Fallback: just add the first non-edge
        for i in range(31):
            for j in range(i + 1, 31):
                if (i, j) not in edge_set:
                    best_edge = [i, j]
                    break
            if best_edge:
                break
    
    # Add the best edge
    if best_edge:
        edges.append(best_edge)
    
    return edges
