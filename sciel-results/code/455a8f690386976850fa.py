"""Construct a 31-vertex, 91-edge graph with few C4s using a polarity-graph structure."""

def construct():
    # We'll build a graph on 31 vertices labeled 0..30.
    # Use a structure inspired by projective plane PG(2,5) with 31 points.
    # First, label vertices 0..24 as points, 25..30 as extra vertices.
    # We'll use a cyclic construction to build 90 C4-free edges, then add 1 more.
    
    n = 31
    q = 5
    
    # Label vertices: 0..24 for "point" vertices, 25..30 for "special" vertices
    # Use a field-like structure mod 25 (not a field, but cyclic group Z/25Z)
    # We'll use a Singer cycle approach adapted to composite modulus.
    
    # Create adjacency set
    edges = set()
    
    def add_edge(u, v):
        if u == v:
            return
        u, v = min(u, v), max(u, v)
        edges.add((u, v))
    
    # Build base graph using difference method
    # For vertices 0..24 (25 vertices), use differences in Z/25Z
    # Connect i to i+d mod 25 for certain differences d
    # We need to avoid C4s, so we need to be careful about which differences to use.
    
    # In Z/25Z, the multiplicative group has structure that doesn't directly give us
    # a polarity graph. Let's use a different approach.
    
    # Alternative: Use a known C4-free construction and extend it.
    # For 31 vertices, we can use the following approach:
    # Split into two sets and use a bipartite-like structure plus some internal edges.
    
    # Let's use a simpler approach: build the graph via adjacency list
    # using a deterministic construction that minimizes C4s.
    
    # Initialize with no edges
    adj = [set() for _ in range(n)]
    
    def add_edge(u, v):
        if u == v:
            return False
        u, v = min(u, v), max(u, v)
        if (u, v) in edges:
            return False
        edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        return True
    
    # Build a C4-free graph on 31 vertices with 90 edges
    # Use the construction from the fact that ex(31, C4) >= 90
    
    # Strategy: use a random-looking but deterministic construction
    # We'll use a greedy approach with a fixed seed-like ordering
    
    # Let's use the following construction:
    # For vertices 0..29, use a 5x6 grid-like structure
    # Vertex i is at position (row, col) where row = i % 5, col = i // 5
    # Connect vertices in same row with some pattern, and across rows with another pattern
    
    # Actually, let's use a well-known approach: 
    # Take the line graph of a bipartite graph, or use a polarity of a partial geometry.
    
    # For practicality, let's construct the graph using the following method:
    # 1. Create 31 vertices
    # 2. Add edges using a structured pattern that avoids C4s as much as possible
    
    # Use a deterministic pseudo-random edge selection with a fixed seed
    # This ensures reproducibility while allowing good C4 avoidance
    
    # We'll use a construction based on the incidence structure of PG(2,5)
    # The projective plane PG(2,5) has 31 points and 31 lines, each line has 6 points.
    # The polarity graph connects each point to all points on its polar line.
    
    # Construct PG(2,5) points as pairs (x, y) where x, y in {0,1,2,3,4} plus (0,0) 
    # No, that's 26. PG(2,5) has 5^2 + 5 + 1 = 31 points.
    # Points are equivalence classes of [x:y:z] in GF(5)^3 \ {0}
    
    # Enumerate all points of PG(2,5)
    points = []
    seen = set()
    for x in range(5):
        for y in range(5):
            for z in range(1, 5):  # z >= 1 to normalize
                pt = (x, y, z)
                key = (x, y, z)
                if key not in seen:
                    seen.add(key)
                    points.append(pt)
    
    # Add points with z=0: [x:y:0] with x,y not both 0
    for x in range(1, 5):
        pt = (x, 0, 0)
        if pt not in seen:
            seen.add(pt)
            points.append(pt)
    for y in range(1, 5):
        pt = (0, y, 0)
        if pt not in seen:
            seen.add(pt)
            points.append(pt)
    
    # Add [1:0:0] if not already there
    if (1, 0, 0) not in seen:
        seen.add((1, 0, 0))
        points.append((1, 0, 0))
    
    # We should have 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define the polarity: a(x, y, z) = (z, y, x) or some symmetric bilinear form
    # For PG(2, q) with q odd, a standard polarity is given by:
    # F(x, y, z) = x*z + y^2 = 0 (symmetric hyperbolic) or x*z = y^2 (conic)
    # Let's use: x*z - y^2 = 0 mod 5, i.e., x*z = y^2 in GF(5)
    
    # The polar line of point P = (px, py, pz) consists of all points Q = (x, y, z)
    # such that the bilinear form B(P, Q) = 0
    # For the form x*z - y^2, the associated bilinear form is:
    # B(P, Q) = px*qz + pz*qx - 2*py*qy (mod 5)
    # Wait, x*z - y^2 is not bilinear. Let's use a proper symmetric bilinear form.
    
    # Use the form: B((x1,y1,z1), (x2,y2,z2)) = x1*z2 + z1*x2 + 2*y1*y2 (mod 5)
    # The corresponding quadratic form is Q(x,y,z) = x*z + y^2 (mod 5)
    # A point P is on its polar line if Q(P) = 0
    # The polarity graph connects P to Q if B(P, Q) = 0 and P != Q (or include self if Q(P)=0)
    
    # For the polarity graph, we connect P to all Q such that B(P, Q) = 0
    # If Q(P) = 0, then P is on its own polar line, so we add a loop (but we want simple graph,
    # so typically we exclude loops or include them differently).
    
    # Let's count edges: for each P, the number of Q with B(P,Q) = 0 is q+1 = 6
    # Since B(P, P) may be 0 or not, the degree is either 5 or 6.
    # Total edges = (sum of degrees) / 2
    
    # Let's compute
    def B(p1, p2):
        x1, y1, z1 = p1
        x2, y2, z2 = p2
        return (x1 * z2 + z1 * x2 + 2 * y1 * y2) % 5
    
    polarity_edges = set()
    for i, p in enumerate(points):
        for j, q_pt in enumerate(points):
            if i < j:
                if B(p, q_pt) == 0:
                    polarity_edges.add((i, j))
    
    polarity_edge_count = len(polarity_edges)
    # Should be around 90 (since 31 vertices * ~6 degree / 2 = ~93, minus self-loops)
    
    if polarity_edge_count > 91:
        # We have too many edges, need to remove some
        # This shouldn't happen for a proper polarity, but let's handle it
        edge_list = sorted(polarity_edges)
        edges_to_remove = polarity_edge_count - 90
        polarity_edges = set(edge_list[edges_to_remove:])
    elif polarity_edge_count < 90:
        # Need to add edges
        all_possible = set()
        for i in range(31):
            for j in range(i+1, 31):
                all_possible.add((i, j))
        available = all_possible - polarity_edges
        need = 90 - polarity_edge_count
        # Add edges greedily to minimize C4 increase
        edge_list = sorted(available)
        for e in edge_list:
            if need <= 0:
                break
            polarity_edges.add(e)
            need -= 1
    
    # Now add edges to get to 90
    while len(polarity_edges) < 90:
        all_possible = set()
        for i in range(31):
            for j in range(i+1, 31):
                all_possible.add((i, j))
        available = all_possible - polarity_edges
        if not available:
            break
        # Pick the edge that minimizes common neighbors
        best_e = None
        best_cn = float('inf')
        for e in available:
            u, v = e
            common = len(adj[u] & adj[v]) if adj else 0
            # Use simple heuristic
            if best_e is None or common < best_cn:
                best_cn = common
                best_e = e
        if best_e:
            polarity_edges.add(best_e)
        else:
            break
    
    polarity_edges = list(polarity_edges)
    assert len(polarity_edges) == 90, f"Expected 90 edges, got {len(polarity_edges)}"
    
    # Now we need to add 1 more edge (to get 91 total)
    # Choose the edge that adds the minimum number of C4s
    # A C4 is formed when we add edge (u,v) if u and v have a common neighbor w,
    # and there's another common neighbor w' such that the path u-w-w'-v forms a C4
    # Actually, adding edge (u,v) creates a C4 for each pair of common neighbors (w, w')
    # where the edges (u,w), (w,v), (w',v), (v,u) all exist. Wait, that's not right.
    # 
    # When we add edge (u,v), a C4 is formed if there exist vertices w1, w2 such that:
    # u-w1-v-w2-u is a 4-cycle, i.e., edges (u,w1), (w1,v), (v,w2), (w2,u) all exist.
    # But (v,w2) and (w2,u) are the same as (w2,v) and (u,w2).
    # So we need u and v to have common neighbors w1 and w2, and the edges (u,w1), (w1,v), (v,w2), (w2,u) must form a cycle.
    # Actually, any two common neighbors w1, w2 of u and v in the current graph, together with the new edge (u,v),
    # form a 4-cycle u-w1-v-w2-u IF the edges (w1,v) and (w2,u) exist. But w1 is a common neighbor of u and v,
    # so (w1,v) exists. Similarly (w2,u) exists. So YES, every pair of common neighbors forms a C4 with the new edge.
    # 
    # Wait, that's only if u-w1-v-w2 is a simple 4-cycle, which it is as long as w1 != w2 and all four edges exist.
    # Since w1, w2 are common neighbors of u and v, edges (u,w1), (v,w1), (u,w2), (v,w2) all exist.
    # The cycle u-w1-v-w2-u uses edges (u,w1), (w1,v), (v,w2), (w2,u). All exist. And it's a 4-cycle as long as
    # u != w1 != v != w2 != u, which is true.
    # So the number of new C4s added by edge (u,v) is C(cn(u,v), 2) where cn(u,v) is the number of common neighbors.
    
    # Let's build the adjacency structure
    for u, v in polarity_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    all_possible = set()
    for i in range(31):
        for j in range(i+1, 31):
            all_possible.add((i, j))
    available = all_possible - polarity_edges
    
    best_edge = None
    best_new_c4s = float('inf')
    for e in available:
        u, v = e
        cn = len(adj[u] & adj[v])
        new_c4s = cn * (cn - 1) // 2
        if new_c4s < best_new_c4s:
            best_new_c4s = new_c4s
            best_edge = e
        if new_c4s == 0:
            break
    
    if best_edge:
        u, v = best_edge
        polarity_edges.append((u, v))
    
    # Sort and return
    result = [list(e) for e in polarity_edges]
    result.sort()
    
    # Verify: exactly 91 distinct edges, vertices 0..30
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    for e in result:
        assert len(e) == 2
        assert 0 <= e[0] < 31
        assert 0 <= e[1] < 31
        assert e[0] < e[1]
    
    # Check distinctness
    seen_edges = set()
    for e in result:
        assert e not in seen_edges, f"Duplicate edge: {e}"
        seen_edges.add(e)
    
    return result

if __name__ == "__main__":
    edges = construct()
    # Count C4s for verification
    n = 31
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    c4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            common = adj[i] & adj[j]
            c4_count += len(common) * (len(common) - 1) // 2
    
    print(f"Number of edges: {len(edges)}")
    print(f"Number of C4s: {c4_count}")
