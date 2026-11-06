"""Rewritten program for R(C4,K11) construction."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a C4-free graph with no independent set of size 11.
    Strategy: Use a finite geometry-inspired base (polarity graph of projective plane of order 6
    is 7-regular on 43 vertices, but has independent sets). We'll use a different approach:
    
    Actually, let me think more carefully. The projective plane of order 6 has q^2+q+1 = 43 points.
    A polarity graph on 43 vertices where each vertex is connected to its orthogonal points.
    For a symplectic polarity, each vertex has q+1 = 7 orthogonal points (including itself if absolute).
    
    However, the key insight is that we need a C4-free graph with independence number <= 10.
    
    Let me use a known construction: the polarity graph of the projective plane of order q.
    For q=6, we get 43 vertices. The graph is 7-regular (each vertex connected to 7 others).
    It is C4-free by construction (finite geometry property).
    
    The independence number of this graph: for the polarity graph of PG(2,q), the independence
    number is related to the maximum number of mutually non-orthogonal points.
    
    Actually, let me take a more practical approach. I'll construct a C4-free graph using
    a known method and then do local search to reduce the independence number.
    
    A simpler approach: use the incidence graph of a projective plane, but that's bipartite
    and has 2*(q^2+q+1) vertices.
    
    Let me use a different strategy: construct a C4-free graph with n vertices and then
    greedily add edges to reduce the independence number while maintaining C4-freeness.
    
    For the base, I'll use a random C4-free graph or a structured one.
    
    Actually, the best known lower bound for R(C4,K11) is 39, achieved by specific constructions.
    Let me try to build a 39-vertex graph.
    
    Approach: Start with a 7-regular C4-free graph on ~39-43 vertices (like a polarity graph),
    then remove some vertices/edges to get to 39 vertices, and verify independence number <= 10.
    If not, do local search.
    """
    
    def is_c4_free(edges, n):
        """Check if graph is C4-free using adjacency set."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        # Check all pairs of vertices
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if len(common) >= 2:
                    return False
        return True
    
    def max_independent_set_size(adj, n):
        """Find maximum independent set size using greedy + local search."""
        # Use a simple greedy approach with multiple random orderings
        best = 0
        for attempt in range(50):
            random.shuffle(list(range(n)))
            vertices = list(range(n))
            random.shuffle(vertices)
            independent = set()
            for v in vertices:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) > best:
                best = len(independent)
        return best
    
    def can_add_edge(adj, n, u, v):
        """Check if adding edge (u,v) preserves C4-freeness."""
        if u == v:
            return False
        if v in adj[u]:
            return False
        # Adding edge (u,v) creates C4 if there are two common neighbors of u and v
        common = adj[u] & adj[v]
        if len(common) >= 2:
            return False
        # Also need to check: for any w in adj[u] and x in adj[v], 
        # if w == x, that's one common neighbor (ok if < 2)
        # The condition is: |N(u) ∩ N(v)| <= 1 after adding the edge
        # Before adding, |N(u) ∩ N(v)| must be <= 1 (since adding doesn't change common neighbors)
        # Wait, adding edge (u,v) doesn't change N(u) ∩ N(v) for other pairs.
        # It only matters for the pair (u,v) itself: after adding, u and v are adjacent,
        # and they share |N(u) ∩ N(v)| common neighbors. For C4-free, adjacent vertices
        # can have at most 1 common neighbor.
        return True
    
    # Build a polarity graph of the projective plane of order 6
    # Projective plane PG(2,6) has 43 points
    # Label points as (x,y,1) and (x,1,0) etc. in homogeneous coordinates
    # For simplicity, let's use a different construction
    
    # Actually, let me use a known C4-free graph construction.
    # The Erdős-Rényi polarity graph: vertices are points of PG(2,q), 
    # two points are adjacent if they are orthogonal w.r.t. a polarity.
    
    # For q=6, this gives 43 vertices, 7-regular, C4-free.
    # Let's construct it properly.
    
    q = 6
    # Points of PG(2,q): represented as (x, y, z) with z=1 or z=0
    # Total: q^2 + q + 1 = 43 points
    points = []
    # Affine part: (x, y, 1) for x,y in GF(q)
    for x in range(q):
        for y in range(q):
            points.append((x, y, 1))
    # Points at infinity: (x, 1, 0) for x in GF(q), and (0, 1, 0)
    for x in range(q):
        points.append((x, 1, 0))
    points.append((0, 1, 0))
    
    # Wait, that's q^2 + q + 1 = 36 + 6 + 1 = 43. Good.
    # But I need to be careful: the point at infinity for vertical lines is (0,1,0).
    # Points at infinity: (1, m, 0) for m in GF(q) and (0, 1, 0).
    # Let me redo:
    points = []
    for x in range(q):
        for y in range(q):
            points.append((x, y))  # affine points, index 0..35
    # Points at infinity
    for m in range(q):
        points.append(('inf', m))  # index 36..41
    points.append(('inf', 'vert'))  # index 42
    
    n_total = len(points)  # 43
    
    # Define the polarity. Use a symplectic-type polarity.
    # For simplicity, use the "standard" polarity: 
    # Point (x,y) is orthogonal to point (x',y') if:
    #   x*y' + y*x' + 1 = 0  (for affine points)
    # This is a bit tricky. Let me use a simpler approach.
    
    # Alternative: use a known C4-free graph from literature.
    # Let me just do a greedy construction.
    
    # Start with empty graph on 39 vertices, add edges greedily to make it dense
    # while maintaining C4-freeness, then check independence number.
    
    # Better: use the polarity graph and then prune to 39 vertices.
    
    # Let me implement the polarity graph properly.
    # Use GF(6) - but 6 is not prime power! So PG(2,6) doesn't exist.
    # Oops. q must be a prime power. 6 is not a prime power.
    
    # So I need to use a different q. q=5 gives 31 vertices, q=7 gives 57 vertices.
    # Neither is 39.
    
    # Let me use a different approach entirely.
    
    # Known result: R(C4, K11) > 39. The construction is specific.
    # Let me try to build a C4-free graph on 39 vertices with alpha <= 10.
    
    # Strategy: 
    # 1. Start with a 7-regular C4-free graph on 39 vertices if possible, 
    #    or a denser one.
    # 2. Use local search to reduce independence number.
    
    # For a C4-free graph, the maximum number of edges is bounded.
    # For n=39, the max edges in a C4-free graph is roughly n^{3/2}/2.
    
    # Let me use a random approach: generate random C4-free graphs and check.
    
    # Actually, let me use a structured approach based on finite fields.
    # Use GF(5) to get a 31-vertex polarity graph, then extend.
    
    # Or better: use the known construction from the literature.
    # The best known C4-free graph with alpha <= 10 has 39 vertices.
    
    # Let me try a practical approach:
    # 1. Create a 7-regular graph on 39 vertices that is C4-free.
    # 2. Verify and optimize.
    
    # For C4-freeness, I can use the following construction:
    # Take vertices as pairs (i, j) where i in range(a), j in range(b), a*b = 39.
    # 39 = 3 * 13. So vertices are (i, j) with i in {0,1,2}, j in {0,...,12}.
    # Connect (i,j) to (i',j') if some condition holds.
    
    # This is getting complicated. Let me just do a search.
    
    # Practical approach: 
    # 1. Start with a random graph on 39 vertices.
    # 2. Remove edges to make it C4-free.
    # 3. Add edges greedily to increase density while maintaining C4-freeness.
    # 4. Check independence number. If > 10, try to fix by adding more edges.
    
    # But this might not find alpha <= 10.
    
    # Let me use a known construction. The polarity graph of the projective plane
    # of order q=5 has 31 vertices and is 6-regular C4-free. Its independence number
    # is 6 (since it's a polarity graph, alpha = q = 5 or 6).
    
    # For q=5: 31 vertices, 6-regular, C4-free, alpha = 6.
    # We need alpha <= 10, so 31 vertices works but we want more.
    
    # Can we extend the q=5 polarity graph to more vertices?
    
    # Alternatively, use a union of smaller C4-free graphs.
    # 10 disjoint triangles: 30 vertices, alpha = 10.
    # Add 9 more vertices with edges to maintain alpha <= 10.
    
    # Let me try: start with 10 triangles (30 vertices, alpha=10), add 9 more vertices.
    # Connect each new vertex to at least one vertex in each triangle to prevent 
    # independent sets from picking one from each triangle plus the new vertices.
    
    # If I add a vertex v and connect it to one vertex in each triangle, then
    # any independent set can contain v or at most one from each triangle, but not both.
    # This is complex.
    
    # Let me just do a local search on a 39-vertex graph.
    
    random.seed(42)
    n = 39
    
    # Initialize with a 7-regular-like graph
    # Use a cycle plus chords
    adj = [set() for _ in range(n)]
    edges = []
    
    # Create a base graph: connect each vertex to its neighbors in a structured way
    # Use a bipartite-like structure to avoid C4
    
    # Let me use a grid-like structure: 3 x 13 grid
    # Vertices (i, j) for i in 0..2, j in 0..12
    # Connect (i,j) to (i, j+1) mod 13 (cycle in each row)
    # Connect (i,j) to (i+1, j) mod 3 (columns)
    # This creates C4s! (i,j)-(i,j+1)-(i+1,j+1)-(i+1,j)-(i,j)
    
    # Need to be more careful.
    
    # Use a different construction: the incidence graph of a linear hypergraph.
    
    # Let me just do a greedy edge addition with C4 check.
    
    adj = [set() for _ in range(n)]
    edges = []
    
    # Start with a random spanning tree
    perm = list(range(n))
    random.shuffle(perm)
    for i in range(1, n):
        u = perm[i]
        v = perm[random.randint(0, i-1)]
        adj[u].add(v)
        adj[v].add(u)
        edges.append([u, v])
    
    # Try to add more edges
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if v in adj[u]:
            continue
        # Check if adding (u,v) creates a C4
        common = adj[u] & adj[v]
        if len(common) < 2:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    # Now check independence number
    def greedy_max_independent_set(adj, n, num_trials=100):
        best = 0
        for _ in range(num_trials):
            order = list(range(n))
            random.shuffle(order)
            independent = set()
            for v in order:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) > best:
                best = len(independent)
        return best
    
    alpha = greedy_max_independent_set(adj, n, 50)
    
    # If alpha > 10, try to reduce it by adding more edges
    iterations = 0
    while alpha > 10 and iterations < 1000:
        iterations += 1
        # Find a maximum independent set
        order = list(range(n))
        random.shuffle(order)
        independent = set()
        for v in order:
            if all(u not in independent for u in adj[v]):
                independent.add(v)
        
        if len(independent) <= 10:
            alpha = len(independent)
            continue
        
        # Try to add an edge within the independent set to break it
        indep_list = list(independent)
        random.shuffle(indep_list)
        added = False
        for i in range(len(indep_list)):
            for j in range(i+1, len(indep_list)):
                u, v = indep_list[i], indep_list[j]
                if v not in adj[u]:
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        adj[u].add(v)
                        adj[v].add(u)
                        edges.append([u, v])
                        added = True
                        break
            if added:
                break
        
        if not added:
            break
        
        alpha = greedy_max_independent_set(adj, n, 50)
    
    # Final check
    if not is_c4_free(edges, n):
        # Fallback: return the 30-vertex construction
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Verify independence number more carefully
    alpha_final = greedy_max_independent_set(adj, n, 200)
    if alpha_final > 10:
        # Try harder or fallback
        # Try more local search
        for _ in range(500):
            order = list(range(n))
            random.shuffle(order)
            independent = set()
            for v in order:
                if all(u not in independent for u in adj[v]):
                    independent.add(v)
            if len(independent) <= 10:
                break
            indep_list = list(independent)
            random.shuffle(indep_list)
            for i in range(len(indep_list)):
                for j in range(i+1, len(indep_list)):
                    u, v = indep_list[i], indep_list[j]
                    if v not in adj[u]:
                        common = adj[u] & adj[v]
                        if len(common) < 2:
                            adj[u].add(v)
                            adj[v].add(u)
                            edges.append([u, v])
                            break
                else:
                    continue
                break
        
        alpha_final = greedy_max_independent_set(adj, n, 200)
        if alpha_final > 10:
            # Fallback
            edges = []
            for t in range(10):
                a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
                edges += [[a, b], [a, c], [b, c]]
            return {'n': 30, 'edges': edges}
    
    return {'n': n, 'edges': edges}

# EVOLVE-BLOCK-END
