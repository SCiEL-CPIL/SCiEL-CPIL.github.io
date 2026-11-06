"""Rewritten program for the suite task; builds a C4-free graph via polarity graph + local search."""
import random
import sys

def _is_prime_power(q):
    """Check if q is a prime power."""
    if q < 2:
        return False
    # Trial division up to sqrt(q)
    i = 2
    while i * i <= q:
        if q % i == 0:
            # Check if q is a power of i
            temp = q
            while temp % i == 0:
                temp //= i
            if temp == 1:
                return True
            return False
        i += 1
    return True  # q is prime

def _largest_prime_power_less_than(n):
    """Find the largest prime power less than n."""
    if n <= 2:
        return 1
    for q in range(n - 1, 1, -1):
        if _is_prime_power(q):
            return q
    return 1

def _build_polarity_graph(q):
    """
    Build the polarity graph of the projective plane of order q.
    The projective plane of order q has q^2 + q + 1 points.
    The polarity graph has n = q^2 + q + 1 vertices.
    Vertex i is adjacent to vertex j if point i lies on line j (under the polarity).
    
    For the standard polarity (orthogonality), in PG(2,q), we can label points as:
    - Affine points: (x, y) for x, y in GF(q) -> indices 0 to q^2-1
    - Points at infinity: one for each direction -> indices q^2 to q^2+q
    - One extra point at infinity (for the 'vertical' direction or similar) -> index q^2+q+1... 
    
    Actually, let's use a simpler combinatorial construction.
    
    Standard polarity graph of PG(2,q):
    n = q^2 + q + 1 vertices.
    Each vertex has degree q + 1.
    The graph is C4-free.
    
    Construction:
    Label vertices 0..n-1.
    For q prime, we can use finite field arithmetic.
    For general prime powers, we'd need GF(q) arithmetic, which is complex.
    Let's stick to prime q for simplicity, or use a known explicit construction.
    
    Actually, for the purpose of this task, we can use a simpler approach:
    Use the incidence graph of a projective plane, but that's bipartite.
    
    Let's use the standard "orthogonality" polarity graph.
    For q = 2: n = 7, it's the Fano plane polarity graph (complement of C7? No, it's a 3-regular graph on 7 vertices, which is the triangular prism? No, it's actually the complement of the 7-cycle? Let's check: degree 3, 7 vertices. The complement has degree 3 as well. It's the Petersen-like? No, it's the 7-vertex graph where each vertex is connected to 3 others. Actually, the polarity graph of the Fano plane is the complement of the 7-cycle C7? No. Let's just compute it properly.
    
    For a general approach without full GF(q) implementation, we can use the following:
    If q is prime, we can use modular arithmetic.
    
    Let's implement for prime q only, and if q is not prime, fall back to a smaller prime.
    """
    n = q * q + q + 1
    edges = []
    
    # For prime q, we can use the standard construction.
    # Points: (x, y) for x,y in Z_q, plus points at infinity.
    # Let's label:
    # 0..q^2-1: affine points (x, y) where index = x*q + y
    # q^2..q^2+q: points at infinity, one for each slope m in Z_q, index = q^2 + m
    # q^2+q: the point at infinity for vertical lines (or the "last" point)
    
    # Actually, the standard polarity in PG(2,q) with q odd is:
    # (x,y,z) is orthogonal to (x',y',z') if xx' + yy' + zz' = 0.
    # With homogeneous coordinates.
    
    # Let's use a simpler explicit construction for the polarity graph.
    # Reference: The polarity graph of PG(2,q) has vertices corresponding to points of PG(2,q).
    # Two points are adjacent if one lies on the polar line of the other.
    
    # For q prime, we can enumerate points as:
    # Type 1: (1, a, b) for a,b in Z_q -> q^2 points
    # Type 2: (0, 1, c) for c in Z_q -> q points
    # Type 3: (0, 0, 1) -> 1 point
    # Total: q^2 + q + 1
    
    # The polar of point P = (x, y, z) is the line x*X + y*Y + z*Z = 0.
    # Point Q = (x', y', z') lies on polar of P if x*x' + y*y' + z*z' = 0.
    
    # So vertex i is adjacent to vertex j if dot product of their homogeneous coordinates is 0 mod q.
    
    # Let's assign coordinates:
    coords = []
    # Type 1: (1, a, b)
    for a in range(q):
        for b in range(q):
            coords.append((1, a, b))
    # Type 2: (0, 1, c)
    for c in range(q):
        coords.append((0, 1, c))
    # Type 3: (0, 0, 1)
    coords.append((0, 0, 1))
    
    # Build edges
    for i in range(n):
        x1, y1, z1 = coords[i]
        for j in range(i + 1, n):
            x2, y2, z2 = coords[j]
            if (x1 * x2 + y1 * y2 + z1 * z2) % q == 0:
                edges.append([i, j])
    
    return n, edges

def _has_c4(n, adj):
    """Check if graph has a C4. adj is a list of sets."""
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            # Check if i and j have 2 common neighbors
            common = adj[i] & adj[j]
            if len(common) >= 2:
                return True
    return False

def _min_degree(n, adj):
    return min(len(adj[i]) for i in range(n))

def _local_search_add_only(n, adj, min_deg_needed, rng, max_iter=50000):
    """
    Local search that only adds edges. Returns True if min degree >= min_deg_needed.
    """
    for _ in range(max_iter):
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_needed:
            return True
        
        # Find vertices with lowest degree
        low_deg_vertices = [i for i in range(n) if len(adj[i]) < min_deg_needed]
        if not low_deg_vertices:
            return True
        
        rng.shuffle(low_deg_vertices)
        
        added = False
        for u in low_deg_vertices:
            if len(adj[u]) >= min_deg_needed:
                continue
            # Try to find a vertex v not adjacent to u such that adding (u,v) doesn't create C4
            candidates = [v for v in range(n) if v != u and v not in adj[u]]
            rng.shuffle(candidates)
            for v in candidates:
                # Check if adding (u,v) creates C4
                # A C4 is created if u and v have 2 or more common neighbors
                common = adj[u] & adj[v]
                if len(common) < 2:
                    # Safe to add
                    adj[u].add(v)
                    adj[v].add(u)
                    added = True
                    break
            if added:
                break
        
        if not added:
            # No more edges can be added without creating C4
            break
    
    min_deg = min(len(adj[i]) for i in range(n))
    return min_deg >= min_deg_needed

def _local_search_with_removal(n, adj, min_deg_needed, rng, max_iter=100000):
    """
    Local search that can add and remove edges to escape local optima.
    Returns True if min degree >= min_deg_needed.
    """
    for _ in range(max_iter):
        min_deg = min(len(adj[i]) for i in range(n))
        if min_deg >= min_deg_needed:
            return True
        
        # First, try to add edges to low-degree vertices
        low_deg_vertices = [i for i in range(n) if len(adj[i]) < min_deg_needed]
        if low_deg_vertices:
            rng.shuffle(low_deg_vertices)
            added = False
            for u in low_deg_vertices:
                if len(adj[u]) >= min_deg_needed:
                    continue
                candidates = [v for v in range(n) if v != u and v not in adj[u]]
                rng.shuffle(candidates)
                for v in candidates:
                    common = adj[u] & adj[v]
                    if len(common) < 2:
                        adj[u].add(v)
                        adj[v].add(u)
                        added = True
                        break
                if added:
                    break
            
            if added:
                continue
        
        # If no edges could be added, try removing an edge that blocks progress
        # Strategy: find a pair (u,v) where u has low degree, and removing some edge (u,w)
        # would allow adding (u,v)
        if not low_deg_vertices:
            return True
            
        rng.shuffle(low_deg_vertices)
        for u in low_deg_vertices:
            if len(adj[u]) >= min_deg_needed:
                continue
            
            # Find potential new neighbors for u
            potential_new = [v for v in range(n) if v != u and v not in adj[u]]
            rng.shuffle(potential_new)
            
            for v in potential_new:
                common = adj[u] & adj[v]
                if len(common) < 2:
                    # Can add directly
                    adj[u].add(v)
                    adj[v].add(u)
                    break
                elif len(common) == 2:
                    # Need to remove one of the common neighbors' edges
                    # Try removing (u, c) for c in common
                    for c in list(common):
                        # Check if removing (u,c) is safe and allows adding (u,v)
                        # After removing (u,c), u and v share only 1 common neighbor
                        # But we need to check that removing (u,c) doesn't break C4-freeness
                        # (removing an edge never creates a C4)
                        adj[u].discard(c)
                        adj[c].discard(u)
                        
                        # Now try to add (u,v)
                        new_common = adj[u] & adj[v]
                        if len(new_common) < 2:
                            adj[u].add(v)
                            adj[v].add(u)
                            break
                        else:
                            # Restore
                            adj[u].add(c)
                            adj[c].add(u)
            
            # Continue to next low-degree vertex if needed
    
    min_deg = min(len(adj[i]) for i in range(n))
    return min_deg >= min_deg_needed

def construct():
    # Target n = 46 or 47.
    # Try to build a C4-free graph on 46 or 47 vertices with min degree >= n - 39.
    
    # Start with polarity graph of largest prime power q such that q^2 + q + 1 <= 47.
    # q=6: 36+6+1=43. q=7: 49+7+1=57 > 47.
    # So q=6 is not prime. Largest prime power <= 47 with q^2+q+1 <= 47:
    # q=5: 25+5+1=31.
    # q=4: 16+4+1=21 (4 is prime power).
    # q=3: 9+3+1=13.
    # q=2: 4+2+1=7.
    
    # Let's try q=5, n=31. Min degree of polarity graph is q+1=6.
    # We need min degree >= 31-39 = -8, so vacuous. But we want to maximize n.
    
    # Better approach: Start with a known C4-free graph on 46 vertices if possible.
    # The Reiman bound: ex(n, C4) <= n/4 (1 + sqrt(4n-3)).
    # For n=46: 46/4 (1 + sqrt(181)) = 11.5 * (1+13.45) = 11.5*14.45 = 166.2.
    # So max edges ~166. Min degree would be ~2*166/46 = 7.2.
    # We need min degree >= 46-39 = 7. So we need a C4-free graph on 46 vertices with min degree >= 7.
    
    # Let's try to construct one.
    # Start with polarity graph of q=5 (n=31, min degree 6), then add vertices and edges?
    # Or start with a random C4-free graph and add edges.
    
    # Alternative: Use the known construction from the literature for R(C4, K1,39).
    # Since the value is known to be 46 or 47, let's try to find a graph on 46 vertices.
    
    # Let's use a local search approach:
    # 1. Start with a random C4-free graph on 46 vertices.
    # 2. Try to add edges that don't create C4s, prioritizing vertices with low degree.
    # 3. If we get min degree >= 7, we're done.
    
    # Try multiple approaches with different seeds
    best_result = None
    
    for n_target in [46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_needed = n_target - 39
        
        if min_deg_needed <= 0:
            # Trivial: any graph works
            return {'n': n_target, 'edges': []}
        
        # Try multiple seeds
        for seed in range(20):
            rng = random.Random(seed * 1000 + n_target)
            
            # Start with polarity graph core
            q = 5
            n_core, core_edges = _build_polarity_graph(q)
            
            adj = [set() for _ in range(n_target)]
            edges = []
            
            # Add core edges (only if vertices are in range)
            for u, v in core_edges:
                if u < n_target and v < n_target:
                    adj[u].add(v)
                    adj[v].add(u)
                    edges.append([u, v])
            
            # Add some random edges to new vertices to give them some connections
            new_vertices = list(range(n_core, n_target))
            if new_vertices:
                # Connect new vertices to random core vertices
                for v in new_vertices:
                    # Connect to 2-3 random core vertices
                    core_verts = list(range(n_core))
                    rng.shuffle(core_verts)
                    for u in core_verts[:rng.randint(2, 3)]:
                        if u not in adj[v]:
                            # Check if adding (u,v) creates C4
                            common = adj[u] & adj[v]
                            if len(common) < 2:
                                adj[u].add(v)
                                adj[v].add(u)
                                edges.append([u, v])
                
                # Add some edges among new vertices
                for i in range(len(new_vertices)):
                    for j in range(i + 1, len(new_vertices)):
                        u = new_vertices[i]
                        v = new_vertices[j]
                        if u not in adj[v]:
                            common = adj[u] & adj[v]
                            if len(common) < 2 and rng.random() < 0.3:
                                adj[u].add(v)
                                adj[v].add(u)
                                edges.append([u, v])
            
            # Run local search with removal capability
            success = _local_search_with_removal(n_target, adj, min_deg_needed, rng, max_iter=50000)
            
            if success:
                # Verify
                if not _has_c4(n_target, adj):
                    min_deg = min(len(adj[i]) for i in range(n_target))
                    if min_deg >= min_deg_needed:
                        # Rebuild edges list from adj
                        edges = []
                        for i in range(n_target):
                            for j in adj[i]:
                                if i < j:
                                    edges.append([i, j])
                        result = {'n': n_target, 'edges': edges}
                        if best_result is None or n_target > best_result['n']:
                            best_result = result
                        return result
        
        # If n_target failed, continue to next smaller n
        if best_result is not None:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: return the polarity graph on 31 vertices
    n_core, core_edges = _build_polarity_graph(5)
    return {'n': n_core, 'edges': core_edges}
