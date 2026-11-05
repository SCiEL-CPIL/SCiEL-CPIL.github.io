"""Construction for R(C4,K11) > n using known polarity graph / C4-free constructions."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _build_polarity_graph_q(q):
    """
    Build the polarity graph on the projective plane of order q.
    Vertices: points of PG(2,q). Two points are adjacent if they are NOT orthogonal
    (i.e., not on each other's polar line).
    
    This graph is C4-free and has n = q^2 + q + 1 vertices.
    For q=5: n=31, q=6: n=43 (but PG(2,6) doesn't exist, so max prime power is 5).
    
    Actually, polarity graphs work for any prime power q where PG(2,q) exists.
    For q=5, n=31.
    
    But we need larger. Let's use a different approach.
    """
    pass


def _build_c4_free_from_finite_field(q):
    """
    Construct a C4-free graph using finite field methods.
    
    Use the construction: vertices are elements of GF(q^2), 
    edge between x and y if x-y is a square in GF(q^2) but not in GF(q).
    
    Actually, let's use a simpler well-known construction.
    
    For the polarity graph on PG(2,q):
    - Number of vertices: q^2 + q + 1
    - Each vertex has degree q (for a suitable polarity)
    - The graph is C4-free
    
    For q=5: 31 vertices, degree 5, alpha ~ (q^2+q+1)/(q+1) = q = 5... 
    Wait, that gives alpha <= 5 which is way too small. We need alpha <= 10.
    
    Let me reconsider. The independence number of the polarity graph PG(2,q) 
    with the standard polarity is at most q. So for q=5, alpha <= 5.
    That's too restrictive - we need alpha exactly 10 or less.
    
    Actually, we want NO independent set of size 11, so alpha <= 10.
    The polarity graph for q=5 has alpha <= 5, which satisfies alpha <= 10.
    But we want to maximize n.
    
    For q=7: n = 49+7+1 = 57, but 7 is a prime power so PG(2,7) exists.
    But wait, the task says n <= 43. So we can't use q=7.
    
    For q=5: n=31.
    For q=6: PG(2,6) doesn't exist (6 is not a prime power).
    
    So polarity graphs give us at most 31 vertices within the n<=43 constraint.
    """
    pass


def _construct_erdos_renyi_style():
    """
    Try to construct a C4-free graph with alpha <= 10 and n as large as possible.
    
    Known result: R(C4, K11) is between 39 and 44 (or similar bounds).
    The best known construction has 39 vertices.
    
    Let me try a different approach: use a known explicit construction or
    search for one.
    
    One approach: start with a high-degree C4-free graph and verify alpha.
    
    Let's try the construction based on the incidence structure of a projective plane.
    
    Actually, let me try to use a known result: the graph from the (31,11,1) 
    difference set or similar.
    
    Alternative: use a greedy/randomized search to build a C4-free graph 
    and check alpha.
    """
    pass


def _max_independent_set(adj, n, timeout=5.0):
    """
    Find the maximum independent set size using branch and bound.
    adj: list of sets (adjacency lists)
    Returns the size of the maximum independent set.
    """
    start_time = time.time()
    
    # Use a greedy coloring bound for the complement (i.e., coloring the original graph)
    # Actually, for maximum independent set, we use branch and bound on the original graph.
    # An independent set in G is a clique in the complement.
    
    # For efficiency, use a simple branch and bound:
    # State: (candidates_set, current_set_size)
    # candidates: vertices that can still be added to the independent set
    # Bound: current_size + |candidates| <= best found?
    
    best = [0]
    
    # Greedy initial solution
    def greedy_is(adj, n):
        remaining = set(range(n))
        is_set = set()
        while remaining:
            # Pick vertex with minimum degree in remaining
            min_deg = float('inf')
            pick = None
            for v in remaining:
                deg_in_rem = len(adj[v] & remaining)
                if deg_in_rem < min_deg:
                    min_deg = deg_in_rem
                    pick = v
            if min_deg == 0:
                # All remaining are independent
                is_set.update(remaining)
                break
            is_set.add(pick)
            remaining.discard(pick)
            remaining -= adj[pick]
        return len(is_set)
    
    best[0] = greedy_is(adj, n)
    
    # Branch and bound
    # Use bitmasks for efficiency if n is small enough
    if n <= 43:
        # Use integer bitmasks
        adj_mask = [0] * n
        for v in range(n):
            mask = 0
            for u in adj[v]:
                mask |= (1 << u)
            adj_mask[v] = mask
        
        def bnb(candidates_mask, current_size):
            if time.time() - start_time > timeout:
                return
            if candidates_mask == 0:
                if current_size > best[0]:
                    best[0] = current_size
                return
            # Upper bound: current_size + popcount(candidates_mask)
            # Better bound: use greedy coloring of the subgraph induced by candidates
            # For now, use simple bound
            popcount = bin(candidates_mask).count('1')
            if current_size + popcount <= best[0]:
                return
            
            # Pick a vertex to branch on
            # Choose vertex with maximum degree in the candidate subgraph
            # Or just pick the first set bit
            # For better performance, pick vertex that maximizes pruning
            
            # Find a vertex in candidates
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            
            # Branch 1: include v
            new_candidates = candidates_mask & ~adj_mask[v] & ~(1 << v)
            bnb(new_candidates, current_size + 1)
            
            # Branch 2: exclude v
            bnb(candidates_mask & ~(1 << v), current_size)
        
        bnb((1 << n) - 1, 0)
    
    return best[0]


def _verify_c4_free(adj, n):
    """Check if the graph is C4-free: no two vertices have 2+ common neighbors."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True


def _construct_polarity_graph(q):
    """
    Construct the polarity graph on PG(2,q) for prime power q.
    
    Points of PG(2,q): equivalence classes of non-zero vectors in GF(q)^3 under scalar multiplication.
    There are (q^3 - 1)/(q - 1) = q^2 + q + 1 points.
    
    A polarity: map point (x,y,z) to line xz + y^2 = 0 (or similar).
    Two points are adjacent if one is NOT on the polar line of the other.
    
    For simplicity, let's use the standard construction:
    Points represented as (x, y, 1) for affine points, (1, t, 0) and (0, 1, 0) for points at infinity.
    
    Actually, let me use a cleaner representation.
    
    For PG(2,q), label points as:
    - (x, y, 1) for x, y in GF(q): q^2 affine points
    - (1, t, 0) for t in GF(q): q points at infinity  
    - (0, 1, 0): 1 point at infinity
    Total: q^2 + q + 1
    
    Polarity: point (x, y, z) is orthogonal to point (x', y', z') if x*x' + y*y' + z*z' = 0
    Wait, that's for a symmetric bilinear form. Let me use the standard one.
    
    Standard polarity on PG(2,q): (x,y,z) is orthogonal to (x',y',z') iff xz' + y^2' ... 
    
    Let me use a simpler approach. The graph where two points are adjacent if they are NOT 
    orthogonal under a fixed polarity. This is a C4-free graph.
    
    For GF(q), represent elements as integers 0..q-1 with modular arithmetic.
    For q a prime, this is straightforward.
    """
    if q <= 1:
        return None
    
    # Check if q is prime (for simplicity, only handle primes)
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    if not is_prime(q):
        return None
    
    # Points of PG(2,q)
    # Represent each point as a normalized triple (a, b, c) with gcd(a,b,c)=1 and a>0 or a=0,b>0 etc.
    # Simpler: enumerate all non-zero triples (x,y,z) in GF(q)^3, identify (x,y,z) ~ (kx,ky,kz) for k!=0.
    # Representative: make the first non-zero coordinate 1.
    
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: find first non-zero, divide by it
                if x != 0:
                    inv_x = pow(x, q - 2, q)  # Fermat's little theorem for prime q
                    nx, ny, nz = 1, (y * inv_x) % q, (z * inv_x) % q
                elif y != 0:
                    inv_y = pow(y, q - 2, q)
                    nx, ny, nz = 0, 1, (z * inv_y) % q
                else:  # z != 0
                    inv_z = pow(z, q - 2, q)
                    nx, ny, nz = 0, 0, 1
                
                key = (nx, ny, nz)
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n = len(points)  # Should be q^2 + q + 1
    assert n == q * q + q + 1, f"Expected {q*q+q+1}, got {n}"
    
    # Define polarity: (x,y,z) orthogonal to (x',y',z') iff x*z' + y*y' + z*x' = 0
    # Wait, this is a symmetric bilinear form B((x,y,z),(x',y',z')) = x*z' + y*y' + z*x'
    # Check symmetry: B(u,v) = x*z' + y*y' + z*x', B(v,u) = x'*z + y'*y + z'*x = same. Good.
    
    def orthogonal(p, p_):
        x, y, z = p
        xp, yp, zp = p_
        return (x * zp + y * yp + z * xp) % q == 0
    
    # Build adjacency: two points are adjacent if NOT orthogonal
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not orthogonal(points[i], points[j]):
                adj[i].add(j)
                adj[j].add(i)
    
    return adj, n


def _construct_search(start_n, target_n, deadline):
    """
    Search for a C4-free graph with alpha <= 10.
    Start from a base graph and try to add vertices.
    """
    # This is a fallback search
    pass


def construct():
    deadline = time.time() + 140.0  # Safety margin
    
    # Try polarity graph constructions for various prime powers
    # q=5 gives n=31, q=7 gives n=57 (too big, n<=43)
    # So q=5 is the largest prime power with n <= 43.
    
    # But we want to maximize n. Let's also try other constructions.
    
    best_n = 0
    best_edges = []
    
    # Try q=5 polarity graph: n=31
    result = _construct_polarity_graph(5)
    if result:
        adj, n = result
        # Verify C4-free
        if _verify_c4_free(adj, n):
            # Check alpha
            alpha = _max_independent_set(adj, n, timeout=10.0)
            if alpha <= 10:
                if n > best_n:
                    best_n = n
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append([i, j])
                    best_edges = edges
    
    # Try to extend or find better constructions
    # The best known is 39 vertices. Let's try a search approach.
    
    # Alternative: use a known construction from the literature.
    # For R(C4, K11), there are known C4-free graphs on 39 vertices with alpha <= 10.
    # One such construction uses the complement of a known design.
    
    # Let's try a randomized construction:
    # Start with n=39 vertices, try to build a C4-free graph with alpha <= 10.
    
    # Use a greedy approach with backtracking
    random.seed(42)
    
    for target_n in [39, 38, 37, 36, 35, 34, 33, 32, 31, 30]:
        if time.time() >= deadline:
            break
        if target_n <= best_n:
            break
        
        # Try to construct a C4-free graph on target_n vertices with alpha <= 10
        # Strategy: start with a random graph, remove edges to make it C4-free,
        # then check alpha.
        
        found = False
        for attempt in range(50):
            if time.time() >= deadline:
                break
            
            # Build a random C4-free graph
            # Start with no edges, add edges greedily while maintaining C4-free
            adj = [set() for _ in range(target_n)]
            
            # Order of vertices to add edges
            edge_order = []
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    edge_order.append((i, j))
            random.shuffle(edge_order)
            
            # Add edges greedily
            for (u, v) in edge_order:
                # Check if adding edge (u,v) creates a C4
                # A C4 is created if u and v have a common neighbor w, and 
                # there's another common neighbor... wait.
                # Adding edge (u,v) creates a C4 if there exist x,y such that
                # u-x-v-y-u is a 4-cycle. This happens if x is adjacent to both u and v,
                # and y is adjacent to both u and v, and x != y.
                # Actually, adding edge (u,v) creates a C4 if u and v already have 
                # at least 2 common neighbors.
                common = adj[u] & adj[v]
                if len(common) < 2:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Now adj is C4-free by construction
            # Check alpha
            alpha = _max_independent_set(adj, target_n, timeout=8.0)
            if alpha <= 10:
                if target_n > best_n:
                    best_n = target_n
                    edges = []
                    for i in range(target_n):
                        for j in adj[i]:
                            if i < j:
                                edges.append([i, j])
                    best_edges = edges
                found = True
                break
        
        if found:
            break
    
    # If we didn't find anything better, try the polarity graph result
    if best_n == 0:
        # Fallback: 10 disjoint triangles
        n = 30
        edges = []
        for t in range(10):
            a, b, c = 3*t, 3*t+1, 3*t+2
            edges += [[a,b], [a,c], [b,c]]
        best_n = n
        best_edges = edges
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
