"""Construct a graph with no regular induced subgraph on 6+ vertices."""
import time
import random
import itertools

def _has_regular_induced_subgraph(adj_bits, n, k=6):
    """Check if there exists a subset of size >= k inducing a regular graph.
    
    adj_bits[i] is a bitset (int) of neighbors of vertex i.
    """
    vertices = list(range(n))
    
    def induced_degree(subset_bits, v):
        # Count neighbors of v within subset
        neighbor_mask = adj_bits[v] & subset_bits
        return bin(neighbor_mask).count('1')
    
    # Check all subsets of size exactly k (if none, then no larger regular induced subgraph
    # can exist? Actually no - a larger regular induced subgraph could exist without
    # containing a smaller regular one. But for k=6, we need to check subsets of size 6, 7, ...
    # However, if a subset S of size m >= 6 induces a regular graph, then every vertex in S
    # has the same degree within S. Let's just check sizes 6 through n.
    
    for size in range(k, n + 1):
        for subset in itertools.combinations(vertices, size):
            subset_bits = 0
            for v in subset:
                subset_bits |= (1 << v)
            
            # Check if all vertices in subset have the same degree within the subset
            degs = set()
            for v in subset:
                d = induced_degree(subset_bits, v)
                degs.add(d)
            if len(degs) == 1:
                return True
    return False

def _is_valid(adj_bits, n, k=6):
    """Check that no induced subgraph on k or more vertices is regular."""
    return not _has_regular_induced_subgraph(adj_bits, n, k)

def _build_adj_bits(edges, n):
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # We'll try to build a graph on up to 22 vertices using a greedy + local search approach.
    # Start with a random graph and try to add edges/vertices while maintaining validity.
    
    # Actually, let's use a different strategy: start with a known structure and grow it.
    # A simple approach: try all edge sets for small n, but that's too slow for n=20+.
    # 
    # Better approach: Use a heuristic. Start with an empty graph on n vertices, then add edges
    # one at a time if they don't create a regular induced subgraph on 6+ vertices.
    # If stuck, try removing some edges and re-adding.
    
    # Let's try a local search: start with a random graph on n vertices, then flip edges
    # to try to maximize the number of vertices while maintaining the constraint.
    
    # Actually, the constraint is that NO subset of 6+ vertices induces a regular graph.
    # This is a hard constraint. Let's think about what regular induced subgraphs look like:
    # - A 6-clique (K6): all degrees 5 within the subset
    # - A 6-cycle (C6): all degrees 2
    # - Two disjoint triangles: each vertex degree 1 within its triangle
    # - K3,3: each vertex degree 3
    # - etc.
    
    # Strategy: Build a graph where every 6-vertex subset has at least two different degrees.
    
    # Let's use a probabilistic/heuristic approach:
    # 1. Start with a random graph on n vertices
    # 2. Check if it's valid
    # 3. If not, identify violating subsets and try to fix by flipping edges
    
    # Given time constraints, let's try a simpler approach:
    # Use a specific construction based on known results.
    
    # For now, let's try a greedy construction:
    # Start with n vertices, no edges.
    # For each potential edge, add it if it doesn't create a regular induced subgraph.
    
    def try_greedy(n):
        """Greedy: add edges in random order if they don't violate the constraint."""
        random.seed(42)
        adj = [0] * n
        edges = []
        
        all_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                all_edges.append((u, v))
        
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            # Tentatively add edge (u, v)
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if _is_valid(adj, n):
                edges.append([u, v])
            else:
                # Remove the edge
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        return adj, edges
    
    # Try greedy for n = 22 down to 20
    for n in range(22, 19, -1):
        if time.time() > deadline:
            break
        adj, edges = try_greedy(n)
        if _is_valid(adj, n):
            return {'n': n, 'edges': edges}
    
    # If greedy failed, try a smaller n
    for n in range(19, 4, -1):
        if time.time() > deadline:
            break
        adj, edges = try_greedy(n)
        if _is_valid(adj, n):
            return {'n': n, 'edges': edges}
    
    # Fallback: empty graph on 5 vertices
    return {'n': 5, 'edges': []}
