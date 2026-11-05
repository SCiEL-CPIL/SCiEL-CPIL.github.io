import time
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 128
    numbers = list(range(1, n + 1))

    # Helper: LCM calculation
    def lcm(a, b):
        return a * b // gcd(a, b)

    # Helper: count distinct prime factors
    def num_distinct_primes(x):
        count = 0
        d = 2
        while d * d <= x:
            if x % d == 0:
                count += 1
                while x % d == 0:
                    x //= d
            d += 1
        if x > 1:
            count += 1
        return count

    # Generate all bad triples
    # A bad triple is {a, b, c} with a < b < c such that
    # lcm(a, b) == lcm(a, c) == lcm(b, c)
    
    bad_triples = []
    # We can optimize this by precomputing lcms
    # For n=128, C(128,3) is about 340k. This is fast.
    
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            lab = lcm(a, b)
            for c in range(b + 1, n + 1):
                if lcm(a, c) == lab and lcm(b, c) == lab:
                    bad_triples.append((a, b, c))
    
    # Build hypergraph
    # Vertices: 1..128
    # Edges: bad_triples (size 3)
    
    # We want to find the maximum independent set in this hypergraph.
    # Equivalently, find the minimum vertex cover (remove vertices to hit all edges).
    # Max independent set size = n - Min Vertex Cover size.
    
    # Greedy Vertex Cover Heuristic:
    # 1. Start with all vertices.
    # 2. While there are edges:
    #    - Find the edge with the highest "weight" (e.g., sum of degrees of its vertices, or just pick one).
    #    - Remove one of the vertices from that edge (the one with highest degree in the remaining hypergraph).
    #    - Update degrees and remove all edges incident to the removed vertex.
    
    # Let's represent the hypergraph as a list of sets (edges).
    # And a set of active vertices.
    
    active_vertices = set(range(1, n + 1))
    active_edges = [set(t) for t in bad_triples]
    
    # Degree of a vertex = number of active edges it belongs to.
    # We'll maintain a list of edges and filter out dead ones.
    
    removed = set()
    
    # To make this efficient, we can just iterate and remove.
    # Since n is small and edges are ~few thousands, O(E * V) is fine.
    
    # Let's precompute vertex-edge incidence for speed?
    # vertex_edges[v] = list of indices of edges containing v
    
    vertex_edges = {v: [] for v in range(1, n + 1)}
    for i, edge in enumerate(active_edges):
        for v in edge:
            vertex_edges[v].append(i)
    
    # active_edge_flags to quickly check if edge is still alive
    edge_alive = [True] * len(active_edges)
    
    # We will greedily pick vertices to remove.
    # Strategy:
    # While there is any alive edge:
    #   Find the vertex with the maximum degree (number of alive edges it is in).
    #   Remove it.
    #   Mark all its edges as dead.
    #   Update degrees of neighbors? No, we can just recompute or maintain counters.
    
    # Let's maintain `degree[v]` = count of alive edges containing v.
    degree = {v: len(vertex_edges[v]) for v in range(1, n + 1)}
    
    # However, when an edge dies, we need to decrement degrees of its vertices.
    # This requires iterating over the vertices of the edge.
    
    while True:
        if time.time() > deadline - 5.0:
            break
            
        # Find a vertex with max degree > 0
        max_deg = 0
        best_v = None
        for v in range(1, n + 1):
            if degree[v] > max_deg:
                max_deg = degree[v]
                best_v = v
                
        if best_v is None or max_deg == 0:
            break
            
        # Remove best_v
        removed.add(best_v)
        
        # Kill all edges containing best_v
        for edge_idx in vertex_edges[best_v]:
            if edge_alive[edge_idx]:
                edge_alive[edge_idx] = False
                # Update degrees of other vertices in this edge
                for v in active_edges[edge_idx]:
                    if v != best_v:
                        degree[v] -= 1
        
        # Optimization: If max_deg becomes 0 for all, break.
        # The loop condition handles this.
        
    # The remaining active vertices form a valid set (independent set).
    # Because all edges (bad triples) have been hit by at least one removed vertex.
    # So no bad triple remains entirely in the active set.
    
    result_set = set(range(1, n + 1)) - removed
    
    # Verify the result
    # It should be valid by construction.
    # Let's double check just in case.
    
    valid = True
    S = list(result_set)
    S.sort()
    for i in range(len(S)):
        if time.time() > deadline - 1.0:
            break
        for j in range(i + 1, len(S)):
            lab = lcm(S[i], S[j])
            for k in range(j + 1, len(S)):
                if lcm(S[i], S[k]) == lab and lcm(S[j], S[k]) == lab:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: return prime powers + 1
        def get_primes(limit):
            sieve = [True] * (limit + 1)
            sieve[0] = sieve[1] = False
            for i in range(2, int(limit**0.5) + 1):
                if sieve[i]:
                    for j in range(i*i, limit + 1, i):
                        sieve[j] = False
            return [i for i, is_prime in enumerate(sieve) if is_prime]
        
        primes = get_primes(n)
        prime_powers = set()
        for p in primes:
            power = p
            while power <= n:
                prime_powers.add(power)
                power *= p
        return list(prime_powers | {1})
        
    return list(result_set)

