# EVOLVE-BLOCK-START
import time

def _has_c5(adj, n):
    """Check if graph has a 5-cycle (C5) as a subgraph."""
    # For each pair (u,v) of non-adjacent vertices, check if there's a path of length 3 between them
    # If u~a~b~c~v with all distinct, that's a C5
    # More efficiently: for each vertex u, look at pairs of non-neighbors a,b of u
    # Check if there's a path a - x - y - b where x,y are neighbors of u
    # Actually, C5 through u: u - a - b - c - d - u
    # This means a and d are neighbors of u, and there's a path a-b-c-d of length 3
    
    # Simpler approach: for each ordered pair (a,b) of vertices,
    # check if there exist distinct c,d such that a-c, c-d, d-b are edges
    # and a,b are non-adjacent, and all 5 vertices distinct.
    # That gives a 5-cycle a-c-d-b-... wait, that's only 4 vertices if a~b.
    
    # Let me think again. A C5 is a cycle of length 5.
    # For each vertex u, consider its neighbors N(u).
    # A C5 through u has the form u - v1 - v2 - v3 - v4 - u
    # where v1, v4 in N(u), and v1-v2-v3-v4 is a path of length 3 not using u.
    
    # For each u:
    #   For each pair (v1, v4) in N(u):
    #     Check if there's a path v1 - v2 - v3 - v4 of length 3
    #     where v2, v3 are not u, and all vertices distinct.
    
    # Even simpler: iterate over all 5-tuples? Too slow for n=33.
    
    # Let's use: for each pair of non-adjacent vertices (a, b),
    # check if there are vertices c, d such that a-c, c-d, d-b are edges,
    # and a, b, c, d are all distinct, and a-d and b-c are not edges (to ensure it's a simple cycle).
    # Actually, if a-c-d-b forms a path of length 3, and a~b is false,
    # then a-c-d-b-a would be a 4-cycle, not 5.
    
    # I need to be more careful. A C5 has 5 vertices.
    # Let's enumerate: for each vertex u, for each pair of non-adjacent vertices (a, b) 
    # both in N(u) [wait, a and b must be neighbors of u for u-a-...-b-u to be a cycle through u]
    
    # Actually, in a C5 through u: u is connected to two vertices, say a and e.
    # The cycle is u-a-b-c-e-u.
    # So a and e are neighbors of u, and a-b-c-e is a path of length 3.
    
    # For each u:
    #   For each pair (a, e) in N(u):
    #     For each b in N(a) \ {u}:
    #       For each c in N(b) \ {u, a}:
    #         If c ~ e and c != u and c != a and c != b:
    #           Check all 5 vertices distinct: u, a, b, c, e
    #           If distinct, we have a C5.
    
    # This is O(n * deg^4) which could be slow but n=33 is small.
    
    for u in range(n):
        nu = adj[u]
        for i, a in enumerate(nu):
            for e in nu[i+1:]:
                # Path a - b - c - e
                for b in adj[a]:
                    if b == u or b == e:
                        continue
                    for c in adj[b]:
                        if c == u or c == a or c == e:
                            continue
                        if c in adj[e]:
                            # Check all distinct
                            if len({u, a, b, c, e}) == 5:
                                return True
    return False

def _max_independent_set_size(adj, n):
    """Find the size of the maximum independent set."""
    # Use a simple branch and bound
    best = [0]
    
    def branch(candidates, current_size):
        if current_size + len(candidates) <= best[0]:
            return
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Pick a vertex
        v = candidates[0]
        remaining = candidates[1:]
        
        # Branch 1: include v
        # Remove v and all its neighbors from candidates
        new_candidates = [x for x in remaining if x not in adj[v]]
        branch(new_candidates, current_size + 1)
        
        # Branch 2: exclude v
        branch(remaining, current_size)
    
    branch(list(range(n)), 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Start with the known valid 32-vertex graph: 8 disjoint K4s
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    # Try to extend to 33 vertices
    # Strategy: add vertex 32 and connect it to some existing vertices
    # The 32-vertex graph has 8 components, each K4.
    # Adding a new vertex v=32, we need to connect it to some vertices.
    # To keep alpha <= 8, v must be connected to enough vertices so that
    # we can't pick v plus one from each K4.
    
    # If v is isolated, alpha becomes 9 (pick v + one from each K4).
    # If v is connected to one vertex in each of 8 K4s, then picking v means
    # we can't pick those 8 vertices, but we can still pick one from each K4 
    # (the other 3 in each K4 are available). So alpha is still 9.
    
    # Actually, if v is connected to exactly one vertex in each K4, then:
    # - If we pick v, we can pick at most 1 from each K4 (the non-neighbor), so 1+8=9. Bad.
    # - If we don't pick v, we pick 1 from each K4, so 8.
    # So alpha = 9. Not valid.
    
    # We need v to be connected to at least 2 vertices in some K4s to reduce the max.
    # If v is connected to 2 vertices in one K4, then picking v means we can pick
    # 0 from that K4 (since the other 2 are also in the K4 and connected to the 2 neighbors? 
    # No, in K4, all vertices are connected. So if v is connected to 2 vertices in a K4,
    # those 2 are not available, but the other 2 in the K4 are still available and 
    # connected to the 2 neighbors. But we're picking an independent set, so we can pick
    # at most 1 from the K4 anyway. So picking v + 1 from each K4 = 1+8=9 still.
    
    # Hmm, this approach isn't working. Let me think differently.
    
    # What if we modify the structure? Instead of 8 K4s, use 7 K4s and one larger component.
    # For example, 7 K4s (28 vertices) + one component on 5 vertices with alpha=1 and no C5.
    # A K5 has alpha=1 and no C5. So 7 K4s + 1 K5 = 28+5=33 vertices.
    # Alpha = 7*1 + 1 = 8. No C5.
    
    # Let's try this!
    n = 33
    edges = []
    # 7 K4s
    for b in range(7):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    # 1 K5 on vertices 28-32
    base = 28
    for i in range(5):
        for j in range(i + 1, 5):
            edges.append([base + i, base + j])
    
    # Verify
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    if _has_c5(adj, n):
        # Fallback to 32-vertex graph
        return {'n': 32, 'edges': [
            [4*b+i, 4*b+j] for b in range(8) for i in range(4) for j in range(i+1, 4)
        ]}
    
    alpha = _max_independent_set_size(adj, n)
    if alpha >= 9:
        # Fallback to 32-vertex graph
        return {'n': 32, 'edges': [
            [4*b+i, 4*b+j] for b in range(8) for i in range(4) for j in range(i+1, 4)
        ]}
    
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
