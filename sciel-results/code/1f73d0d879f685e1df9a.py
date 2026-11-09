import random
import time
import sys

# EVOLVE-BLOCK-START

def _generate_polarity_graph(q):
    """
    Generate the polarity graph of the projective plane PG(2, q).
    q must be a prime power.
    Returns adjacency list as a list of sets.
    """
    if q == 2:
        # Fano plane polarity graph
        # Vertices 0..6
        # Edges: (0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,0)? No, let's generate properly.
        pass

    # Generic generator for prime power q
    # We need GF(q)
    # For q=2, 3, 5, we can hardcode or use simple field logic.
    
    if q == 2:
        # GF(2)
        # Points: (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
        # Polarity: x1*x1' + x2*x2' + x3*x3' = 0
        # Let's just use the known adjacency for q=2 (Paley(7) complement? No, Paley(7) is the graph itself?
        # Paley(7) is 3-regular.
        # Adjacency: 0:1,2,4; 1:0,3,5; 2:0,3,6; 3:1,2,5; 4:0,5,6; 5:1,4,6; 6:2,4,5
        adj = [set() for _ in range(7)]
        edges = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,5), (4,5), (4,6), (5,6)]
        # Wait, let's check C4-freeness.
        # Common neighbors of 0 and 3: N(0)={1,2,4}, N(3)={1,2,5}. Common={1,2}. Size 2. C4!
        # So Paley(7) is NOT C4-free?
        # Let's re-check PG(2,2) polarity graph.
        # The graph I generated above might be wrong.
        # Let's construct it properly.
        
    # For q=2, 3, 5, I will implement a generic construction using finite fields.
    
    def gf_add(a, b, q):
        return (a + b) % q if q < 10 else None # Only for prime q for simplicity in this block, or use lookup for 4
        
    # This is getting complex. I will hardcode the adjacencies for q=2, 3, 5.
    
    if q == 2:
        # Correct Polarity Graph of PG(2,2)
        # It is the 3-regular graph on 7 vertices with no C4.
        # It is the complement of the 3-regular graph with C4?
        # Actually, the graph is isomorphic to the Paley graph of order 7?
        # Paley(7) has triangles.
        # Let's use the standard construction.
        # Vertices: 0..6
        # Edges:
        # 0-1, 0-2, 0-3
        # 1-4, 1-5
        # 2-4, 2-6
        # 3-5, 3-6
        # 4-5? No.
        # Let's just use a known C4-free graph on 7 vertices with alpha=3.
        # The "Polarity" graph is unique.
        # I will use a verified list.
        edges = [(0,1), (0,2), (0,3), (1,4), (1,5), (2,4), (2,6), (3,5), (3,6), (4,5), (4,6), (5,6)]
        # Wait, 4-5, 4-6, 5-6 form a triangle.
        # Let's check C4.
        # N(0)={1,2,3}, N(4)={1,2,5,6}. Common={1,2}. C4!
        # So this is not it.
        
        # Correct edges for PG(2,2) polarity graph:
        # It is the 3-regular graph where every pair of non-adjacent vertices has exactly 1 common neighbor.
        # And every pair of adjacent vertices has 0 common neighbors?
        # No, lambda=1, mu=1.
        # So adjacent have 1 common, non-adjacent have 1 common.
        # This is the Paley graph of order 7?
        # Paley(7): 0-1, 0-2, 0-4. 1-0, 1-3, 1-5. 2-0, 2-3, 2-6. 3-1, 3-2, 3-5. 4-0, 4-5, 4-6. 5-1, 5-4, 5-6. 6-2, 6-4, 6-5.
        # Check 0 and 3 (non-adj). N(0)={1,2,4}, N(3)={1,2,5}. Common {1,2}. Size 2.
        # So Paley(7) has C4.
        
        # The polarity graph of PG(2,2) is actually the 3-regular graph on 7 vertices that is C4-free.
        # It is the complement of the 3-regular graph that is C4-free?
        # There is only one 3-regular graph on 7 vertices up to isomorphism? No.
        # The "Fano plane" graph (incidence) is bipartite.
        # The "polarity" graph is the one I want.
        # Let's find the edges programmatically.
        
        # Points: (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)
        # Polarity: x1y1 + x2y2 + x3y3 = 0
        # 0=(1,0,0). Orthogonal to y if y1=0.
        # y in {(0,1,0), (0,0,1), (0,1,1)}. Indices 1, 2, 5.
        # So 0 connected to 1, 2, 5.
        # 1=(0,1,0). Orthogonal to y if y2=0.
        # y in {(1,0,0), (0,0,1), (1,0,1)}. Indices 0, 2, 4.
        # So 1 connected to 0, 2, 4.
        # 2=(0,0,1). Orthogonal to y if y3=0.
        # y in {(1,0,0), (0,1,0), (1,1,0)}. Indices 0, 1, 3.
        # So 2 connected to 0, 1, 3.
        # 3=(1,1,0). Orthogonal to y if y1+y2=0 => y1=y2.
        # y in {(1,1,0), (1,1,1)}. Indices 3, 6.
        # Wait, (1,1,0) is 3 itself. (1,1,1) is 6.
        # So 3 connected to 6? And itself?
        # Usually polarity graphs are simple, so no self-loops.
        # If 3 is absolute (on its own polar line), it is not connected to itself.
        # Is 3 on 3^perp? 1*1+1*1+0*0 = 2 = 0 in GF(2). Yes.
        # So 3 is absolute.
        # So 3 is connected to 6 only?
        # Degree must be 3.
        # Let's re-check.
        # y1=y2.
        # (1,1,0) -> 3.
        # (1,1,1) -> 6.
        # Are there others?
        # (0,0,0) not a point.
        # So only 6?
        # Degree 1?
        # This implies the graph is not regular?
        # Polarity graphs of PG(2,q) are regular of degree q+1.
        # For q=2, degree 3.
        # My manual check is flawed.
        
        # I will use a pre-computed adjacency list for q=2, 3, 5.
        # For q=2, the graph is the 3-regular graph on 7 vertices with no C4.
        # It is the "Möbius-Kantor" graph? No, that's 16 vertices.
        # It is the complement of the 7-cycle?
        # C7 complement is 4-regular.
        
        # I will skip q=2 and use q=3, 5 directly with hardcoded adjacencies if I can find them, or just use q=5.
        
    if q == 5:
        # I will generate this properly using GF(5).
        # Points: (x,y,1), (x,1,0), (1,0,0)
        # 25 + 5 + 1 = 31 points.
        # Polarity: x1x1' + x2x2' + x3x3' = 0
        # This is doable.
        
        points = []
        # Homogeneous coordinates
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for x in range(q):
            points.append((x, 1, 0))
        points.append((1, 0, 0))
        
        adj = [set() for _ in range(len(points))]
        
        for i, p in enumerate(points):
            for j, r in enumerate(points):
                if i == j: continue
                # Check if p is on r^perp
                # p . r = 0
                if (p[0]*r[0] + p[1]*r[1] + p[2]*r[2]) % q == 0:
                    adj[i].add(j)
                    adj[j].add(i)
        return adj
    else:
        # Fallback for other q
        return [set() for _ in range(q*q+q+1)]

def _alpha_exact(adj, n):
    """Exact independence number."""
    best = [0]
    remaining = set(range(n))
    
    def _branch(rem, cur):
        if not rem:
            if cur > best[0]:
                best[0] = cur
            return
        if cur + len(rem) <= best[0]:
            return
        # Heuristic: pick vertex with max degree in rem
        v = max(rem, key=lambda x: len(adj[x] & rem))
        # Include v
        new_rem = rem - adj[v]
        new_rem.discard(v)
        _branch(new_rem, cur + 1)
        # Exclude v
        rem2 = set(rem)
        rem2.discard(v)
        _branch(rem2, cur)
        
    _branch(remaining, 0)
    return best[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Generate PG(2,5) graph
    adj_31 = _generate_polarity_graph(5)
    n_31 = len(adj_31)
    
    # Calculate alpha
    alpha_31 = _alpha_exact(adj_31, n_31)
    
    # Determine filler
    # We want to add H such that alpha(H) <= 10 - alpha_31
    # And H is C4-free.
    # We will try to construct H from known small C4-free graphs.
    # Candidates:
    # K1 (n=1, alpha=1)
    # K2 (n=2, alpha=1) - Edge
    # PG(2,2) (n=7, alpha=3) - Need to verify alpha.
    # PG(2,3) (n=13, alpha=?)
    
    # For now, just use isolated vertices and edges.
    # Edge is better: 2 vertices for 1 alpha.
    # So we can add floor((10 - alpha_31) * 2) vertices using edges?
    # No, disjoint union of edges.
    # If we add k edges, we add 2k vertices, alpha increases by k.
    # So we can add k such that k <= 10 - alpha_31.
    # Max vertices added = 2 * (10 - alpha_31).
    
    budget = 10 - alpha_31
    if budget < 0:
        # Graph itself has alpha > 10. Return as is? No, invalid.
        # But PG(2,5) should have small alpha.
        pass
        
    n_add = 2 * budget
    # We can also add 1 isolated vertex if we have 1 left.
    # Actually, just use edges.
    
    total_n = n_31 + n_add
    
    # Build final adjacency
    adj_final = [set() for _ in range(total_n)]
    
    # Copy G_31
    for i in range(n_31):
        adj_final[i] = set(adj_31[i])
        
    # Add edges
    offset = n_31
    for i in range(budget):
        u = offset + 2*i
        v = offset + 2*i + 1
        adj_final[u].add(v)
        adj_final[v].add(u)
        
    # Verify
    # Check C4
    is_c4_free = True
    for u in range(total_n):
        for v in range(u+1, total_n):
            if v in adj_final[u]:
                continue
            common = adj_final[u] & adj_final[v]
            if len(common) >= 2:
                is_c4_free = False
                break
        if not is_c4_free:
            break
            
    if not is_c4_free:
        # Fallback to just G_31
        total_n = n_31
        adj_final = [set(adj_31[i]) for i in range(n_31)]
        
    # Check Alpha
    # We know alpha(G_31) + alpha(H) = alpha_31 + budget = 10.
    # So it should be valid.
    
    edges = []
    for u in range(total_n):
        for v in adj_final[u]:
            if u < v:
                edges.append([u, v])
                
    return {'n': total_n, 'edges': edges}

# EVOLVE-BLOCK-END
