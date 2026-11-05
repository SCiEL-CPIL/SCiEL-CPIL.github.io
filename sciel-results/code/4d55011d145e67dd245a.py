"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """Construct a graph with 31 vertices and 91 edges minimizing C4 count.
    
    Strategy: Build the polarity graph (C4-free, 90 edges) using finite geometry
    for q=5, then add the 91st edge that creates the fewest new C4s.
    Then perform local search swaps to further reduce C4 count.
    """
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n = 31
    q = 5
    
    # Build polarity graph on PG(2, 5)
    # Vertices: points of PG(2,5) = (25 affine + 5 line-at-infinity + 1 point-at-infinity)
    # Actually, let's use the standard construction:
    # For odd q, the polarity graph has q^2 + q + 1 vertices and q(q+1)/2 * 2... 
    # Wait, let me reconsider. For q=5: n = 31 = 5^2 + 5 + 1
    # The polarity graph (Erdos-Renyi) has n = q^2 + q + 1 vertices
    # Each vertex has degree q+1 = 6, so total edges = 31*6/2 = 93? No...
    # Actually for the orthogonality graph: vertices are points of PG(2,q),
    # two vertices adjacent if they are orthogonal under a polarity.
    # For q odd, each point is self-orthogonal to exactly one other point? No.
    # Let me use a concrete construction.
    
    # Standard construction for q=5:
    # Label vertices 0..30
    # Use the fact that we can build a C4-free graph with 90 edges
    # One approach: use a known C4-free graph construction
    
    # Let's build using the incidence structure of PG(2,5)
    # Points: (x, y) for x,y in GF(5) -> 25 points, labeled 0..24
    # Points at infinity: one for each slope m in GF(5), labeled 25..29
    # Plus the point (0,0,1) at infinity... 
    
    # Actually, let me use a simpler known C4-free graph:
    # The polarity graph construction:
    # For GF(5), define vertices as pairs (a,b) where a,b in {0,1,2,3,4}
    # That's 25 vertices. We need 31.
    
    # Let me use the standard finite geometry approach:
    # PG(2,5) has 31 points. The polarity sigma: (x,y,z) -> (x, y, -x*y - z) or similar
    # Two points are adjacent if they are orthogonal under the polarity.
    
    # Let me just use a direct construction that I know works:
    # For q=5, construct a graph where:
    # - 25 "affine" vertices labeled 0..24, corresponding to (i,j) in GF(5)^2
    # - 5 "infinity" vertices labeled 25..29, corresponding to slopes m in GF(5)
    # - 1 "extra" vertex labeled 30
    
    # Adjacency rules (this gives a C4-free graph with 90 edges):
    # Actually, let me use a well-known construction.
    
    # The Erdos-Renyi graph for q=5:
    # Vertices: all points of PG(2,5)
    # Edge between P and Q if P lies on the polar line of Q
    
    # Let me implement this properly:
    # Points of PG(2,5) can be represented as [x:y:z] with (x,y,z) != (0,0,0)
    # up to scalar. We normalize: if z != 0, use [x/z : y/z : 1], else [x/y : 1 : 0]
    
    # For q=5, the field elements are 0,1,2,3,4 with arithmetic mod 5.
    
    # Let me enumerate all 31 points:
    # Affine points: (a, b) for a,b in GF(5) -> 25 points, index = a*5 + b
    # Points at infinity: one for each direction (a,b) with gcd(a,b)=1, up to sign
    # Directions in GF(5)^2 \ {(0,0)} up to scalar: there are (5^2-1)/(5-1) = 24/4 = 6
    # Wait, that's 6, not 5. Let me recount.
    # PG(2,5) has (5^3-1)/(5-1) = 124/4 = 31 points. Correct.
    # Affine: 25. At infinity: 6. Total: 31.
    
    # So vertices 0..24 are affine (a,b), vertices 25..30 are at infinity.
    # The 6 points at infinity correspond to 1D subspaces of GF(5)^2:
    # (1,0), (0,1), (1,1), (1,2), (1,3), (1,4) -> 6 directions
    
    # Let me define the polarity. For q odd, a standard polarity is:
    # (x, y, z) is self-orthogonal to (x', y', z') iff x*x' + y*y' + z*z' = 0 (symmetric bilinear form)
    # But we need a polarity (self-adjoint), so the bilinear form B((x,y,z),(x',y',z')) = x*x' + y*y' + z*z'
    # Wait, for a polarity in PG(2,q) with q odd, we need a non-degenerate symmetric bilinear form.
    # The form x*x' + y*y' + z*z' works.
    
    # Two points P=[x:y:z] and Q=[x':y':z'] are adjacent iff B(P,Q) = 0.
    # A point is a self-loop iff B(P,P) = 0, i.e., x^2 + y^2 + z^2 = 0.
    # For q=5, squares mod 5: 0^2=0, 1^2=1, 2^2=4, 3^2=4, 4^2=1
    # So x^2+y^2+z^2 = 0 mod 5 has solutions. E.g., (1,1,2): 1+1+4=6=1 mod 5. No.
    # (1,2,2): 1+4+4=9=4. No. (1,1,1): 3. (1,2,1): 1+4+1=6=1. 
    # (2,2,2): 4+4+4=12=2. (1,1,2): 1+1+4=6=1.
    # Hmm, let me check: does x^2+y^2+z^2=0 have non-trivial solutions mod 5?
    # x^2 in {0,1,4}. Sums of three: 0+0+0=0, 0+0+1=1, 0+0+4=4, 0+1+1=2, 0+1+4=0! 
    # So (0,1,2) gives 0+1+4=5=0. Yes! So there are self-orthogonal points.
    
    # For the graph, we exclude self-loops. So edge between P and Q (P != Q) iff B(P,Q)=0.
    
    # Let me enumerate all 31 points and compute adjacencies.
    
    # Points in homogeneous coordinates, normalized:
    points = []
    # Affine points: [a, b, 1] for a,b in 0..4
    for a in range(5):
        for b in range(5):
            points.append([a, b, 1])
    # Points at infinity: [a, b, 0] where we take representatives
    # (1,0), (0,1), (1,1), (1,2), (1,3), (1,4)
    inf_points = [[1,0,0], [0,1,0], [1,1,0], [1,2,0], [1,3,0], [1,4,0]]
    points.extend(inf_points)
    
    # Now compute adjacency: B(P,Q) = P[0]*Q[0] + P[1]*Q[1] + P[2]*Q[2] mod 5
    adj = [[False]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            b_val = (points[i][0]*points[j][0] + points[i][1]*points[j][1] + points[i][2]*points[j][2]) % 5
            if b_val == 0:
                adj[i][j] = True
                adj[j][i] = True
    
    # Count edges
    edge_count = sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    # For a polarity graph in PG(2,q) with q odd, the number of edges should be
    # q(q+1)/2 * ... let me just count.
    # Each point is orthogonal to q+1=6 other points (including itself if self-orthogonal).
    # So degree is either 5 or 6 depending on self-orthogonality.
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                edges.append([i, j])
    
    # We need exactly 91 edges. If we have 90, add one. If more, we need to remove some.
    # Let me check: for q=5, the polarity graph should have 90 edges (C4-free).
    
    if edge_count < 91:
        # Add edges to reach 91
        # Find non-edges and add the one that creates fewest C4s
        # First, count C4s in current graph
        def count_c4(adj_matrix, n):
            """Count C4 subgraphs."""
            c4_count = 0
            # For each pair of vertices, count common neighbors
            # C4 count = sum over all pairs {u,v} of C(common_neighbors(u,v), 2)
            # But this counts each C4 exactly once if we're careful
            # Actually: number of C4s = (1/2) * sum over all unordered pairs {u,v} of C(d_common(u,v), 2)
            # No wait. Each C4 is counted how many times?
            # A C4 has 4 vertices a,b,c,d. The pairs of opposite vertices are (a,c) and (b,d).
            # So each C4 contributes to C(common_neighbors(a,c), 2) via the pair (b,d) being common neighbors,
            # and to C(common_neighbors(b,d), 2) via (a,c).
            # So total = sum over all pairs {u,v} of C(common_neighbors(u,v), 2) / 2
            # Wait no. Let me think again.
            # For a fixed pair {u,v}, the number of C4s containing u and v as opposite vertices is C(common_neighbors(u,v), 2).
            # Each C4 has exactly 2 pairs of opposite vertices.
            # So total C4s = (1/2) * sum_{all pairs {u,v}} C(common_neighbors(u,v), 2)
            
            common = np.zeros((n, n), dtype=int)
            adj_np = np.array(adj_matrix, dtype=int)
            # common[i][j] = number of common neighbors of i and j
            # common = adj @ adj (diagonal adjusted)
            common = adj_np @ adj_np
            # For i != j, common[i][j] is the number of common neighbors
            # For i == j, common[i][i] is the degree of i
            
            total = 0
            for i in range(n):
                for j in range(i+1, n):
                    cn = common[i][j]
                    if cn >= 2:
                        total += cn * (cn - 1) // 2
            return total // 2  # Wait, is it /2 or not?
            
            # Let me verify: a C4 a-b-c-d-a has opposite pairs (a,c) and (b,d).
            # For pair (a,c): b and d are common neighbors, so C(2,2)=1 is added.
            # For pair (b,d): a and c are common neighbors, so C(2,2)=1 is added.
            # Total contribution: 2. But we want to count this C4 once.
            # So yes, divide by 2.
        
        current_c4 = count_c4(adj, n)
        
        # Find the best edge to add
        best_edge = None
        best_c4_increase = float('inf')
        
        # Precompute common neighbors for efficiency
        adj_np = np.array(adj, dtype=int)
        common = adj_np @ adj_np
        
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for j in range(i+1, n):
                if not adj[i][j]:
                    # Adding edge (i,j) creates C4s: for each common neighbor pair
                    # The new C4s containing edge (i,j): for each pair of common neighbors (k,l) of i and j... 
                    # No wait. Adding edge (i,j) creates C4s of the form i-j-k-? -i where k is neighbor of j and ? is neighbor of i, and ?-k is an edge.
                    # Actually, a new C4 containing edge (i,j) must have the form i-j-a-b-i where a is neighbor of j, b is neighbor of i, and a-b is an edge.
                    # Equivalently: for each a in N(j), for each b in N(i), if a-b is an edge and a != b and a != i and b != j, then i-j-a-b-i is a C4.
                    # But we need a != i (a is neighbor of j, could be i? No, i is not neighbor of j since we're adding the edge).
                    # And b != j similarly.
                    # Also a != b to avoid degenerate cases.
                    
                    # New C4s = number of pairs (a,b) with a in N(j), b in N(i), a!=b, a-b in E
                    # This equals: for each a in N(j), count neighbors b of a that are also in N(i), minus cases where b=i or b=j
                    # b=i: i is in N(i)? No, no self-loops. So b != i automatically.
                    # b=j: j in N(i)? We're adding edge (i,j), so j will be in N(i). But currently j is NOT in N(i).
                    # So in the current graph, j is not a neighbor of i. So b=j is not in N(i) currently.
                    
                    # So new C4s = |{(a,b) : a in N(j), b in N(i), a-b in E, a != b}|
                    # = sum_{a in N(j)} |N(a) ∩ N(i) \ {a}| but a is not in N(i) necessarily...
                    # Simpler: new C4s = sum_{a in N(j)} |N(a) ∩ N(i)| - (number of a in N(j) where a in N(i)) 
                    # Wait, if a is in N(i), then a is a common neighbor, and the pair (a,a) would be counted but a!=a is false.
                    # Let me just compute directly.
                    
                    ni = set()
                    for k in range(n):
                        if adj[i][k] and k != i:
                            ni.add(k)
                    nj = set()
                    for k in range(n):
                        if adj[j][k] and k != j:
                            nj.add(k)
                    
                    new_c4s = 0
                    for a in nj:
                        for b in ni:
                            if a != b and adj[a][b]:
                                new_c4s += 1
                    # Each C4 is counted once here? Let's see: C4 i-j-a-b-i. 
                    # a is in N(j), b is in N(i), a-b in E. Yes, counted once.
                    
                    if new_c4s < best_c4_increase:
                        best_c4_increase = new_c4s
                        best_edge = (i, j)
        
        if best_edge is not None:
            i, j = best_edge
            adj[i][j] = True
            adj[j][i] = True
            edges.append([i, j])
        
        # Now do local search: try swapping edges to reduce C4 count
        # But we must maintain exactly 91 edges
        # A swap: remove edge (u,v), add edge (x,y) where (x,y) is not an edge
        # This keeps edge count at 91
        
        # Let's do a simple local search
        for iteration in range(500):
            if time.time() > deadline - 10:
                break
            
            best_swap = None
            best_c4_after = count_c4(adj, n)
            
            # Try a few random swaps
            for _ in range(200):
                if time.time() > deadline - 10:
                    break
                # Pick a random edge to remove
                e_idx = random.randint(0, len(edges)-1)
                u, v = edges[e_idx]
                # Pick a random non-edge to add
                while True:
                    x = random.randint(0, n-1)
                    y = random.randint(0, n-1)
                    if x < y and not adj[x][y] and (x,y) != (u,v) and (x,y) != (v,u):
                        break
                
                # Try the swap
                adj[u][v] = False
                adj[v][u] = False
                adj[x][y] = True
                adj[y][x] = True
                
                c4_after = count_c4(adj, n)
                
                if c4_after < best_c4_after:
                    best_c4_after = c4_after
                    best_swap = (u, v, x, y)
                
                # Revert
                adj[u][v] = True
                adj[v][u] = True
                adj[x][y] = False
                adj[y][x] = False
            
            if best_swap is not None:
                u, v, x, y = best_swap
                # Apply swap
                adj[u][v] = False
                adj[v][u] = False
                adj[x][y] = True
                adj[y][x] = True
                edges.remove([u, v])
                edges.append([x, y])
            else:
                break
    
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i][j]:
                result.append([i, j])
    
    # Ensure exactly 91 edges
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
# EVOLVE-BLOCK-END
