import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    n = 31
    m = 91  # number of edges required
    
    # Use a greedy randomized local search approach
    # We want to minimize the number of C4s (4-cycles)
    
    # First, let's use a structured approach based on projective geometry
    # PG(2,5) has 31 points. The polarity graph of PG(2,5) has 31 vertices
    # and each vertex has degree 6 (since q+1 = 6 for q=5).
    # That gives 31*6/2 = 93 edges, which is 2 more than 91.
    
    # Let's construct the polarity graph of PG(2,5) and then remove 2 edges
    # to get exactly 91 edges, choosing which edges to remove to minimize C4s.
    
    # Build PG(2,5) points as 1D subspaces of GF(5)^3
    q = 5
    points = []
    seen = set()
    for a in range(q):
        for b in range(q):
            for c in range(q):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize: make first non-zero coordinate 1
                vec = [a, b, c]
                first_nonzero = -1
                for i in range(3):
                    if vec[i] != 0:
                        first_nonzero = i
                        break
                inv = pow(vec[first_nonzero], q-2, q)
                vec = [(x * inv) % q for x in vec]
                key = tuple(vec)
                if key not in seen:
                    seen.add(key)
                    points.append(vec)
    
    # There should be q^2 + q + 1 = 25 + 5 + 1 = 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Polarity: a point p=(x,y,z) is perpendicular to q'=(x',y',z') if xx'+yy'+zz' = 0 mod 5
    # The polarity graph connects p and q if p is perpendicular to q (and p != q)
    
    def perp(p, qpt):
        return (p[0]*qpt[0] + p[1]*qpt[1] + p[2]*qpt[2]) % q == 0
    
    # Build adjacency
    adj = [[False]*n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if perp(points[i], points[j]):
                adj[i][j] = adj[j][i] = True
                edges.append((i, j))
    
    # Count edges in polarity graph
    polarity_edges = len(edges)
    # Should be 93 (each vertex has degree 6, so 31*6/2 = 93)
    
    if polarity_edges >= m:
        # We need to remove (polarity_edges - m) edges to get exactly m edges
        # Choose which edges to remove to minimize C4s
        # Start with all polarity edges, then greedily remove edges that are in the most C4s
        
        # First, count C4s for each edge
        # An edge (u,v) is in a C4 if there exists w such that u-w and v-w form a path of length 2
        # More precisely, (u,v) is in a C4 if there exist w,x such that u-w-v-x-u forms a cycle
        # Actually, (u,v) is in a C4 if there is a vertex w such that w is adjacent to both u and v,
        # and w != u, w != v. Then for each such w, we need another vertex x adjacent to both u and v
        # with x != w. Wait, that's not quite right.
        
        # A C4 containing edge (u,v): u-w-x-v-u where w,x are distinct from u,v and from each other.
        # So we need common neighbors of u and v (excluding each other... no, w and x are neighbors of u and v respectively)
        # Actually: u is connected to w, w to x, x to v, v to u.
        # So w is a neighbor of u (w != v), x is a neighbor of v (x != u), and w-x is an edge, w != x.
        
        # Alternative: (u,v) is in a C4 iff there exist distinct w,x such that:
        # u~w, w~x, x~v, and w != x, w != u, w != v, x != u, x != v
        # This means w is in N(u) \ {v}, x is in N(v) \ {u}, and w~x.
        
        # For efficiency, let's precompute common neighbors
        # For each pair (u,v), count the number of C4s containing edge (u,v)
        
        # Let's use a different approach: count C4s per edge using matrix multiplication
        # A^2[i][j] = number of common neighbors of i and j
        # A C4 is u-v-w-x-u. The number of C4s = (1/8) * sum over all u,v of (A^2[u][v] - A[u][v]) * (A^2[v][u] - A[v][u])... 
        # Actually, number of C4s = (1/8) * trace(A^4) - ... this gets complicated.
        
        # Simpler: for each edge (u,v), the number of C4s it belongs to equals the number of pairs (w,x) 
        # such that w in N(u)\{v}, x in N(v)\{u}, w~x, w!=x.
        # This is sum over w in N(u)\{v} of |N(w) ∩ N(v) \ {u}|
        
        # Let's compute this for all edges
        A = np.array(adj, dtype=np.int64)
        
        # For each edge, count C4s
        edge_c4_count = {}
        for (u, v) in edges:
            count = 0
            for w in range(n):
                if w == u or w == v:
                    continue
                if adj[u][w]:
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[v][x] and adj[w][x]:
                            count += 1
            edge_c4_count[(u,v)] = count
        
        # Greedily remove edges with highest C4 count
        current_edges = set(edges)
        current_adj = [row[:] for row in adj]
        
        to_remove = polarity_edges - m
        removed = []
        
        for _ in range(to_remove):
            # Find edge with max C4 count
            best_edge = None
            best_count = -1
            for e in current_edges:
                u, v = e
                # Recompute C4 count for this edge in current graph
                count = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if current_adj[u][w]:
                        for x in range(n):
                            if x == u or x == v or x == w:
                                continue
                            if current_adj[v][x] and current_adj[w][x]:
                                count += 1
                if count > best_count:
                    best_count = count
                    best_edge = e
            
            if best_edge is not None:
                current_edges.remove(best_edge)
                removed.append(best_edge)
                u, v = best_edge
                current_adj[u][v] = current_adj[v][u] = False
        
        # Build final edge list
        final_edges = list(current_edges)
        
        # Verify we have exactly m edges
        assert len(final_edges) == m, f"Expected {m} edges, got {len(final_edges)}"
        
        # Verify all edges are distinct and valid
        for e in final_edges:
            u, v = e
            assert 0 <= u < n and 0 <= v < n
            assert u != v
        
        # Convert to list of lists
        result = [[u, v] for u, v in final_edges]
        return result
    
    # Fallback: use random construction
    # This shouldn't happen, but just in case
    random.seed(42)
    all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_possible)
    return [[i, j] for i, j in all_possible[:m]]
# EVOLVE-BLOCK-END
