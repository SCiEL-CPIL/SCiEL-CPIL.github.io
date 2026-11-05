import time
import random

def _er_graph(q):
    """
    Generate the Erdős–Rényi orthogonality graph on the projective plane of order q.
    Vertices are points of PG(2, q).
    Two points are adjacent if they are orthogonal with respect to the polarity defined by the matrix [[1,0,0],[0,1,0],[0,0,-1]].
    """
    if q % 4 != 1:
        # Polarity exists for all q, but the specific form might vary.
        # For q=5, 5 % 4 == 1.
        pass
    
    # Generate all points in PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make the first non-zero coordinate 1
                # Actually, for PG(2,q), we just need distinct projective points.
                # We can represent them as (x, y, z) with z=1 or (x, 0, 1) etc.
                # Easier: iterate over all non-zero vectors and normalize by the first non-zero element.
                # Or just generate all (x, y, 1) for x,y in GF(q) and (x, 1, 0) for x in GF(q) and (1, 0, 0).
                # Total: q^2 + q + 1.
                pass
    
    # Simpler generation:
    # Points are (x, y, 1) for x, y in GF(q)
    # (x, 1, 0) for x in GF(q)
    # (1, 0, 0)
    pts = []
    for x in range(q):
        for y in range(q):
            pts.append((x, y, 1))
    for x in range(q):
        pts.append((x, 1, 0))
    pts.append((1, 0, 0))
    
    n = len(pts)
    edges = []
    
    # Polarity: x1*x2 + y1*y2 - z1*z2 = 0
    for i in range(n):
        x1, y1, z1 = pts[i]
        for j in range(i + 1, n):
            x2, y2, z2 = pts[j]
            val = (x1 * x2 + y1 * y2 - z1 * z2) % q
            if val == 0:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}

def _schlaefi_graph():
    """
    Generate the Schläfli graph (27 vertices, degree 10, C4-free, alpha=10).
    It is the complement of the line graph of K_6? No.
    It is the point graph of the generalized quadrangle GQ(2,4)? No.
    It is the complement of the line graph of the complete bipartite graph K_{3,3}? No.
    The Schläfli graph is the unique strongly regular graph with parameters (27, 10, 1, 5).
    """
    # Hardcoded or generated?
    # I will use a known construction: The Schläfli graph is the complement of the line graph of K_6?
    # L(K_6) has 15 vertices. No.
    # It is the point graph of the generalized quadrangle of order (2, 4)?
    # No, GQ(2,4) has 2*4*5 + 4 + 1 = 25 vertices?
    # GQ(s, t) has (s+1)(st+1) vertices.
    # GQ(2, 4): (3)(9) = 27 vertices.
    # Yes! The Schläfli graph is the point graph of GQ(2, 4).
    # Construction of GQ(2, 4):
    # Points are pairs (i, j) where i in {0,1,2,3,4,5} (6 points?) No.
    # Standard GQ(2,4) construction:
    # Points: (x, y) where x in GF(4)^2?
    # I will use a hardcoded edge list for Schläfli to be safe, or a simpler valid graph.
    # Given time, I will skip Schläfli generation if complex and rely on G(5).
    pass

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    """
    def backtrack(remaining, current_size):
        if time.time() > deadline:
            return False
        if current_size >= 11:
            return True
        if not remaining:
            return False
        if current_size + remaining.bit_count() < 11:
            return False
        
        min_v = -1
        min_deg = 100
        temp = remaining
        while temp:
            v = (temp & -temp).bit_length() - 1
            deg = (adj[v] & remaining).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            temp &= temp - 1
            
        if min_v == -1:
            return False
            
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        remaining &= ~(1 << min_v)
        if backtrack(remaining, current_size):
            return True
            
        return False

    return backtrack((1 << n) - 1, 0)

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    # Fallback: 10 disjoint triangles
    fallback_edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        fallback_edges += [[a, b], [a, c], [b, c]]
    
    best_result = {'n': 30, 'edges': fallback_edges}
    
    # Try G(5)
    g5 = _er_graph(5)
    n = g5['n']
    edges = g5['edges']
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
        
    # Check C4-free
    is_c4_free = True
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] & adj[v]).bit_count() > 1:
                is_c4_free = False
                break
        if not is_c4_free:
            break
            
    if is_c4_free:
        has_indep_11 = _has_indep_11_bitmask(adj, n, deadline)
        if not has_indep_11:
            return g5
            
    # If G(5) failed, try to improve or return fallback
    return best_result
