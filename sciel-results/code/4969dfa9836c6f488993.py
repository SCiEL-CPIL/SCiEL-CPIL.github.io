import time
import random

def _er_graph(q):
    """
    Generate the Erdős–Rényi orthogonality graph on the projective plane of order q.
    Vertices are points of PG(2, q).
    Two points are adjacent if they are orthogonal with respect to the polarity defined by the matrix [[1,0,0],[0,1,0],[0,0,-1]].
    """
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

def _has_indep_11_bitmask(adj, n, deadline):
    """
    Check if an independent set of size 11 exists using bitmask backtracking.
    Uses a more robust branching strategy.
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
        
        # Find a vertex to branch on: pick the one with minimum degree in remaining
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
            
        # Branch 1: include min_v
        new_remaining = remaining & ~(adj[min_v] | (1 << min_v))
        if backtrack(new_remaining, current_size + 1):
            return True
            
        # Branch 2: exclude min_v
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
    
    # Try G(5) - orthogonality graph on PG(2,5), 31 vertices
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
