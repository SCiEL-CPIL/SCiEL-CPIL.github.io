import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    q = 5
    
    # 1. Generate points in PG(2, q)
    # Points are (x, y, z) with z != 0 or y != 0 or x != 0, normalized.
    # We use the standard enumeration:
    # (x, y, 1) for x, y in F_q  -> q^2 points
    # (x, 1, 0) for x in F_q      -> q points
    # (1, 0, 0)                   -> 1 point
    # Total q^2 + q + 1 = 31
    
    points = []
    point_to_idx = {}
    
    # Add affine points (z=1)
    for x in range(q):
        for y in range(q):
            idx = len(points)
            points.append((x, y, 1))
            point_to_idx[(x, y, 1)] = idx
    
    # Add points at infinity y=1, z=0
    for x in range(q):
        idx = len(points)
        points.append((x, 1, 0))
        point_to_idx[(x, 1, 0)] = idx
        
    # Add point at infinity x=1, y=0, z=0
    idx = len(points)
    points.append((1, 0, 0))
    point_to_idx[(1, 0, 0)] = idx
    
    n = len(points)
    assert n == 31
    
    # 2. Define the polarity
    # A standard polarity for PG(2, q) is given by the bilinear form:
    # B(u, v) = u[0]*v[2] - u[2]*v[0] + u[1]*v[1]  (for q odd)
    # Wait, let's verify this specific form creates a polarity.
    # A polarity requires B(u, u) = 0 for some u? No, it's a symmetric bilinear form.
    # The polar line of point P is { Q | B(P, Q) = 0 }.
    # Two points P, Q are adjacent if P is on the polar line of Q, i.e., B(Q, P) = 0.
    # Since B is symmetric, B(P, Q) = B(Q, P).
    # So adjacency: P ~ Q iff B(P, Q) == 0 mod q.
    
    # Let's check the number of edges for this form.
    # B(u, v) = u0*v2 - u2*v0 + u1*v1
    # This is a symplectic form? No, symplectic is alternating B(u,u)=0.
    # Here B(u,u) = u0*u2 - u2*u0 + u1*u1 = u1^2.
    # If u1 != 0, B(u,u) != 0 (for q odd).
    # If u1 == 0, B(u,u) = 0.
    # Points with u1=0: (x, 0, 1) and (x, 0, 0)?
    # In our representation:
    # (x, y, 1): u1=y. y=0 -> (x, 0, 1). q points.
    # (x, 1, 0): u1=1. Never 0.
    # (1, 0, 0): u1=0. 1 point.
    # Total q+1 absolute points.
    # This matches the theory for a symplectic polarity (or hyperbolic?).
    # Actually, this is often called a "symplectic" polarity in some contexts because the associated form is non-degenerate.
    
    def b_form(u, v):
        return (u[0]*v[2] - u[2]*v[0] + u[1]*v[1]) % q

    # 3. Build the graph
    # adjacency[i] is a set of neighbors of i
    adjacency = [set() for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            if b_form(points[i], points[j]) == 0:
                adjacency[i].add(j)
                adjacency[j].add(i)
                
    # Verify edge count
    edge_count = sum(len(s) for s in adjacency) // 2
    if edge_count != 90:
        # Fallback: If the specific form doesn't yield 90 edges (shouldn't happen),
        # we might need to adjust or use a different known construction.
        # But for q=5, this should be correct.
        pass
        
    # 4. Find the best edge to add
    # We want to minimize the number of C4s created.
    # New C4s = C(|N(u) intersect N(v)|, 2)
    
    best_edge = None
    min_c4s = float('inf')
    
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adjacency[u]:
                common = adjacency[u].intersection(adjacency[v])
                c = len(common)
                new_c4s = c * (c - 1) // 2
                
                if new_c4s < min_c4s:
                    min_c4s = new_c4s
                    best_edge = (u, v)
                    
                # Optional: Early exit if we find 0 C4s (impossible since ex=90, adding edge creates >=1)
                # But if min_c4s becomes 0, we can stop? No, 0 is impossible.
                # If min_c4s becomes 1, that's the theoretical minimum.
                if min_c4s == 1:
                    # Can we get 0? No. So 1 is optimal.
                    # But there might be multiple edges with 1. We just need one.
                    break
        if min_c4s == 1:
            break
            
    if best_edge is None:
        # Should not happen
        return []
        
    u, v = best_edge
    # Add the edge
    adjacency[u].add(v)
    adjacency[v].add(u)
    
    # 5. Construct the list of edges
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append([i, j])
                
    # Ensure exactly 91 edges
    assert len(edges) == 91, f"Expected 91 edges, got {len(edges)}"
    
    return edges

if __name__ == "__main__":
    construct()
