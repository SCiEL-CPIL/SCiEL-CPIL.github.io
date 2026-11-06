import time
import random

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    q = 6
    # Generate all points in PG(2, q)
    points = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize to unique representative
                # Find first non-zero
                if x != 0:
                    inv = pow(x, q-2, q)
                    points.append((1, (y*inv)%q, (z*inv)%q))
                elif y != 0:
                    inv = pow(y, q-2, q)
                    points.append((0, 1, (z*inv)%q))
                else:
                    points.append((0, 0, 1))
    
    n_pts = len(points)
    # n_pts should be 43
    
    # Define polarity
    # Use the form x1*y2 + x2*y3 + x3*y1 = 0
    # This is a symmetric bilinear form?
    # f(u, v) = u[0]*v[1] + u[1]*v[2] + u[2]*v[0]
    # In char 2, f(u, u) = u[0]*u[1] + u[1]*u[2] + u[2]*u[0]
    # This is not necessarily 0.
    # If f(u, u) != 0, then u is not on its polar line.
    # If f(u, u) == 0, then u is on its polar line.
    
    # Check if this is a polarity.
    # A polarity requires f(u, v) = f(v, u).
    # f(u, v) = u0 v1 + u1 v2 + u2 v0
    # f(v, u) = v0 u1 + v1 u2 + v2 u0
    # These are not equal in general.
    # We need a symmetric form.
    # f(u, v) = u0 v1 + u1 v0 + u1 v2 + u2 v1 + u2 v0 + u0 v2
    # This is the standard dot product in char 2?
    # f(u, v) = sum ui vi.
    # f(u, v) = f(v, u).
    # So use dot product.
    
    def dot(u, v):
        return (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q

    # Construct graph
    edges = set()
    for i in range(n_pts):
        for j in range(i+1, n_pts):
            if dot(points[i], points[j]) == 0:
                edges.add((i, j))
    
    # Remove 4 vertices to get n=39
    # Remove indices 0, 1, 2, 3
    removed = {0, 1, 2, 3}
    new_indices = []
    for i in range(n_pts):
        if i not in removed:
            new_indices.append(i)
            
    n_new = len(new_indices)
    # n_new should be 39
    
    # Map old indices to new indices
    old_to_new = {old: new for new, old in enumerate(new_indices)}
    
    new_edges = []
    for u, v in edges:
        if u in old_to_new and v in old_to_new:
            nu = old_to_new[u]
            nv = old_to_new[v]
            new_edges.append([nu, nv])
            
    return {'n': n_new, 'edges': new_edges}
