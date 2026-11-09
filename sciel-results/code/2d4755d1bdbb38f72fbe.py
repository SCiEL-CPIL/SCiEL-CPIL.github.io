import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time

def construct():
    start_time = time.time()
    
    d = 7
    n = 1 << d
    total_edges = d * (n // 2)
    
    # 1. Enumerate edges
    edges = []
    edge_index = {}
    for u in range(n):
        for k in range(d):
            v = u ^ (1 << k)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_index[(u, v)] = idx
                
    # 2. Enumerate faces
    face_edge_indices = []
    for i in range(d):
        for j in range(i + 1, d):
            # Iterate over vertices where bit i is 0 and bit j is 0
            # This uniquely identifies the face for dimensions i, j
            # There are 2^(d-2) such vertices
            for u in range(n):
                if (u & (1 << i)) == 0 and (u & (1 << j)) == 0:
                    v1 = u ^ (1 << i)
                    v2 = u ^ (1 << j)
                    v3 = u ^ (1 << i) ^ (1 << j)
                    
                    # Edges of the face
                    e1 = (min(u, v1), max(u, v1))
                    e2 = (min(v1, v3), max(v1, v3))
                    e3 = (min(v3, v2), max(v3, v2))
                    e4 = (min(v2, u), max(v2, u))
                    
                    # Get indices
                    try:
                        idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
                        face_edge_indices.append(idxs)
                    except KeyError:
                        # Should not happen
                        pass
                        
    # 3. Setup ILP
    n_edges = len(edges)
    n_faces = len(face_edge_indices)
    
    # A matrix: n_faces x n_edges
    A = np.zeros((n_faces, n_edges), dtype=int)
    for f_idx, edge_idxs in enumerate(face_edge_indices):
        for e_idx in edge_idxs:
            A[f_idx, e_idx] = 1
            
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(n_edges)
    
    # Constraints: sum(x_e for e in face) <= 3
    # 0 <= sum <= 3
    constraints = LinearConstraint(A, 0, 3)
    
    # Bounds: 0 <= x <= 1
    bounds = Bounds(0, 1)
    
    # Integrality: all binary
    integrality = np.ones(n_edges)
    
    # Solve
    try:
        res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x = res.x
            selected_edges = [list(edges[i]) for i in range(n_edges) if x[i] > 0.5]
            return selected_edges
        else:
            # Fallback if ILP fails
            pass
    except Exception:
        pass
        
    # Fallback: Greedy
    # Re-use the logic from the seed program or a simple random greedy
    chosen = set()
    out = []
    for u in range(n):
        for k in range(d):
            v = u ^ (1 << k)
            if u < v:
                # Check if adding (u, v) creates a C4
                # A C4 is formed if there is another edge (w, z) parallel to (u, v)
                # such that the other two edges exist.
                # This check is O(1) per edge if we maintain adjacency or just check the specific face.
                # Simpler: just check the 2 faces containing this edge? No, a C4 can be in any face.
                # Actually, an edge is in exactly 2 faces? No, in Q7, an edge is in 2 faces?
                # Edge is defined by dim k. It is in faces (k, j) for all j != k.
                # So 6 faces.
                # If we add edge e, we must ensure it doesn't complete a C4 in any of the 6 faces.
                # A C4 in face (k, j) is completed if the other 3 edges of that face are present.
                ok = True
                for j in range(d):
                    if j == k: continue
                    # Determine the 4 vertices of the face (k, j) containing u, v
                    # u and v differ in bit k.
                    # Let w = u ^ (1 << j)
                    # Let z = v ^ (1 << j)
                    w = u ^ (1 << j)
                    z = v ^ (1 << j)
                    
                    # The 4 edges are:
                    # (u, v) - current
                    # (w, z) - parallel
                    # (u, w) - side
                    # (v, z) - side
                    
                    if all(p in chosen for p in [
                        (min(w, z), max(w, z)),
                        (min(u, w), max(u, w)),
                        (min(v, z), max(v, z))
                    ]):
                        ok = False
                        break
                if ok:
                    chosen.add((u, v))
                    out.append([u, v])
                    
    return out

