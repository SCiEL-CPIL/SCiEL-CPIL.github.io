import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # 1. Generate points of PG(2, 5)
    # Points are 1D subspaces of F_5^3
    # Represented as (x, y, z) where x,y,z in {0..4} and not all 0
    # Normalized: first non-zero component is 1
    points = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize
                if x != 0:
                    # x is already 1..4, scale to make it 1
                    inv_x = pow(x, -1, 5)
                    px = 1
                    py = (y * inv_x) % 5
                    pz = (z * inv_x) % 5
                elif y != 0:
                    inv_y = pow(y, -1, 5)
                    px = 0
                    py = 1
                    pz = (z * inv_y) % 5
                else:
                    # z != 0
                    inv_z = pow(z, -1, 5)
                    px = 0
                    py = 0
                    pz = 1
                
                point = (px, py, pz)
                if point not in points:
                    points.append(point)
    
    # Should have 31 points
    n = len(points)
    if n != 31:
        # Fallback to random or simple graph if construction fails
        # But it shouldn't fail
        pass
        
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # 2. Construct polarity graph edges
    # Edge (u, v) if u dot v == 0 and u != v
    edges = []
    adjacency = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i + 1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % 5
            if dot == 0:
                edges.append([i, j])
                adjacency[i][j] = True
                adjacency[j][i] = True
                
    # Verify edge count
    # Expected 90
    # If not 90, we might need to adjust, but theory says 90.
    
    # 3. Find a non-edge to add
    # We need an edge (u, v) such that u != v and not adjacency[u][v]
    # Any non-edge will work as it will add exactly 1 C4.
    added_edge = None
    for i in range(n):
        for j in range(i + 1, n):
            if not adjacency[i][j]:
                added_edge = [i, j]
                break
        if added_edge:
            break
            
    if added_edge is None:
        # Graph is complete? Impossible for n=31, m=90.
        # Fallback
        added_edge = [0, 1]
        
    edges.append(added_edge)
    
    # Return exactly 91 edges
    return edges[:91]
