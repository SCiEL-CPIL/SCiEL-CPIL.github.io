import math

def construct():
    N = 11
    
    # Calculate smallest K such that 3*K*K - 3*K + 1 >= N
    # K = ceil((3 + sqrt(12*N - 3)) / 6)
    K = math.ceil((3 + math.sqrt(12 * N - 3)) / 6.0)
    
    # Outer side length
    # Aligning vertices of the inner cluster with vertices of the outer hexagon
    # is more efficient than aligning edges.
    # Max distance from center to vertex of inner cluster: (K-1)*sqrt(3) + 1
    side = (K - 1) * math.sqrt(3) + 1.0
    
    # Generate hex coordinates in a hex cluster of side K
    # Axial coordinates (q, r)
    # Condition for being in cluster of side K: max(abs(q), abs(r), abs(q+r)) < K
    # Wait, if K=1, max < 1 means only (0,0). Correct.
    # If K=2, max < 2 means max is 0 or 1.
    # (0,0) -> 0.
    # (1,0) -> 1.
    # (1,-1) -> 1.
    # (0,1) -> 1.
    # (-1,1) -> 1.
    # (-1,0) -> 1.
    # (0,-1) -> 1.
    # Total 7. Correct.
    
    points = []
    for q in range(-K, K):
        for r in range(-K, K):
            if abs(q + r) < K:
                # Convert axial to pixel coordinates (flat-top)
                # x = 1.5 * q
                # y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = math.sqrt(3) * (r + q / 2.0)
                
                # Distance from center
                dist = math.sqrt(x*x + y*y)
                points.append((dist, q, r, x, y))
                
    # Sort by distance from center to pack tightly
    points.sort(key=lambda p: p[0])
    
    # Take first N points
    inner = []
    for i in range(N):
        dist, q, r, x, y = points[i]
        # Angle 0.0
        inner.append([x, y, 0.0])
        
    return inner, (0.0, 0.0), side, 0.0
