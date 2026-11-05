import math

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Determine the required cluster radius K
    # Hexagonal numbers: H_k = 1 + 3k(k+1)
    # We need smallest K such that H_K >= N
    K = 0
    while 1 + 3 * K * (K + 1) < N:
        K += 1
    
    # The side length of the containing hexagon aligned with the grid
    side_length = K + 1
    
    # Generate coordinates for the hexagons
    # We use axial coordinates (q, r)
    # Convert to Cartesian (pointy-top)
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    inner = []
    
    # Add center
    inner.append([0.0, 0.0, 0.0])
    
    # Add rings
    count = 1
    while count < N:
        K += 1
        # Ring K has 6*K hexagons
        # Directions for the 6 corners of the hexagon in axial coords
        # (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)
        # We can generate the ring by walking along the edges
        
        # Start at (K, 0)
        q, r = K, 0
        for i in range(6):
            # Step direction
            # Edge 0: (1, -1) ? No, let's use standard cube coords or just step
            # Let's just generate all points in ring K and sort?
            # Or simpler:
            # The points in ring K are those with max(|q|, |r|, |q+r|) == K? No, that's for cube.
            # Axial: distance is (|q| + |r| + |q+r|) / 2
            # Let's just generate candidates for ring K
            
            # A simpler way to generate ring K:
            # Start at (K, 0)
            # Move (1, -1) K times? No.
            # Let's use the 6 corner vectors and interpolate.
            # Corners:
            # (K, 0)
            # (K, -K)
            # (0, -K)
            # (-K, 0)
            # (-K, K)
            # (0, K)
            
            corners = [
                (K, 0),
                (K, -K),
                (0, -K),
                (-K, 0),
                (-K, K),
                (0, K)
            ]
            
            # We are on edge i from corners[i] to corners[(i+1)%6]
            # We add K points on this edge (excluding the start, including end? No, we need 6K points total)
            # So we add K points for each of the 6 edges, excluding the starting corner of the edge to avoid duplication.
            
            for j in range(K):
                # Point is corners[i] + j * step
                # Step is (corners[(i+1)%6] - corners[i]) / K
                q1, r1 = corners[i]
                q2, r2 = corners[(i+1) % 6]
                dq = (q2 - q1) / K
                dr = (r2 - r1) / K
                
                # j goes from 1 to K-1? No, we need K points.
                # If we do j=1..K, we hit the next corner.
                # The next corner will be the start of the next edge.
                # So we should include j=1..K-1?
                # Let's check total. 6 edges * (K-1) internal points?
                # No, the corners are shared.
                # Total points in ring K is 6K.
                # If we iterate 6 edges, and for each edge we take K points, we get 6K points.
                # We must be careful not to duplicate corners.
                # Let's take j from 0 to K-1 for each edge.
                # Point = corners[i] + j * step.
                # For j=0, we get corners[i].
                # For j=K-1, we get corners[i] + (K-1)/K * (next - current).
                # This misses the next corner?
                # The next corner is corners[i+1].
                # If we do this for all 6 edges, we get 6 corners (at j=0 of each edge) and 6*(K-1) internal points.
                # Total 6K. Correct.
                
                q = q1 + j * dq
                r = r1 + j * dr
                
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                
                inner.append([x, y, 0.0])
                count += 1
                if count >= N:
                    break
            if count >= N:
                break
                
    return inner, (0.0, 0.0), float(side_length), 0.0
# EVOLVE-BLOCK-END
