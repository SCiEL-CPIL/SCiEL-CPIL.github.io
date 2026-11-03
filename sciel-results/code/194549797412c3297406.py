import numpy as np

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs 200 equal circles in a unit square using a hexagonal close packing strategy.
    We binary search for the largest spacing 's' such that the N points closest to the origin
    in a hexagonal lattice with spacing 's' fit inside the square [-0.5, 0.5]^2.
    """
    
    def generate_hex_grid(s, max_radius):
        """
        Generate points on a hexagonal grid with spacing s, within max_radius of origin.
        """
        points = []
        # Hex grid rows
        # y = k * s * sqrt(3) / 2
        # x = (j + (k % 2) * 0.5) * s
        
        max_k = int(np.ceil(max_radius / (s * np.sqrt(3) / 2))) + 2
        max_j = int(np.ceil(max_radius / s)) + 2
        
        for k in range(-max_k, max_k + 1):
            y = k * s * np.sqrt(3) / 2
            offset = (k % 2) * 0.5
            for j in range(-max_j, max_j + 1):
                x = (j + offset) * s
                # Only keep points within max_radius to keep the list small
                if x*x + y*y <= max_radius*max_radius:
                    points.append([x, y])
        return np.array(points)

    def get_cluster(s, N):
        """
        Get the N points closest to the origin from a hexagonal grid with spacing s.
        """
        # Estimate max radius needed. For N points, area ~ N * s^2 * sqrt(3)/2.
        # Radius ~ sqrt(N * s^2 * sqrt(3)/2 / pi)
        # Add a buffer.
        max_r = np.sqrt(N) * s * 1.5
        pts = generate_hex_grid(s, max_r)
        
        if len(pts) < N:
            # Increase radius and retry
            max_r *= 1.5
            pts = generate_hex_grid(s, max_r)
            
        if len(pts) < N:
            return None
            
        # Calculate distances to origin
        dists = np.linalg.norm(pts, axis=1)
        # Get indices of N smallest distances
        idx = np.argpartition(dists, N)[:N]
        return pts[idx]

    def is_valid(points):
        """
        Check if points are inside the square and non-overlapping.
        Returns the radius if valid, else None.
        """
        if points is None or len(points) != N:
            return None
        
        # Check bounds
        if np.any(points[:, 0] > 0.5) or np.any(points[:, 0] < -0.5):
            return None
        if np.any(points[:, 1] > 0.5) or np.any(points[:, 1] < -0.5):
            return None
        
        # Check min distance between points
        # Using scipy for efficiency if available, else brute force (N=200 is small)
        # Brute force O(N^2) = 40,000 ops, very fast.
        min_dist = float('inf')
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    min_dist = d
        
        # Check min distance to boundary
        min_boundary_dist = float('inf')
        for p in points:
            # Distance to x=0.5, x=-0.5, y=0.5, y=-0.5
            d1 = 0.5 - p[0]
            d2 = 0.5 + p[0]
            d3 = 0.5 - p[1]
            d4 = 0.5 + p[1]
            min_b = min(d1, d2, d3, d4)
            if min_b < min_boundary_dist:
                min_boundary_dist = min_b
        
        radius = min(min_dist / 2, min_boundary_dist)
        
        # Ensure circles are strictly inside (or on boundary) and non-overlapping
        # The task says "on or outside ... makes invalid". So strictly inside or on boundary is ok?
        # "inside the square [-1/2, 1/2]^2". Usually means inclusive.
        # "non-overlapping" means dist >= 2r.
        # My check ensures dist >= 2r (since r = min_dist/2) and boundary dist >= r.
        return radius

    # Binary search for the largest spacing s
    # Lower bound: square grid spacing
    s_low = 1.0 / np.sqrt(N)
    
    # Upper bound: theoretical hex spacing + buffer
    # r_hex ~ 1/sqrt(2*N*sqrt(3)) -> s_hex ~ 2/sqrt(2*N*sqrt(3))
    s_theoretical = 2.0 / np.sqrt(2.0 * N * np.sqrt(3))
    s_high = s_theoretical * 1.2
    
    best_points = None
    best_radius = 0.0
    
    # Perform binary search
    for _ in range(30): # 30 iterations is enough for double precision
        s_mid = (s_low + s_high) / 2.0
        pts = get_cluster(s_mid, N)
        
        if pts is not None:
            # Center the cluster at the origin to maximize symmetry and fit
            # The cluster generated is centered at 0 by construction (points closest to 0)
            # But the "center of mass" might be slightly off. Let's center it.
            centroid = np.mean(pts, axis=0)
            pts_centered = pts - centroid
            
            # Check validity of centered points
            rad = is_valid(pts_centered)
            
            if rad is not None:
                best_radius = rad
                best_points = pts_centered
                s_low = s_mid # Try larger spacing
            else:
                s_high = s_mid # Spacing too large, points stick out
        else:
            s_high = s_mid

    if best_points is None:
        # Fallback to square grid if something went wrong
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_points

# EVOLVE-BLOCK-END
