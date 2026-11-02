import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Define the cluster of 12 unit hexagons.
    # We use a 3x4 parallelogram arrangement in the honeycomb lattice.
    # This is a compact, valid packing.
    centers = []
    dx = np.sqrt(3)
    dy = 1.5
    for j in range(3):
        for i in range(4):
            x = (i + j / 2.0) * dx
            y = j * dy
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the cluster at the origin
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid

    # Vertices of a single unit regular hexagon (side length 1)
    # Centered at origin, vertex at angle 0
    vertices = []
    for k in range(6):
        angle = np.radians(60 * k)
        vertices.append([np.cos(angle), np.sin(angle)])
    vertices = np.array(vertices)

    def calculate_min_side(alpha_deg):
        """
        Calculates the minimum side length of the outer hexagon
        required to contain the rotated cluster.
        The outer hexagon is fixed in orientation (angle 0).
        """
        alpha = np.radians(alpha_deg)
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate centers and vertices
        rot_centers = centers @ R.T
        rot_vertices = vertices @ R.T
        
        max_A = 0.0
        
        # For each inner hexagon center, check all its vertices
        for c_pt in rot_centers:
            # All 6 vertices of this inner hexagon
            pts = c_pt + rot_vertices
            
            # Calculate required apothem A for each point to be inside
            # Outer hexagon (flat-top relative to y-axis? No, pointy-top?
            # Our inequalities:
            # |y| <= A
            # |sqrt(3)*y - x| <= 2A
            # |sqrt(3)*y + x| <= 2A
            # This corresponds to a hexagon with horizontal top/bottom edges?
            # Let's verify.
            # If x=0, y=A, it's on boundary.
            # If x=sqrt(3)*A, y=A, then sqrt(3)*A - sqrt(3)*A = 0 <= 2A.
            # The vertices of this outer hexagon are at (0, A*2/sqrt(3)? No.
            # Apothem A. Radius R = 2A/sqrt(3).
            # Vertices at angles 30, 90, 150...
            # At 90 deg: (0, R). y=R.
            # Inequality |y| <= A. If y=R=2A/sqrt(3) > A, it fails.
            # So my inequalities describe a hexagon with vertices at 0, 60...
            # Let's re-derive.
            
            # Standard Hexagon with "flat" sides on top/bottom (vertices at 30, 90, 150...):
            # Equations:
            # |y| <= A
            # |x| <= sqrt(3)*A
            # |x + y/sqrt(3)| <= A ? No.
            
            # Let's use the general support function logic or standard inequalities for a hexagon
            # with vertices at angles 0, 60, 120, 180, 240, 300.
            # This is a "pointy-top" hexagon.
            # Apothem A is distance to side.
            # Side 1 (top right): Normal at 30 deg.
            # Distance from origin to line is A.
            # Line: x*cos(30) + y*sin(30) = A
            # sqrt(3)/2 * x + 1/2 * y = A  => sqrt(3)*x + y = 2A
            # Similarly for other sides.
            
            # Inequalities for Pointy-Top Hexagon (vertices at 0, 60, ...):
            # 1. sqrt(3)*x + y <= 2A
            # 2. -sqrt(3)*x + y <= 2A  (Normal at 150? No, 90+60=150? No.)
            
            # Let's list normals:
            # Side between vertex 0 and 60: Normal at 30.
            # Side between 60 and 120: Normal at 90.
            # Side between 120 and 180: Normal at 150.
            # Side between 180 and 240: Normal at 210.
            # Side between 240 and 300: Normal at 270.
            # Side between 300 and 360: Normal at 330.
            
            # Conditions:
            # 1. x*cos(30) + y*sin(30) <= A  =>  sqrt(3)/2 x + 1/2 y <= A
            # 2. x*cos(90) + y*sin(90) <= A  =>  y <= A
            # 3. x*cos(150) + y*sin(150) <= A => -sqrt(3)/2 x + 1/2 y <= A
            # 4. x*cos(210) + y*sin(210) <= A => -sqrt(3)/2 x - 1/2 y <= A
            # 5. x*cos(270) + y*sin(270) <= A => -y <= A
            # 6. x*cos(330) + y*sin(330) <= A => sqrt(3)/2 x - 1/2 y <= A
            
            # So A >= max(
            #   sqrt(3)/2 x + 1/2 y,
            #   y,
            #   -sqrt(3)/2 x + 1/2 y,
            #   -sqrt(3)/2 x - 1/2 y,
            #   -y,
            #   sqrt(3)/2 x - 1/2 y
            # )
            # This can be simplified to:
            # A >= |y|
            # A >= sqrt(3)/2 |x| + 1/2 |y|  (from combining 1&6 and 3&4? No)
            
            # Actually:
            # max( sqrt(3)/2 x + 1/2 y, -sqrt(3)/2 x - 1/2 y ) = |sqrt(3)/2 x + 1/2 y|
            # max( -sqrt(3)/2 x + 1/2 y, sqrt(3)/2 x - 1/2 y ) = |-sqrt(3)/2 x + 1/2 y|
            
            # So A_req = max( |y|, |sqrt(3)/2 x + 1/2 y|, |-sqrt(3)/2 x + 1/2 y| )
            
            xs = pts[:, 0]
            ys = pts[:, 1]
            
            term1 = np.abs(ys)
            term2 = np.abs(np.sqrt(3)/2.0 * xs + 0.5 * ys)
            term3 = np.abs(-np.sqrt(3)/2.0 * xs + 0.5 * ys)
            
            A_pts = np.maximum(np.maximum(term1, term2), term3)
            local_max = np.max(A_pts)
            
            if local_max > max_A:
                max_A = local_max
        
        # Side length S is related to Apothem A by A = S * sqrt(3) / 2
        # So S = 2 * A / sqrt(3)
        S = 2.0 * max_A / np.sqrt(3)
        return S

    # Optimization
    # Grid search
    best_alpha = 0.0
    best_S = float('inf')
    
    # Coarse grid
    alphas = np.linspace(0, 60, 601)
    for a in alphas:
        S = calculate_min_side(a)
        if S < best_S:
            best_S = S
            best_alpha = a
            
    # Fine grid around best
    alphas_fine = np.linspace(best_alpha - 0.5, best_alpha + 0.5, 101)
    for a in alphas_fine:
        S = calculate_min_side(a)
        if S < best_S:
            best_S = S
            best_alpha = a

    # Construct return values
    alpha_rad = np.radians(best_alpha)
    c, s = np.cos(alpha_rad), np.sin(alpha_rad)
    R = np.array([[c, -s], [s, c]])
    final_centers = centers @ R.T
    
    inner = []
    for c_pt in final_centers:
        inner.append([float(c_pt[0]), float(c_pt[1]), float(best_alpha)])
        
    return inner, (0.0, 0.0), float(best_S), 0.0
# EVOLVE-BLOCK-END
