import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 10.0  # This will be very fast, but guard just in case

    # 1. Define the 12 centers
    # We use the 12 most central hexagons from a side-3 hexagonal lattice.
    # Axial coordinates (q, r) for flat-top hexagons.
    # Condition for side-3 (radius 2): max(|q|, |r|, |q+r|) <= 2
    # We select the 12 with smallest "distance" from center.
    candidates = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                # Calculate cartesian coordinates
                # x = 1.5 * q
                # y = sqrt(3) * (r + q/2)
                x = 1.5 * q
                y = math.sqrt(3) * (r + q / 2.0)
                # Use q^2 + r^2 + (q+r)^2 as a proxy for centrality
                dist_metric = q*q + r*r + (q+r)*(q+r)
                candidates.append((dist_metric, x, y))
    
    # Sort by centrality and take the first 12
    candidates.sort(key=lambda x: x[0])
    centers = [(x, y) for _, x, y in candidates[:12]]
    
    # 2. Generate all vertices of the 12 unit hexagons
    # Unit hexagon vertices relative to center (flat-top, angle 0):
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    hex_verts = []
    for i in range(6):
        angle = math.pi / 3 * i
        hex_verts.append((math.cos(angle), math.sin(angle)))
        
    all_vertices = []
    for cx, cy in centers:
        for vx, vy in hex_verts:
            all_vertices.append((cx + vx, cy + vy))
            
    # 3. Minimize the outer hexagon side length L
    # The outer hexagon is regular, centered at (0,0), with rotation angle theta.
    # A point (x, y) is inside the outer hexagon of side L and rotation theta
    # if its projection onto the 6 normal directions is <= L * sqrt(3) / 2.
    # The 6 normal directions are at angles theta + k * 60 degrees.
    # Actually, the apothem A = L * sqrt(3) / 2.
    # So L = 2 * A / sqrt(3).
    # We need to find the minimum A such that max(dot(v, n_k)) <= A for all k.
    # A(theta) = max_{v in V, k in 0..5} dot(v, n_k(theta))
    # L(theta) = 2 * A(theta) / sqrt(3)
    
    def get_required_L(theta):
        max_proj = 0.0
        # Normals for the outer hexagon
        # The sides of a flat-top hexagon are horizontal? No, let's define normals.
        # If theta=0, let's say the first normal is at angle 0 (pointing right).
        # The hexagon is defined by |x| <= A, |x/2 + sqrt(3)y/2| <= A, etc.
        # Simpler: iterate 6 angles.
        for k in range(6):
            angle_n = theta + k * math.pi / 3
            nx = math.cos(angle_n)
            ny = math.sin(angle_n)
            for vx, vy in all_vertices:
                proj = vx * nx + vy * ny
                if proj > max_proj:
                    max_proj = proj
        # max_proj is the required apothem A
        L = 2.0 * max_proj / math.sqrt(3)
        return L

    # Optimize theta in [0, 60 degrees]
    # Use a fine grid search, it's very fast.
    best_L = float('inf')
    best_theta = 0.0
    num_steps = 1000
    for i in range(num_steps + 1):
        theta = (i / num_steps) * (math.pi / 3)
        L = get_required_L(theta)
        if L < best_L:
            best_L = L
            best_theta = theta
            
    # Refine around best_theta with a smaller grid
    if best_L < float('inf'):
        start_t = best_theta - 0.01
        end_t = best_theta + 0.01
        for i in range(1000):
            theta = start_t + (i / 1000.0) * (end_t - start_t)
            if theta < 0 or theta > math.pi/3:
                continue
            L = get_required_L(theta)
            if L < best_L:
                best_L = L
                best_theta = theta

    # 4. Construct the return value
    # inner: list of [x, y, angle_degrees]
    # We can keep all inner hexagons at angle 0 (aligned with grid)
    inner = []
    for cx, cy in centers:
        inner.append([cx, cy, 0.0])
        
    # outer: center (0,0), side length best_L, rotation best_theta
    # The problem asks for outer hexagon center, side length, and rotation.
    # Rotation is in degrees? The seed used 0.0. Let's assume degrees for consistency with inner angle.
    outer_angle_deg = math.degrees(best_theta)
    
    return inner, (0.0, 0.0), best_L, outer_angle_deg
# EVOLVE-BLOCK-END
