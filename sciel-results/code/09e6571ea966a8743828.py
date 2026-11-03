import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    
    # Lattice vectors
    eps = 0.01
    d = math.sqrt(3) + eps
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    centers = [np.array([0.0, 0.0])]
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    current_centers = np.array(centers)
    inner_angle = 30.0
    
    # Precompute support function r(n) for inner hexagons
    # Inner hexagon angle is 30 deg.
    # Normals of inner hexagon are at 30+60k.
    # Support function r(alpha) where alpha is angle of normal n.
    # r(alpha) = max(cos(alpha - (30+60k))) * 1? No.
    # The vertices are at 30+60k.
    # The distance from center to vertex is 1.
    # The distance from center to side midpoint is sqrt(3)/2.
    # r(alpha) is the projection of the hexagon onto direction alpha.
    # It is 1 * max(cos(alpha - theta_v)) for vertices theta_v?
    # No, the support function of a convex polygon is max(v . n).
    # For a regular hexagon with circumradius 1, r(alpha) = cos(delta) where delta is angle to nearest vertex?
    # Actually, r(alpha) = 1 / cos(min_delta_to_normal_of_side)?
    # Let's just compute it numerically for the specific directions we need.
    
    def get_support(alpha_rad, hex_angle_deg):
        # alpha_rad: angle of the direction vector n
        # hex_angle_deg: orientation of the hexagon
        hex_angle_rad = math.radians(hex_angle_deg)
        max_proj = 0.0
        for k in range(6):
            theta = hex_angle_rad + k * math.pi / 3
            proj = math.cos(alpha_rad - theta)
            if proj > max_proj:
                max_proj = proj
        return max_proj

    def compute_cost_fast(centers, hex_angle_deg):
        min_side = float('inf')
        # Sample outer angles
        for theta_deg in np.linspace(0, 60, 13):
            theta = math.radians(theta_deg)
            max_width = 0.0
            for k in range(3):
                normal_angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                
                # Project centers
                projs = centers[:, 0] * nx + centers[:, 1] * ny
                w_centers = np.max(projs) - np.min(projs)
                
                # Support of one hexagon
                r = get_support(normal_angle, hex_angle_deg)
                
                # Total width
                w_total = w_centers + 2 * r
                
                if w_total > max_width:
                    max_width = w_total
            side = max_width / math.sqrt(3)
            if side < min_side:
                min_side = side
        return min_side

    best_cost = compute_cost_fast(current_centers, inner_angle)
    best_centers = current_centers.copy()
    
    temp = 1.0
    step_size = 0.5
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        new_centers = current_centers + rng.normal(0, step_size, (N, 2))
        
        # Check disjointness
        valid = True
        min_dist = d
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(new_centers[i] - new_centers[j])
                if dist < min_dist:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            new_cost = compute_cost_fast(new_centers, inner_angle)
            if new_cost < best_cost:
                best_cost = new_cost
                best_centers = new_centers.copy()
            
            current_cost = compute_cost_fast(current_centers, inner_angle)
            delta = new_cost - current_cost
            
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_centers = new_centers
                temp = max(0.01, temp * 0.99)
                step_size = max(0.01, step_size * 0.99)

    # Final calculation for return values
    # Use best_centers
    # Find best theta
    best_theta_deg = 0
    best_min_side = float('inf')
    
    for theta_deg in np.linspace(0, 60, 101):
        theta = math.radians(theta_deg)
        max_width = 0.0
        for k in range(3):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            projs = best_centers[:, 0] * nx + best_centers[:, 1] * ny
            w_centers = np.max(projs) - np.min(projs)
            r = get_support(normal_angle, inner_angle)
            w_total = w_centers + 2 * r
            if w_total > max_width:
                max_width = w_total
        side = max_width / math.sqrt(3)
        if side < best_min_side:
            best_min_side = side
            best_theta_deg = theta_deg

    # Compute actual center and side to ensure validity
    # Use centroid of centers? Or vertices?
    # Centroid of centers is a good guess for the center of the cluster.
    cx = np.mean(best_centers[:, 0])
    cy = np.mean(best_centers[:, 1])
    
    # Verify and adjust side
    theta_rad = math.radians(best_theta_deg)
    max_proj = 0.0
    for k in range(6):
        normal_angle = theta_rad + math.pi/6 + k * math.pi/3
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # Project all vertices
        # To be precise, we should project vertices.
        # But for a quick check, we can use the support function logic.
        # Max projection of cluster = max_c (c.n) + r(n)
        # Min projection of cluster = min_c (c.n) - r(n)
        # We need c_outer . n to be in the middle.
        # Let's just compute the max deviation from the centroid.
        
        projs = best_centers[:, 0] * nx + best_centers[:, 1] * ny
        max_c = np.max(projs)
        min_c = np.min(projs)
        r = get_support(normal_angle, inner_angle)
        
        # The cluster spans [min_c - r, max_c + r]
        # The hexagon centered at (cx, cy) spans [cx.n - S*sqrt(3)/2, cx.n + S*sqrt(3)/2]
        # We need [min_c - r, max_c + r] to be inside the hexagon span.
        # So cx.n - S*sqrt(3)/2 <= min_c - r
        # And cx.n + S*sqrt(3)/2 >= max_c + r
        
        # This gives:
        # S*sqrt(3)/2 >= cx.n - (min_c - r)
        # S*sqrt(3)/2 >= (max_c + r) - cx.n
        
        # So S >= 2/sqrt(3) * max( cx.n - min_c + r, max_c + r - cx.n )
        
        dev1 = cx * nx + cy * ny - min_c + r
        dev2 = max_c + r - (cx * nx + cy * ny)
        dev = max(dev1, dev2)
        if dev > max_proj:
            max_proj = dev
            
    outer_side = 2.0 * max_proj / math.sqrt(3)
    outer_angle = best_theta_deg
    outer_center = (cx, cy)
    
    inner = [[best_centers[i][0], best_centers[i][1], inner_angle] for i in range(N)]
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
