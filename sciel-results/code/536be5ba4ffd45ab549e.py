"""Pack 12 unit hexagons in a minimal regular hexagon."""
import numpy as np
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Get vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def min_enclosing_hexagon(points, outer_angle_deg=0.0):
    """Find minimal circumradius R of regular hexagon (with given angle) containing all points.
    Returns (R, center_x, center_y)."""
    pts = np.array(points)
    angle_rad = np.radians(outer_angle_deg)
    
    # Outward normals for the 6 faces
    # Face normals are at angles outer_angle + 30° + k*60°
    normals = []
    for k in range(6):
        theta = angle_rad + np.pi/6 + k * np.pi / 3
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    normals = np.array(normals)  # shape (6, 2)
    
    # We want to minimize R such that for all points p and all normals n:
    # (p - c) · n <= R * sqrt(3)/2
    # i.e., p · n - c · n <= R * sqrt(3)/2
    # R >= (p · n - c · n) * 2 / sqrt(3) for all p, n
    # R >= max_{p,n} (p · n - c · n) * 2 / sqrt(3)
    
    # To minimize R, we need to choose c to minimize max_{p,n} (p · n - c · n)
    # = max_{p,n} (p - c) · n
    
    # This is equivalent to finding the smallest regular hexagon containing the points.
    # For a fixed angle, we can solve via LP, but let's use a simpler approach:
    # The optimal center is related to the Chebyshev center in a transformed space.
    
    # Simpler: try center at centroid, then try to improve
    c = np.mean(pts, axis=0)
    
    def compute_R(c):
        # (p - c) · n for all p, n
        # Shape: (n_points, 6)
        diffs = pts - c  # (n_points, 2)
        projs = diffs @ normals.T  # (n_points, 6)
        max_proj = np.max(projs)
        return max_proj * 2.0 / np.sqrt(3.0)
    
    R = compute_R(c)
    
    # Try to improve by moving center
    # The gradient of R with respect to c is in the direction of the normal that achieves the max
    # Let's do a few iterations of gradient descent
    for _ in range(50):
        diffs = pts - c
        projs = diffs @ normals.T
        max_idx = np.unravel_index(np.argmax(projs), projs.shape)
        n_max = normals[max_idx[1]]
        # R = (p_max - c) · n_max * 2 / sqrt(3)
        # dR/dc = -n_max * 2 / sqrt(3)
        # So we should move c in the direction of n_max to decrease R
        # But we need to be careful not to increase other projections
        # Simple step
        step = 0.1
        c_new = c + step * n_max
        R_new = compute_R(c_new)
        if R_new < R:
            c = c_new
            R = R_new
        else:
            # Try smaller steps
            found = False
            for s in [0.05, 0.01, 0.001]:
                c_new = c + s * n_max
                R_new = compute_R(c_new)
                if R_new < R:
                    c = c_new
                    R = R_new
                    found = True
                    break
            if not found:
                break
    
    return R, c[0], c[1]

def compute_enclosing_side(hex_data, outer_angle_deg=0.0):
    """Compute minimal enclosing hexagon side for given hexagons."""
    all_points = []
    for (cx, cy, angle) in hex_data:
        verts = hex_vertices(cx, cy, 1.0, angle)
        all_points.extend(verts.tolist())
    
    R, cx, cy = min_enclosing_hexagon(all_points, outer_angle_deg)
    return R, (cx, cy)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_side = float('inf')
    best_result = None
    
    spacing = np.sqrt(3)  # Center-to-center distance for touching hexagons
    
    # Generate candidate configurations
    arrangements = []
    
    # Config 1: Hexagonal cluster (1 + 6 + 5)
    config1 = [(0, 0, 0)]
    for i in range(6):
        theta = i * np.pi / 3
        config1.append((spacing * np.cos(theta), spacing * np.sin(theta), 0))
    
    # Ring 2 positions
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = spacing * (q + r / 2.0)
                y = spacing * (r * np.sqrt(3) / 2.0)
                ring2.append((x, y))
    
    # Sort by angle
    ring2_sorted = sorted(ring2, key=lambda p: np.arctan2(p[1], p[0]))
    for i in range(5):
        config1.append((ring2_sorted[i][0], ring2_sorted[i][1], 0))
    arrangements.append(config1)
    
    # Config 2: Two rows of 6
    config2 = []
    for i in range(6):
        config2.append((i * spacing, 0, 0))
        config2.append((i * spacing + spacing/2, spacing * np.sqrt(3)/2, 0))
    arrangements.append(config2)
    
    # Config 3: 3x4 staggered
    config3 = []
    for row in range(3):
        for col in range(4):
            x = col * spacing + (row % 2) * spacing / 2
            y = row * spacing * np.sqrt(3) / 2
            config3.append((x, y, 0))
    arrangements.append(config3)
    
    # Config 4: More compact - use all ring 2 positions that are "close"
    config4 = [(0, 0, 0)]
    for i in range(6):
        theta = i * np.pi / 3
        config4.append((spacing * np.cos(theta), spacing * np.sin(theta), 0))
    # Pick 5 ring 2 positions that minimize spread
    # Try different subsets
    from itertools import combinations
    for combo in list(combinations(range(len(ring2_sorted)), 5)):
        if time.time() > deadline - 30:
            break
        config = [(0, 0, 0)]
        for i in range(6):
            theta = i * np.pi / 3
            config.append((spacing * np.cos(theta), spacing * np.sin(theta), 0))
        for idx in combo:
            config.append((ring2_sorted[idx][0], ring2_sorted[idx][1], 0))
        arrangements.append(config)
    
    # Evaluate all arrangements
    for arr in arrangements:
        if time.time() > deadline:
            break
        if len(arr) != N:
            continue
        
        for outer_angle in [0, 30]:
            side, center = compute_enclosing_side(arr, outer_angle)
            if side < best_side:
                best_side = side
                best_result = (arr, center, side, outer_angle)
    
    # Local search on best result
    if best_result is not None and time.time() < deadline - 10:
        best_arr, best_center, best_side_val, best_angle = best_result
        best_arr = [list(a) for a in best_arr]
        current_side = best_side_val
        current_center = best_center
        current_angle = best_angle
        
        # Local search
        for step in [0.2, 0.1, 0.05, 0.01]:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(N):
                    if time.time() > deadline - 5:
                        break
                    orig = best_arr[i][:]
                    for dx in [-step, step]:
                        for dy in [-step, step]:
                            if time.time() > deadline - 5:
                                break
                            best_arr[i][0] = orig[0] + dx
                            best_arr[i][1] = orig[1] + dy
                            
                            centers_angles = [(a[0], a[1], a[2]) for a in best_arr]
                            side, center = compute_enclosing_side(centers_angles, current_angle)
                            
                            if side < current_side - 1e-10:
                                current_side = side
                                current_center = center
                                improved = True
                            else:
                                best_arr[i] = orig[:]
        
        if current_side < best_side:
            best_side = current_side
            best_result = (best_arr, current_center, current_side, current_angle)
    
    if best_result is None:
        inner = [[spacing * k, 0.0, 0.0] for k in range(-N//2, N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    arr, center, side, angle = best_result
    inner = [[a[0], a[1], a[2]] for a in arr]
    return inner, center, side, angle
