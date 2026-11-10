# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    deadline = time.time() + 580
    best_score = 0.0
    best_result = None

    # Precompute constants
    sqrt3 = np.sqrt(3)
    
    def hex_vertices(cx, cy, s, a_deg):
        a = np.radians(a_deg)
        angles = a + np.radians(60 * np.arange(6))
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

    def check_separation(v1, v2, tol=1e-6):
        """Check if two unit hexagons (given as 6x2 vertex arrays) are separated."""
        # Edge normals for hexagon 1
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            # Normal pointing outward (perpendicular to edge)
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
                # Project all vertices of hex1 and hex2 onto n
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1 = proj1.min()
                max1 = proj1.max()
                min2 = proj2.min()
                max2 = proj2.max()
                # If separated on this normal, return True
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        # Edge normals for hexagon 2
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1 = proj1.min()
                max1 = proj1.max()
                min2 = proj2.min()
                max2 = proj2.max()
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True
        return False

    def check_in_hexagon(point, outer_v, tol=1e-6):
        """Check if point is inside or on the boundary of outer hexagon."""
        for i in range(6):
            p1 = outer_v[i]
            p2 = outer_v[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
                # The normal should point inward for this check to work consistently
                # Let's check cross product instead
                # cross(p2 - p1, point - p1) should have consistent sign
                cross = edge[0] * (point[1] - p1[1]) - edge[1] * (point[0] - p1[0])
                if cross < -tol:
                    return False
        return True

    def is_valid_packing(inner_data, outer_data, outer_side):
        """Validate the packing."""
        # Check all finite
        if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
            return False
        if outer_side < np.sqrt(11) - 1e-6:
            return False
        
        ox, oy, o_a = outer_data
        ov = hex_vertices(ox, oy, outer_side, o_a)
        
        inner_verts = []
        for i in range(11):
            ix, iy, i_a = inner_data[i]
            iv = hex_vertices(ix, iy, 1.0, i_a)
            inner_verts.append(iv)
            # Check edge lengths
            for j in range(6):
                p1 = iv[j]
                p2 = iv[(j + 1) % 6]
                edge_len = np.linalg.norm(p2 - p1)
                if abs(edge_len - 1.0) > 1e-8:
                    return False
            # Check all vertices inside outer hexagon
            for j in range(6):
                if not check_in_hexagon(iv[j], ov):
                    return False
        
        # Check pairwise separation
        for i in range(11):
            for j in range(i + 1, 11):
                if not check_separation(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def score_packing(inner_data, outer_data, outer_side):
        if not is_valid_packing(inner_data, outer_data, outer_side):
            return 0.0
        return 1.0 / outer_side

    def generate_candidate(seed_offset=0):
        """Generate a candidate packing using hexagonal grid logic with some jitter."""
        random.seed(42 + seed_offset)
        
        # Base positions in a compact hexagonal arrangement
        # Hex spacing: horizontal = 2.0, vertical = sqrt(3) ~ 1.732
        # For 11 hexagons, a common pattern is a hexagon of radius 2 (1+6+4=11? No, 1+6=7, 1+6+12=19)
        # 11 hexagons: central + 6 around + 4 more?
        # Let's try a cluster:
        # Row 0: 3 hexagons
        # Row 1: 5 hexagons
        # Row 2: 3 hexagons
        # Total: 11
        
        dx = 2.0
        dy = np.sqrt(3)
        
        positions = []
        # Row 1 (center)
        for i in range(-2, 3):
            positions.append([i * dx, 0])
        # Row 0 (top)
        for i in range(-1, 2):
            positions.append([i * dx + dx/2, dy])
        # Row 2 (bottom)
        for i in range(-1, 2):
            positions.append([i * dx + dx/2, -dy])
        
        # This gives 5 + 3 + 3 = 11 positions
        # Apply small random jitter
        for p in positions:
            p[0] += random.uniform(-0.1, 0.1)
            p[1] += random.uniform(-0.1, 0.1)
        
        inner_data = np.zeros((11, 3))
        for i, p in enumerate(positions):
            inner_data[i] = [p[0], p[1], 0]
        
        # Estimate outer hexagon size
        # Find bounding box of all vertices
        all_pts = []
        for i in range(11):
            v = hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
            all_pts.append(v)
        all_pts = np.vstack(all_pts)
        
        # Center of mass
        center = np.mean(all_pts, axis=0)
        
        # Find max distance from center to any vertex
        max_dist = np.max(np.linalg.norm(all_pts - center, axis=1))
        
        # For a regular hexagon with side s, the distance from center to vertex is s
        # So we need s >= max_dist
        # Add a small margin
        outer_side = max_dist + 0.01
        outer_data = np.array([center[0], center[1], 0])
        
        return inner_data, outer_data, outer_side

    def optimize_candidate(inner_data, outer_data, outer_side):
        """Try to optimize the candidate by adjusting outer hexagon."""
        # For a fixed set of inner hexagons, the optimal outer hexagon is the smallest one that contains them.
        # This is a complex problem, but we can try a few orientations.
        best_local_side = outer_side
        best_local_outer = outer_data
        
        # Calculate required side for each of 3 main orientations (0, 30, 60 degrees)
        # Actually, we can just try to shrink the side length until it becomes invalid.
        # But shrinking might make it invalid. Instead, let's try to find the minimal side.
        
        # A simpler heuristic: Calculate the max distance from the centroid to all vertices,
        # and use that as a lower bound. Then add a small buffer.
        
        # Let's try rotating the outer hexagon and centering it differently.
        for angle in range(0, 180, 30):
            for dx in [-0.5, 0, 0.5]:
                for dy in [-0.5, 0, 0.5]:
                    test_outer = np.array([outer_data[0] + dx, outer_data[1] + dy, angle])
                    
                    # Find minimal side length
                    # We can binary search for the minimal side length
                    lo = np.sqrt(11)
                    hi = outer_side + 5
                    for _ in range(20):
                        mid = (lo + hi) / 2
                        if is_valid_packing(inner_data, test_outer, mid):
                            hi = mid
                        else:
                            lo = mid
                    if hi < best_local_side:
                        best_local_side = hi
                        best_local_outer = test_outer
        
        return inner_data, best_local_outer, best_local_side

    # Main search loop
    num_attempts = 200
    for i in range(num_attempts):
        if time.time() > deadline:
            break
        
        inner_data, outer_data, outer_side = generate_candidate(i)
        
        # Try to optimize
        if time.time() > deadline - 10:
            break
            
        optimized_inner, optimized_outer, optimized_side = optimize_candidate(inner_data, outer_data, outer_side)
        
        s = score_packing(optimized_inner, optimized_outer, optimized_side)
        if s > best_score:
            best_score = s
            best_result = (optimized_inner, optimized_outer, optimized_side)
            
        # Also try a static, known-good configuration as a fallback if optimization fails
        if i == 0:
            # The seed program's configuration
            seed_inner = np.array([
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ])
            seed_outer = np.array([0, 0, 0])
            seed_side = 8.0
            s_seed = score_packing(seed_inner, seed_outer, seed_side)
            if s_seed > best_score:
                best_score = s_seed
                best_result = (seed_inner, seed_outer, seed_side)

    if best_result is None:
        # Absolute fallback
        return np.zeros((11, 3)), np.array([0, 0, 0]), 10.0

    return best_result

# EVOLVE-BLOCK-END
