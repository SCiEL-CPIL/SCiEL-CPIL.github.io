# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        ang = a + np.radians(60 * i)
        verts.append((cx + side * np.cos(ang), cy + side * np.sin(ang)))
    return np.array(verts)


def _separating_axis_ok(A, B, eps=1e-8):
    """Check if convex polygons A and B are separated using SAT.
    Returns True if separated (non-overlapping)."""
    for poly in (A, B):
        for i in range(6):
            p1 = poly[i]
            p2 = poly[(i + 1) % 6]
            edge = p2 - p1
            # Normal perpendicular to edge
            n = np.array([-edge[1], edge[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            # Project both polygons onto normal
            projA = A @ n
            projB = B @ n
            minA, maxA = np.min(projA), np.max(projA)
            minB, maxB = np.min(projB), np.max(projB)
            if maxA < minB - eps or maxB < minA - eps:
                return True  # Separated
    return False  # Not separated (overlapping or touching within eps)


def _is_inside_outer(verts_inner, outer_verts, eps=1e-6):
    """Check if all vertices of inner are inside the outer hexagon.
    outer_verts is in order around the hexagon."""
    for v in verts_inner:
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            edge = p2 - p1
            # cross product: (p2-p1) x (v-p1)
            cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
            if cross < -eps:
                return False
    return True


def _validate(inner_data, outer_data, s):
    """Validate the packing. Returns True if valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-10:
        return False
    if len(inner_data) != 12:
        return False
    
    # Check outer vertices are distinct
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons don't overlap
    inner_verts_list = []
    for i in range(12):
        verts = _hex_vertices(inner_data[i, 0], inner_data[i, 1], 1.0, inner_data[i, 2])
        inner_verts_list.append(verts)
    
    for i in range(12):
        for j in range(i + 1, 12):
            if not _separating_axis_ok(inner_verts_list[i], inner_verts_list[j], 1e-8):
                return False
    
    # Check all inner vertices are inside outer hexagon
    # Shift all by minus outer center
    ocx, ocy = outer_data[0], outer_data[1]
    outer_verts_shifted = outer_verts - np.array([ocx, ocy])
    
    for i in range(12):
        iv = inner_verts_list[i] - np.array([ocx, ocy])
        for v in iv:
            for k in range(6):
                p1 = outer_verts_shifted[k]
                p2 = outer_verts_shifted[(k + 1) % 6]
                edge = p2 - p1
                cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def hexagon_packing_12():
    deadline = time.time() + 600  # Leave 30s safety margin
    sqrt3 = np.sqrt(3)
    
    # Strategy: Use hexagonal grid arrangements with flat-top hexagons (angle=0)
    # Then optimize the outer hexagon orientation and position
    
    def axial_to_pixel_flat(q, r, size=1.0):
        """Convert axial coordinates to pixel coordinates for flat-top hexagons."""
        x = size * 1.5 * q
        y = size * (sqrt3 / 2.0 * q + sqrt3 * r)
        return x, y
    
    def generate_hex_grid_positions(radius):
        """Generate all positions in a hexagonal grid of given radius."""
        positions = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if abs(q + r) <= radius:
                    x, y = axial_to_pixel_flat(q, r)
                    positions.append((x, y))
        return positions
    
    def compute_min_outer_s(inner_x, inner_y, inner_angle, n_outer_steps=60):
        """Compute minimum outer hexagon side length for given inner arrangement.
        Returns (s, outer_x, outer_y, outer_angle_deg)"""
        # Compute all inner vertices
        all_verts = []
        for i in range(len(inner_x)):
            v = _hex_vertices(inner_x[i], inner_y[i], 1.0, inner_angle[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        
        # The optimal outer hexagon center is somewhere, and orientation varies
        # For a regular hexagon with side s, the apothem is s*sqrt(3)/2
        # For a given orientation, we need s*sqrt(3)/2 >= max projection of all verts 
        # in each of the 6 normal directions, relative to the center
        
        # The optimal center for a given orientation minimizes the max of (max_proj - min_proj)/2
        # in each normal direction, and s is determined by the worst direction
        
        best_s = np.inf
        best_ocx, best_ocy, best_oa = 0, 0, 0
        
        for oa_deg in np.arange(0, 30, 30.0 / n_outer_steps):
            oa = np.radians(oa_deg)
            # Normal angles for outer hexagon
            normals = np.array([[np.cos(oa + np.radians(30 + 60 * k)), 
                                  np.sin(oa + np.radians(30 + 60 * k))] for k in range(6)])
            
            # For each normal, compute min and max projection
            # The width in direction n is max(n·v) - min(n·v)
            # The center should be at the midpoint for each direction
            # s*sqrt(3)/2 >= max over all normals of (max_proj - min_proj)/2
            
            widths = []
            for n in normals:
                proj = all_verts @ n
                width = (np.max(proj) - np.min(proj)) / 2.0
                widths.append(width)
            widths = np.array(widths)
            
            s_candidate = 2.0 / sqrt3 * np.max(widths)
            
            if s_candidate < best_s:
                best_s = s_candidate
                best_oa = oa_deg
                # Center at the midpoint of the bounding box in each direction
                # Use the average of midpoints
                midpoints = []
                for n in normals:
                    proj = all_verts @ n
                    mid = (np.max(proj) + np.min(proj)) / 2.0
                    midpoints.append(mid)
                # The center in 2D: we need to find (cx, cy) such that for each normal,
                # n · (cx, cy) = midpoint
                # This is an overdetermined system; use least squares
                # normals @ [cx, cy]^T = midpoints
                try:
                    center = np.linalg.lstsq(normals, np.array(midpoints), rcond=None)[0]
                    best_ocx, best_ocy = center[0], center[1]
                except:
                    # Fallback: centroid of all vertices
                    best_ocx = np.mean(all_verts[:, 0])
                    best_ocy = np.mean(all_verts[:, 1])
        
        return best_s, best_ocx, best_ocy, best_oa
    
    # Generate candidate arrangements
    candidates = []
    
    # 1. Hexagonal cluster: center + 6 neighbors + 5 more from ring 2
    # Using flat-top hexagons
    # Ring 0: (0,0)
    # Ring 1: 6 positions
    # Ring 2: 12 positions
    
    ring0 = [(0, 0)]
    ring1 = []
    ring2 = []
    
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q + r)) == 1:
                x, y = axial_to_pixel_flat(q, r)
                ring1.append((x, y))
            elif max(abs(q), abs(r), abs(q + r)) == 2:
                x, y = axial_to_pixel_flat(q, r)
                ring2.append((x, y))
    
    # Candidate 1: 1 + 6 + 5 = 12 (take first 5 from ring2)
    cand1_pos = ring0 + ring1 + ring2[:5]
    candidates.append((cand1_pos, np.zeros(12), "hex_cluster_1"))
    
    # Candidate 2: 1 + 6 + 5 (different 5 from ring2 - the ones at "corners")
    # Ring 2 has 12 positions. Pick the 6 that are at distance 2*sqrt(3) from center
    # and 3 more
    # Actually, let's try: 1 + 6 + 6 = 13, too many. 
    # Try: 3 + 4 + 3 + 2 = 12 in a row-based arrangement
    candidates.append((cand1_pos, np.zeros(12), "hex_cluster_2"))
    
    # 2. Row-based arrangement
    # Pointy-top hexagons: angle = 90
    # For pointy-top: width = sqrt(3), height = 2
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Odd rows offset by sqrt(3)/2
    
    # Arrangement: 3-4-3-2 or 2-4-3-3 etc.
    # Let's try 3-4-3-2
    def make_row_arrangement(rows, angle_deg=90.0):
        """Create row-based arrangement."""
        pos = []
        y = 0
        for row_idx, count in enumerate(rows):
            x_offset = (sqrt3 / 2) * (row_idx % 2)  # Offset odd rows
            for i in range(count):
                x = i * sqrt3 - (count - 1) * sqrt3 / 2 + x_offset
                pos.append((x, y))
            y += 1.5
        return pos
    
    # Try several row arrangements
    row_configs = [
        [3, 4, 3, 2],
        [2, 4, 3, 3],
        [3, 3, 3, 3],
        [4, 4, 4],
        [2, 3, 4, 3],
        [3, 4, 4, 1],
    ]
    
    for rows in row_configs:
        if sum(rows) != 12:
            continue
        pos = make_row_arrangement(rows)
        angles = np.full(12, 90.0)  # Pointy-top
        candidates.append((pos, angles, f"rows_{rows}"))
    
    # Also try flat-top row arrangements
    # For flat-top: width = 2, height = sqrt(3)
    # Horizontal spacing: 2
    # Vertical spacing: sqrt(3)/2 * ... 
    # Actually for flat-top:
    # x-spacing = 2 (vertex to vertex)
    # But for packing, the "width" is 2 and "height" is sqrt(3)
    # Horizontal neighbors: distance 2 in x
    # Vertical neighbors: offset by 1 in x, sqrt(3)/2 in y
    
    def make_flat_top_rows(rows):
        """Flat-top row arrangement."""
        pos = []
        y = 0
        for row_idx, count in enumerate(rows):
            x_offset = 1.0 * (row_idx % 2)  # Offset odd rows by 1
            for i in range(count):
                x = i * 2.0 - (count - 1) * 2.0 / 2 + x_offset
                pos.append((x, y))
            y += sqrt3 / 2.0
        return pos
    
    for rows in row_configs:
        if sum(rows) != 12:
            continue
        pos = make_flat_top_rows(rows)
        angles = np.zeros(12)  # Flat-top
        candidates.append((pos, angles, f"flat_rows_{rows}"))
    
    # 3. Try various hexagonal cluster variations
    # Different subsets of ring 2
    for subset_idx in range(0, min(12, len(ring2)), 3):
        pos = ring0 + ring1 + ring2[subset_idx:subset_idx+5]
        if len(pos) == 12:
            angles = np.zeros(12)
            candidates.append((pos, angles, f"hex_subset_{subset_idx}"))
    
    # Evaluate all candidates
    best_result = None
    best_s = np.inf
    
    for pos, angles, name in candidates:
        if time.time() > deadline - 30:
            break
        inner_x = np.array([p[0] for p in pos])
        inner_y = np.array([p[1] for p in pos])
        inner_angle = angles.copy()
        
        s, ocx, ocy, oa = compute_min_outer_s(inner_x, inner_y, inner_angle, n_outer_steps=60)
        
        if s < best_s:
            best_s = s
            inner_data = np.column_stack([inner_x, inner_y, inner_angle])
            outer_data = np.array([ocx, ocy, oa])
            best_result = (inner_data, outer_data, s)
    
    # Now try local optimization on the best candidate
    if best_result is not None and time.time() < deadline - 60:
        inner_data, outer_data, s = best_result
        
        # Local search: perturb positions and see if we can improve
        def objective(params):
            px = params[:24:2]  # x coords
            py = params[1:24:2]  # y coords
            # Check separation
            for i in range(12):
                for j in range(i + 1, 12):
                    dx = px[i] - px[j]
                    dy = py[i] - py[j]
                    d = np.sqrt(dx * dx + dy * dy)
                    if d < sqrt3 - 1e-4:
                        return 1e10 + (sqrt3 - d) * 100
            # Compute min outer s
            s_local, _, _, _ = compute_min_outer_s(px, py, np.zeros(12), n_outer_steps=20)
            return s_local
        
        # Start from current best
        current_x = inner_data[:, 0].copy()
        current_y = inner_data[:, 1].copy()
        params = np.zeros(24)
        params[0:24:2] = current_x
        params[1:24:2] = current_y
        
        # Simple Nelder-Mead style local search
        best_params = params.copy()
        best_obj = objective(params)
        
        # Try random perturbations
        np.random.seed(42)
        step_sizes = [0.1, 0.05, 0.01]
        
        for step in step_sizes:
            iterations = 200
            for it in range(iterations):
                if time.time() > deadline - 60:
                    break
                # Random perturbation
                delta = np.random.normal(0, step, 24)
                new_params = best_params + delta
                new_obj = objective(new_params)
                if new_obj < best_obj:
                    best_params = new_params
                    best_obj = new_obj
        
        if best_obj < best_s:
            px = best_params[0:24:2]
            py = best_params[1:24:2]
            s_new, ocx, ocy, oa = compute_min_outer_s(px, py, inner_data[:, 2], n_outer_steps=60)
            if s_new < best_s:
                inner_data = np.column_stack([px, py, inner_data[:, 2]])
                outer_data = np.array([ocx, ocy, oa])
                best_s = s_new
    
    # Final validation
    if best_result is not None:
        inner_data, outer_data, s = best_result
        if not _validate(inner_data, outer_data, s):
            # Fall back to a known valid configuration
            # Simple grid: 4x3
            sqrt3 = np.sqrt(3)
            inner_x = []
            inner_y = []
            inner_a = []
            for row in range(3):
                for col in range(4):
                    inner_x.append(col * 2.0)
                    inner_y.append(row * sqrt3)
                    inner_a.append(0.0)
            inner_data = np.array(inner_x + inner_y + inner_a).reshape(12, 3)
            # Actually, let's use a proper flat-top grid
            inner_data = np.zeros((12, 3))
            for i in range(12):
                row = i // 4
                col = i % 4
                inner_data[i, 0] = col * 2.0
                inner_data[i, 1] = row * sqrt3
                inner_data[i, 2] = 0.0
            
            s, ocx, ocy, oa = compute_min_outer_s(inner_data[:, 0], inner_data[:, 1], 
                                                   inner_data[:, 2], n_outer_steps=60)
            outer_data = np.array([ocx, ocy, oa])
            best_s = s
    
    inner_data, outer_data, s = best_result
    return inner_data, outer_data, float(s)


# EVOLVE-BLOCK-END
