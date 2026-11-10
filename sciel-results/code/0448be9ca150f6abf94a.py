# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _hex_edge_normals(verts):
    """Compute outward unit normals for each edge of a hexagon given its 6 vertices."""
    norms = []
    for k in range(6):
        p1 = verts[k]
        p2 = verts[(k + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        norm = np.linalg.norm(n)
        if norm > 1e-15:
            n = n / norm
        else:
            n = np.array([1.0, 0.0])
        norms.append(n)
    return np.array(norms)


def _min_enclosing_hex_side(inner_hex_data, outer_angle_deg=0.0):
    """
    Compute minimum side of regular hexagon (centered at origin, given angle) containing all inner hexes.
    Returns side length (float).
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = outer_angle + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        projs = verts @ normals.T
        m = np.max(projs)
        if m > max_proj:
            max_proj = m
    
    return max_proj * 2.0 / np.sqrt(3.0)


def _check_validity(inner_hex_data, outer_hex_data, side):
    """Check all validity conditions. Returns True if valid."""
    if not (np.all(np.isfinite(inner_hex_data)) and np.all(np.isfinite(outer_hex_data))):
        return False
    if not np.isfinite(side):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    
    ox, oy, oa = outer_hex_data
    outer_verts = _hex_vertices(ox, oy, side, oa)
    
    # Check every inner vertex is inside outer hexagon
    for i in range(len(inner_hex_data)):
        x, y, a_deg = inner_hex_data[i]
        verts = _hex_vertices(x, y, 1.0, a_deg)
        for p in verts:
            for j in range(6):
                p1 = outer_verts[j]
                p2 = outer_verts[(j + 1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < -1e-6:
                    return False
    
    # Check every pair of inner hexagons is separated
    n_inner = len(inner_hex_data)
    for i in range(n_inner):
        for j in range(i + 1, n_inner):
            x1, y1, a1 = inner_hex_data[i]
            x2, y2, a2 = inner_hex_data[j]
            v1 = _hex_vertices(x1, y1, 1.0, a1)
            v2 = _hex_vertices(x2, y2, 1.0, a2)
            
            n1 = _hex_edge_normals(v1)
            n2 = _hex_edge_normals(v2)
            all_normals = np.vstack([n1, n2])
            
            separated = False
            for n in all_normals:
                proj1 = v1 @ n
                proj2 = v2 @ n
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                    separated = True
                    break
            
            if not separated:
                return False
    
    return True


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    
    Structural change: Use a contact-graph based parameterization with a small number of
    free parameters (lattice spacing, rotations, cluster rotation) rather than 33 free
    parameters. This reduces the search space dramatically and allows more effective
    optimization.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    
    def _build_from_params(params):
        """
        Build 11 hexagon positions from a compact parameter set.
        params: [spacing, cluster_rot_deg, outer_rot_deg, 
                 hex_rot_0, hex_rot_1, ..., hex_rot_10]
        spacing: scale factor for inter-hex distances (1.0 = touching)
        cluster_rot: rotation of the entire cluster
        outer_rot: rotation of the outer hexagon
        hex_rot_i: individual rotation of each hexagon
        """
        spacing = params[0]
        cluster_rot = np.radians(params[1])
        outer_rot = params[2]
        hex_rots = params[3:14]
        
        d = np.sqrt(3) * spacing  # center-to-center distance for touching
        
        # Define the 11 positions in a hexagonal cluster pattern
        # Center + 6 first ring + 4 second ring
        positions = [(0.0, 0.0)]  # center
        
        # First ring: 6 hexes at distance d
        for k in range(6):
            ang = cluster_rot + np.radians(60 * k)
            positions.append((d * np.cos(ang), d * np.sin(ang)))
        
        # Second ring: 4 hexes at distance 2*d, at angles offset by 30 degrees
        # Pick 4 of the 6 possible positions for maximum compactness
        second_ring_angles = [30, 90, 150, 210, 270, 330]
        # Use 4 that are symmetric
        chosen_angles = [30, 150, 270, 30]  # This gives a non-standard but potentially compact arrangement
        # Actually, let me use a different topology: 1 center + 6 first ring + 4 in specific gaps
        
        # Better: use the "flower" pattern where 4 second-ring hexes fill 4 of 6 gaps
        for k in [0, 1, 3, 4]:  # Pick 4 gaps
            ang = cluster_rot + np.radians(60 * k + 30)
            dist = d * np.sqrt(3)  # distance to second ring
            positions.append((dist * np.cos(ang), dist * np.sin(ang)))
        
        # Convert to array with rotations
        inner = np.zeros((11, 3))
        for i, (x, y) in enumerate(positions):
            inner[i, 0] = x
            inner[i, 1] = y
            inner[i, 2] = hex_rots[i]
        
        return inner, outer_rot
    
    def _objective(params):
        """
        Objective: minimize enclosing side length with penalty for overlaps.
        """
        if time.time() > deadline - 10:
            return 1e10
        
        inner, outer_rot = _build_from_params(params)
        
        # Compute enclosing side
        side = _min_enclosing_hex_side(inner, outer_rot)
        
        # Check overlaps
        total_penalty = 0.0
        n = len(inner)
        verts_list = []
        for i in range(n):
            x, y, a_deg = inner[i]
            verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
        
        for i in range(n):
            for j in range(i + 1, n):
                v1 = verts_list[i]
                v2 = verts_list[j]
                n1 = _hex_edge_normals(v1)
                n2 = _hex_edge_normals(v2)
                all_normals = np.vstack([n1, n2])
                
                min_gap = float('inf')
                for nrm in all_normals:
                    proj1 = v1 @ nrm
                    proj2 = v2 @ nrm
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    gap = max(max1 - min2, max2 - min1)
                    if gap < min_gap:
                        min_gap = gap
                
                if min_gap < 1e-6:
                    total_penalty += (1e-6 - min_gap)
        
        # Also penalize if side is too small (below theoretical minimum)
        if side < np.sqrt(11) - 0.1:
            total_penalty += (np.sqrt(11) - side) * 10
        
        return side + 100.0 * total_penalty
    
    # Initial parameters
    # spacing=1.0 (touching), cluster_rot=0, outer_rot=0, all hex rotations=0
    x0 = np.array([1.0, 0.0, 0.0] + [0.0] * 11)
    
    # Bounds
    bounds = [(0.8, 1.5),  # spacing
              (-30, 30),   # cluster_rot
              (0, 60),     # outer_rot
              ] + [(-180, 180)] * 11  # individual hex rotations
    
    # Phase 1: Multi-start with Nelder-Mead
    best_params = x0.copy()
    best_obj = _objective(x0)
    best_inner, best_outer_rot = _build_from_params(x0)
    
    # Generate diverse starting points
    rng = np.random.default_rng(42)
    start_points = [x0.copy()]
    for _ in range(10):
        pt = x0.copy()
        pt[0] = rng.uniform(0.85, 1.35)  # spacing
        pt[1] = rng.uniform(-30, 30)     # cluster_rot
        pt[2] = rng.uniform(0, 60)       # outer_rot
        pt[3:] = rng.uniform(-30, 30, 11)  # hex rotations
        start_points.append(pt)
    
    for i, x_start in enumerate(start_points):
        if time.time() > deadline - 60:
            break
        
        try:
            res = minimize(_objective, x_start, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                best_inner, best_outer_rot = _build_from_params(best_params)
        except Exception:
            pass
    
    # Phase 2: Fine-tune with smaller step sizes
    if time.time() < deadline - 30:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                  'initial_simplex': None})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x.copy()
                best_inner, best_outer_rot = _build_from_params(best_params)
        except Exception:
            pass
    
    # Phase 3: Try different topologies
    # Topology 2: Linear-ish arrangement
    if time.time() < deadline - 30:
        def _build_topology2(params):
            """
            Alternative topology: 3 rows of 3, 5, 3 hexes
            params: [h_spacing, v_spacing, row_offset, cluster_rot, outer_rot, hex_rots...]
            """
            h_sp = np.sqrt(3) * params[0]  # horizontal spacing
            v_sp = 1.5 * params[0]          # vertical spacing
            row_offset = np.sqrt(3) / 2 * params[0]  # offset between rows
            cluster_rot = np.radians(params[3])
            outer_rot = params[4]
            hex_rots = params[5:16]
            
            positions = []
            # Row 0 (top): 3 hexes
            for i in range(3):
                x = (i - 1) * h_sp + row_offset
                y = v_sp
                positions.append((x, y))
            # Row 1 (mid): 5 hexes
            for i in range(5):
                x = (i - 2) * h_sp
                y = 0.0
                positions.append((x, y))
            # Row 2 (bottom): 3 hexes
            for i in range(3):
                x = (i - 1) * h_sp + row_offset
                y = -v_sp
                positions.append((x, y))
            
            # Apply cluster rotation
            c, s = np.cos(cluster_rot), np.sin(cluster_rot)
            rotated_positions = []
            for x, y in positions:
                rx = x * c - y * s
                ry = x * s + y * c
                rotated_positions.append((rx, ry))
            
            inner = np.zeros((11, 3))
            for i, (x, y) in enumerate(rotated_positions):
                inner[i, 0] = x
                inner[i, 1] = y
                inner[i, 2] = hex_rots[i]
            
            return inner, outer_rot
        
        def _objective2(params):
            if time.time() > deadline - 10:
                return 1e10
            inner, outer_rot = _build_topology2(params)
            side = _min_enclosing_hex_side(inner, outer_rot)
            
            total_penalty = 0.0
            n = len(inner)
            verts_list = []
            for i in range(n):
                x, y, a_deg = inner[i]
                verts_list.append(_hex_vertices(x, y, 1.0, a_deg))
            
            for i in range(n):
                for j in range(i + 1, n):
                    v1 = verts_list[i]
                    v2 = verts_list[j]
                    n1 = _hex_edge_normals(v1)
                    n2 = _hex_edge_normals(v2)
                    all_normals = np.vstack([n1, n2])
                    
                    min_gap = float('inf')
                    for nrm in all_normals:
                        proj1 = v1 @ nrm
                        proj2 = v2 @ nrm
                        min1, max1 = np.min(proj1), np.max(proj1)
                        min2, max2 = np.min(proj2), np.max(proj2)
                        gap = max(max1 - min2, max2 - min1)
                        if gap < min_gap:
                            min_gap = gap
                    
                    if min_gap < 1e-6:
                        total_penalty += (1e-6 - min_gap)
            
            return side + 100.0 * total_penalty
        
        x0_2 = np.array([1.0, 0, 0, 0, 0] + [0.0] * 11)
        bounds_2 = [(0.8, 1.5), (0, 0), (0, 0), (-30, 30), (0, 60)] + [(-180, 180)] * 11
        
        try:
            res = minimize(_objective2, x0_2, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_obj:
                inner2, outer_rot2 = _build_topology2(res.x)
                side2 = _min_enclosing_hex_side(inner2, outer_rot2)
                if _check_validity(inner2, np.array([0.0, 0.0, outer_rot2]), side2):
                    best_inner = inner2
                    best_outer_rot = outer_rot2
                    best_obj = res.fun
        except Exception:
            pass
    
    # Final: compute side and check validity
    best_side = _min_enclosing_hex_side(best_inner, best_outer_rot)
    best_outer = np.array([0.0, 0.0, best_outer_rot])
    
    # Validate
    if not _check_validity(best_inner, best_outer, best_side):
        # Try to fix by slightly increasing side
        best_side += 0.01
        if not _check_validity(best_inner, best_outer, best_side):
            # Fallback to a known valid configuration
            d = np.sqrt(3)
            inner_fb = np.zeros((11, 3))
            inner_fb[0] = [0, 0, 0]
            for k in range(6):
                ang = np.radians(60 * k)
                inner_fb[k+1] = [d * np.cos(ang), d * np.sin(ang), 0]
            for k in [0, 1, 3, 4]:
                ang = np.radians(60 * k + 30)
                dist = d * np.sqrt(3)
                inner_fb[k+7] = [dist * np.cos(ang), dist * np.sin(ang), 0]
            best_inner = inner_fb
            best_outer = np.array([0.0, 0.0, 0.0])
            best_side = _min_enclosing_hex_side(best_inner, 0.0)
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
