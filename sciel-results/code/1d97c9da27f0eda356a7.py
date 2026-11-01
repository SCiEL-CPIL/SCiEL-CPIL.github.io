import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Base hexagon vertices (side 1, angle 0): flat-top orientation
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_verts = np.array([(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)])

    def get_hex_verts(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with given center, side, and rotation."""
        a = math.radians(angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        verts = (base_verts * side) @ R.T + np.array([cx, cy])
        return verts  # shape (6, 2)

    def compute_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        ocx, ocy = outer_center
        a = math.radians(outer_angle)
        # 6 outward normals of outer hexagon (at 30, 90, 150, 210, 270, 330 degrees relative to rotation)
        normals = []
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            normals.append((math.cos(theta), math.sin(theta)))
        
        apothem_factor = math.sqrt(3) / 2  # apothem = side * sqrt(3)/2
        
        max_S = 0.0
        for nx, ny in normals:
            max_proj = 0.0
            for k in range(N):
                ix, iy = inner_positions[k]
                ia = inner_angles[k]
                verts = get_hex_verts(ix, iy, 1.0, ia)
                # Project all vertices onto normal
                rel = verts - np.array([ocx, ocy])
                projs = rel[:, 0] * nx + rel[:, 1] * ny
                mp = np.max(projs)
                if mp > max_proj:
                    max_proj = mp
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def hex_overlap_distance(va, vb):
        """
        Compute a smooth measure of overlap between two hexagons.
        Returns 0 if disjoint, negative if overlapping.
        Uses SAT: check all 12 edge normals (6 from each hex).
        """
        # Edge normals for hexagon a
        min_sep = float('inf')
        for i in range(6):
            # Edge from vertex i to vertex (i+1)%6
            v1, v2 = va[i], va[(i+1) % 6]
            edge = v2 - v1
            # Normal (outward)
            nx, ny = -edge[1], edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-10:
                continue
            nx, ny = nx/norm, ny/norm
            # Project both hexagons onto this normal
            proj_a = va @ np.array([nx, ny])
            proj_b = vb @ np.array([nx, ny])
            sep_a = np.max(proj_a) - np.min(proj_b)
            sep_b = np.max(proj_b) - np.min(proj_a)
            # If they overlap on this axis, separation is negative
            sep = min(sep_a, sep_b)
            if sep < min_sep:
                min_sep = sep
            # If separated on any axis, no overlap
            if sep > 1e-9:
                return 0.0
        
        # Also check edges of b
        for i in range(6):
            v1, v2 = vb[i], vb[(i+1) % 6]
            edge = v2 - v1
            nx, ny = -edge[1], edge[0]
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-10:
                continue
            nx, ny = nx/norm, ny/norm
            proj_a = va @ np.array([nx, ny])
            proj_b = vb @ np.array([nx, ny])
            sep_a = np.max(proj_a) - np.min(proj_b)
            sep_b = np.max(proj_b) - np.min(proj_a)
            sep = min(sep_a, sep_b)
            if sep < min_sep:
                min_sep = sep
            if sep > 1e-9:
                return 0.0
        
        # If we get here, check if actually overlapping
        # min_sep is the minimum separation (most negative = most overlap)
        if min_sep > 1e-9:
            return 0.0
        else:
            return -min_sep  # negative value indicating overlap depth

    def objective(params):
        """
        Objective function for scipy optimizer.
        params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10, cx, cy, outer_angle]
        Returns: outer_side + penalty * overlap_penalty
        """
        inner_positions = []
        inner_angles = []
        for k in range(N):
            inner_positions.append((params[3*k], params[3*k+1]))
            inner_angles.append(params[3*k+2])
        ocx = params[3*N]
        ocy = params[3*N+1]
        outer_angle = params[3*N+2]
        
        # Compute outer side
        S = compute_outer_side(inner_positions, inner_angles, (ocx, ocy), outer_angle)
        
        # Compute overlap penalty
        penalty = 0.0
        penalty_weight = 100.0
        for i in range(N):
            for j in range(i+1, N):
                va = get_hex_verts(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
                vb = get_hex_verts(inner_positions[j][0], inner_positions[j][1], 1.0, inner_angles[j])
                dist = hex_overlap_distance(va, vb)
                if dist < 0:
                    penalty += dist * dist  # quadratic penalty for overlap
        
        # Also penalize if inner hexagons go outside the outer hexagon
        outer_verts = get_hex_verts(ocx, ocy, S, outer_angle)
        containment_penalty = 0.0
        for k in range(N):
            verts = get_hex_verts(inner_positions[k][0], inner_positions[k][1], 1.0, inner_angles[k])
            for v in verts:
                # Check if v is inside outer hexagon using point-in-polygon
                x, y = v
                inside = False
                j = 5
                for i in range(6):
                    xi, yi = outer_verts[i]
                    xj, yj = outer_verts[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                if not inside:
                    # Penalty proportional to how far outside
                    containment_penalty += 1.0
        
        return S + penalty_weight * penalty + 50.0 * containment_penalty

    def is_valid(inner_positions, inner_angles, ocx, ocy, S, outer_angle):
        """Check if configuration is valid (no overlaps, all contained)."""
        outer_verts = get_hex_verts(ocx, ocy, S, outer_angle)
        
        # Check containment
        for k in range(N):
            verts = get_hex_verts(inner_positions[k][0], inner_positions[k][1], 1.0, inner_angles[k])
            for v in verts:
                x, y = v
                inside = False
                j = 5
                for i in range(6):
                    xi, yi = outer_verts[i]
                    xj, yj = outer_verts[j]
                    if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                        inside = not inside
                    j = i
                if not inside:
                    return False
        
        # Check disjointness
        all_verts = []
        for k in range(N):
            all_verts.append(get_hex_verts(inner_positions[k][0], inner_positions[k][1], 1.0, inner_angles[k]))
        
        for i in range(N):
            for j in range(i+1, N):
                if hex_overlap_distance(all_verts[i], all_verts[j]) < -1e-6:
                    return False
        return True

    def local_refine(inner_positions, inner_angles, ocx, ocy, outer_angle, seed):
        """Use scipy optimizer to refine a configuration."""
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        
        if time.time() > deadline:
            return
        
        # Build parameter vector
        params0 = []
        for k in range(N):
            params0.extend([inner_positions[k][0], inner_positions[k][1], inner_angles[k]])
        params0.extend([ocx, ocy, outer_angle])
        
        # Try a few optimization methods
        for method in ['L-BFGS-B', 'BFGS']:
            if time.time() > deadline:
                break
            try:
                result = minimize(
                    objective,
                    params0,
                    method=method,
                    options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4}
                )
                
                if result.success or result.fun < best_S:
                    # Extract solution
                    new_positions = []
                    new_angles = []
                    for k in range(N):
                        new_positions.append((result.x[3*k], result.x[3*k+1]))
                        new_angles.append(result.x[3*k+2])
                    new_ocx = result.x[3*N]
                    new_ocy = result.x[3*N+1]
                    new_outer_angle = result.x[3*N+2]
                    
                    # Recompute exact S
                    new_S = compute_outer_side(new_positions, new_angles, (new_ocx, new_ocy), new_outer_angle)
                    
                    # Check validity
                    if is_valid(new_positions, new_angles, new_ocx, new_ocy, new_S, new_outer_angle):
                        if new_S < best_S:
                            best_S = new_S
                            best_inner = [[new_positions[k][0], new_positions[k][1], new_angles[k]] for k in range(N)]
                            best_center = (new_ocx, new_ocy)
                            best_angle = new_outer_angle
                            best_valid = True
            except Exception:
                pass

    # Generate initial configurations using hexagonal lattice
    sp = math.sqrt(3)  # touching distance for unit hexagons
    
    def make_lattice_config(n_per_row, angle_offset=0.0):
        """Create a configuration on a hexagonal lattice."""
        positions = []
        angles = []
        # Place hexagons in a hexagonal pattern
        # Center + 6 around + 12 around
        positions.append((0.0, 0.0))
        angles.append(0.0)
        for i in range(6):
            theta = math.radians(60 * i + angle_offset)
            positions.append((sp * math.cos(theta), sp * math.sin(theta)))
            angles.append(0.0)
        for i in range(12):
            if i < 6:
                theta = math.radians(60 * i + 30 + angle_offset)
                r = 2 * sp
            else:
                theta = math.radians(60 * (i-6) + angle_offset)
                r = 2 * sp
            positions.append((r * math.cos(theta), r * math.sin(theta)))
            angles.append(0.0)
        
        # If we have more than N, trim; if fewer, add
        while len(positions) < N:
            theta = random.uniform(0, 2*math.pi)
            r = random.uniform(0, 2*sp)
            positions.append((r * math.cos(theta), r * math.sin(theta)))
            angles.append(random.choice([0, 30, 60]))
        
        return positions[:N], angles[:N]

    # Run multiple restarts with different initial configurations
    rng = random.Random(42)
    n_restarts = 30
    restarts_done = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:  # Leave 10s safety margin
            break
        
        # Different initial configurations
        if restart % 3 == 0:
            # Hexagonal lattice
            positions, angles = make_lattice_config(11, angle_offset=rng.uniform(0, 30))
            ocx, ocy = 0.0, 0.0
            outer_angle = rng.choice([0, 30, 60])
        elif restart % 3 == 1:
            # Random perturbation of lattice
            positions, angles = make_lattice_config(11, angle_offset=0)
            for k in range(N):
                positions[k] = (positions[k][0] + rng.gauss(0, 0.3), positions[k][1] + rng.gauss(0, 0.3))
                angles[k] = rng.choice([0, 30, 60, 90, 120, 150])
            ocx, ocy = rng.gauss(0, 0.5), rng.gauss(0, 0.5)
            outer_angle = rng.uniform(0, 60)
        else:
            # Fully random
            positions = [(rng.uniform(-2, 2), rng.uniform(-2, 2)) for _ in range(N)]
            angles = [rng.choice([0, 30, 60, 90, 120, 150]) for _ in range(N)]
            ocx, ocy = rng.uniform(-1, 1), rng.uniform(-1, 1)
            outer_angle = rng.uniform(0, 60)
        
        # Check if this initial config is already valid
        S_init = compute_outer_side(positions, angles, (ocx, ocy), outer_angle)
        if is_valid(positions, angles, ocx, ocy, S_init, outer_angle):
            if S_init < best_S:
                best_S = S_init
                best_inner = [[positions[k][0], positions[k][1], angles[k]] for k in range(N)]
                best_center = (ocx, ocy)
                best_angle = outer_angle
                best_valid = True
        
        # Run local refinement
        local_refine(positions, angles, ocx, ocy, outer_angle, restart)
        restarts_done += 1

    # If no valid solution found, create a simple fallback
    if not best_valid or best_inner is None:
        # Place hexagons in a line with touching spacing
        fallback_positions = []
        fallback_angles = []
        for k in range(N):
            fallback_positions.append((k * sp, 0.0))
            fallback_angles.append(0.0)
        
        S_fb = compute_outer_side(fallback_positions, fallback_angles, (0.0, 0.0), 0.0)
        # Scale up until valid
        for attempt in range(100):
            if is_valid(fallback_positions, fallback_angles, (0.0, 0.0), S_fb, 0.0):
                best_S = S_fb
                best_inner = [[fallback_positions[k][0], fallback_positions[k][1], fallback_angles[k]] for k in range(N)]
                best_center = (0.0, 0.0)
                best_angle = 0.0
                best_valid = True
                break
            # Scale positions outward
            for k in range(N):
                fallback_positions[k] = (fallback_positions[k][0] * 1.05, fallback_positions[k][1] * 1.05)
            S_fb = compute_outer_side(fallback_positions, fallback_angles, (0.0, 0.0), 0.0)
        
        if not best_valid:
            # Last resort: return something
            best_S = S_fb
            best_inner = [[fallback_positions[k][0], fallback_positions[k][1], fallback_angles[k]] for k in range(N)]
            best_center = (0.0, 0.0)
            best_angle = 0.0
            best_valid = True

    return best_inner, best_center, best_S, best_angle
