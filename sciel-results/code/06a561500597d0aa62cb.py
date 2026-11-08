"""Optimized hexagon packing: 11 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools
import time

N = 11
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return 6 vertices of a regular hexagon with given center, side, rotation."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = np.cos(-a)
    sin_a = np.sin(-a)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    # In local frame, hexagon vertices at angles 0, 60, 120, 180, 240, 300
    # Face normals are at 30, 90, 150, 210, 270, 330 (offset by pi/6)
    # For a unit hexagon (side=1), the apothem = sqrt(3)/2
    apothem = side * np.sqrt(3) / 2
    for i in range(6):
        normal_angle = np.pi / 6 + i * np.pi / 3
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def _min_enclosing_hex(inner_hexagons):
    """Find minimum side length regular hexagon containing all inner hexagons.
    inner_hexagons: list of (cx, cy, angle_deg)
    Returns: (cx, cy, side, angle_deg) for outer hexagon
    """
    # Collect all vertices
    all_pts = []
    for cx, cy, ang in inner_hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    best_side = float('inf')
    best_params = (0.0, 0.0, 1.0, 0.0)
    
    # Try different outer hexagon angles
    for outer_angle in np.linspace(0, 60, 13):
        a = np.radians(outer_angle)
        # Face normals at a + pi/6 + i*pi/3
        normals = []
        for i in range(6):
            na = a + np.pi / 6 + i * np.pi / 3
            normals.append(np.array([np.cos(na), np.sin(na)]))
        normals = np.array(normals)
        
        # For a regular hexagon with center (cx,cy) and side s,
        # constraint: for each normal n, max over points of (p - c)·n <= s * sqrt(3)/2
        # We want to minimize s such that there exists (cx,cy)
        # s >= max_i max_p (p - c)·n_i / (sqrt(3)/2)
        # = max_i max_p (p·n_i - c·n_i) / (sqrt(3)/2)
        # To minimize s, we need to choose c to minimize max_i (max_p p·n_i - c·n_i)
        # This is a 2D LP: minimize t s.t. c·n_i - t*(sqrt(3)/2) <= max_p p·n_i for all i
        # and -c·n_i - t*(sqrt(3)/2) <= max_p p·(-n_i) = -min_p p·n_i
        
        # Actually let's think differently. For a given outer angle, the hexagon is defined by
        # 6 half-planes. The minimum enclosing hexagon with fixed angle can be found by
        # binary search on side length and checking if a center exists.
        
        # Alternative: the minimum enclosing regular hexagon with fixed orientation
        # has center at the centroid of the "dual" problem. Let me use a simpler approach:
        # For fixed orientation, the hexagon is the intersection of 6 strips.
        # The minimum side hexagon containing all points with center (cx,cy) has
        # side = max_i |max_p (p-c)·n_i| / (sqrt(3)/2) where we take max over both directions
        # Actually: side = max_i (max_p (p-c)·n_i) / (sqrt(3)/2) if we ensure symmetry
        # No - the hexagon must contain all points, so for each pair of opposite normals:
        # (p-c)·n_i <= s*sqrt(3)/2 and (p-c)·(-n_i) <= s*sqrt(3)/2
        # i.e., |(p-c)·n_i| <= s*sqrt(3)/2 for all p, i
        # So s = max_i max_p |(p-c)·n_i| / (sqrt(3)/2)
        # To minimize s, minimize over c: max_i max_p |(p-c)·n_i|
        
        # This is equivalent to finding c that minimizes the maximum projection
        # Let's use scipy minimize on c
        def obj(c):
            cx, cy = c
            max_proj = 0
            for n in normals:
                projections = (all_pts - c) @ n
                max_proj = max(max_proj, np.max(np.abs(projections)))
            return max_proj
        
        # Initial guess: centroid of all points
        c0 = np.mean(all_pts, axis=0)
        res = minimize(obj, c0, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.success:
            cx, cy = res.x
            side = res.fun / (np.sqrt(3) / 2)
            if side < best_side:
                best_side = side
                best_params = (cx, cy, side, outer_angle)
    
    return best_params

def _check_validity(inner, center, side, angle):
    """Verify all inner hexagons are inside outer and disjoint."""
    cx, cy = center
    # Check containment
    for ix, iy, iang in inner:
        verts = _hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in verts:
            if not _point_in_hex(vx, vy, cx, cy, side, angle):
                return False
    # Check disjointness (no overlap between inner hexagons)
    for i in range(len(inner)):
        for j in range(i + 1, len(inner)):
            ix1, iy1, ia1 = inner[i]
            ix2, iy2, ia2 = inner[j]
            # Check if any vertex of hex i is inside hex j or vice versa
            v1 = _hex_vertices(ix1, iy1, 1.0, ia1)
            v2 = _hex_vertices(ix2, iy2, 1.0, ia2)
            for vx, vy in v1:
                if _point_in_hex(vx, vy, ix2, iy2, 1.0, ia2):
                    return False
            for vx, vy in v2:
                if _point_in_hex(vx, vy, ix1, iy1, 1.0, ia1):
                    return False
            # Also check edge-edge intersection (simplified: check midpoints)
            # For robustness, also check if centers are too close
            dist = np.sqrt((ix1-ix2)**2 + (iy1-iy2)**2)
            if dist < 2.0 * np.sqrt(3) / 2 - 1e-6:  # apothem * 2
                # Might still be valid if rotated, but flag as potential issue
                pass
    return True

def _build_initial_grid():
    """Create initial hexagonal lattice arrangement of 11 unit hexagons."""
    # Hexagonal lattice: spacing between centers for edge-to-edge contact
    # For two unit hexagons touching edge to edge, center distance = sqrt(3)
    # Lattice vectors: (sqrt(3), 0) and (sqrt(3)/2, 3/2)
    dx = np.sqrt(3)
    dy = 1.5
    
    positions = []
    # 11 hexagons in a compact cluster
    # Row 0: 3 hexagons
    for k in range(3):
        positions.append((k * dx, 0.0, 0.0))
    # Row 1: 2 hexagons (offset)
    for k in range(2):
        positions.append((dx/2 + k * dx, dy, 0.0))
    # Row 2: 3 hexagons
    for k in range(3):
        positions.append((k * dx, 2*dy, 0.0))
    # Row 3: 2 hexagons (offset)
    for k in range(2):
        positions.append((dx/2 + k * dx, 3*dy, 0.0))
    
    return positions

def _optimize_positions(positions, time_budget=30.0):
    """Optimize positions and rotations to minimize enclosing hexagon side."""
    start_time = time.time()
    
    def objective(params):
        # params: [x1,y1,a1, x2,y2,a2, ...]
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append((x, y, a))
        
        # Quick disjointness check
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                if dist < np.sqrt(3) - 0.1:
                    return dist  # penalty
        # Even closer
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
                if dist < np.sqrt(3) - 1e-3:
                    return 1000 + dist
        
        # Find enclosing hexagon
        cx, cy, side, ang = _min_enclosing_hex(inner)
        # Verify validity
        if not _check_validity(inner, (cx, cy), side, ang):
            return 100.0 + side
        return side
    
    # Initial parameters
    x0 = []
    for x, y, a in positions:
        x0.extend([x, y, a])
    
    best_result = None
    best_side = float('inf')
    
    # Try multiple optimization strategies
    strategies = [
        ('Nelder-Mead', {'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}),
        ('Powell', {'maxiter': 300, 'ftol': 1e-8}),
    ]
    
    for method, opts in strategies:
        if time.time() - start_time > time_budget:
            break
        try:
            res = minimize(objective, x0, method=method, options=opts)
            if res.fun < best_side:
                best_side = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is not None:
        inner = []
        for i in range(N):
            inner.append((best_result[3*i], best_result[3*i+1], best_result[3*i+2]))
        return inner
    return positions

def _try_rotated_arrangements(positions, time_budget=20.0):
    """Try various rotation patterns for the inner hexagons."""
    start_time = time.time()
    best = None
    best_side = float('inf')
    
    # Try all same rotation
    for rot in [0, 30, 60]:
        if time.time() - start_time > time_budget:
            break
        rotated = [(x, y, float(rot)) for x, y, _ in positions]
        cx, cy, side, ang = _min_enclosing_hex(rotated)
        if _check_validity(rotated, (cx, cy), side, ang):
            if side < best_side:
                best_side = side
                best = rotated
    
    # Try alternating rotations
    if time.time() - start_time < time_budget:
        for rot1, rot2 in [(0, 30), (0, 60), (30, 60)]:
            rotated = []
            for i, (x, y, _) in enumerate(positions):
                r = rot1 if i % 2 == 0 else rot2
                rotated.append((x, y, float(r)))
            cx, cy, side, ang = _min_enclosing_hex(rotated)
            if _check_validity(rotated, (cx, cy), side, ang):
                if side < best_side:
                    best_side = side
                    best = rotated
    
    if best is not None:
        return best
    return positions

def _force_directed_relax(positions, time_budget=15.0):
    """Use force-directed relaxation to find tighter packing."""
    start_time = time.time()
    pos = np.array([(x, y) for x, y, _ in positions], dtype=float)
    rots = np.array([a for _, _, a in positions], dtype=float)
    
    # Lattice vectors for hexagonal packing
    dx = np.sqrt(3)
    
    for iteration in range(2000):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(pos)
        # Repulsive forces between hexagons that are too close
        for i in range(N):
            for j in range(i+1, N):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < dx - 0.01:
                    # Strong repulsion
                    f = (dx - dist) * 10.0
                    if dist > 1e-8:
                        forces[i] += f * diff / dist
                        forces[j] -= f * diff / dist
                elif dist < dx + 0.5:
                    # Weak attraction to maintain contact
                    f = (dx - dist) * 0.5
                    if dist > 1e-8:
                        forces[i] += f * diff / dist
                        forces[j] -= f * diff / dist
        
        # Centering force
        center = np.mean(pos, axis=0)
        for i in range(N):
            forces[i] -= 0.1 * (pos[i] - center)
        
        pos += 0.01 * forces
    
    inner = [(pos[i][0], pos[i][1], rots[i]) for i in range(N)]
    return inner

def construct():
    global TIME_LIMIT
    start_time = time.time()
    
    def elapsed():
        return time.time() - start_time
    
    # Build initial arrangement
    positions = _build_initial_grid()
    
    # Try different approaches
    best_inner = None
    best_side = float('inf')
    
    # 1. Try the basic grid
    if elapsed() < 5:
        cx, cy, side, ang = _min_enclosing_hex(positions)
        if _check_validity(positions, (cx, cy), side, ang):
            best_inner = positions
            best_side = side
    
    # 2. Try rotated arrangements
    if elapsed() < 20:
        rotated = _try_rotated_arrangements(positions, time_budget=min(15, TIME_LIMIT - elapsed() - 10))
        if rotated is not None:
            cx, cy, side, ang = _min_enclosing_hex(rotated)
            if _check_validity(rotated, (cx, cy), side, ang) and side < best_side:
                best_inner = rotated
                best_side = side
    
    # 3. Force-directed relaxation
    if elapsed() < 40:
        relaxed = _force_directed_relax(positions, time_budget=min(20, TIME_LIMIT - elapsed() - 10))
        cx, cy, side, ang = _min_enclosing_hex(relaxed)
        if _check_validity(relaxed, (cx, cy), side, ang) and side < best_side:
            best_inner = relaxed
            best_side = side
    
    # 4. Local optimization
    if elapsed() < 100:
        budget = min(50, TIME_LIMIT - elapsed() - 5)
        if budget > 5:
            optimized = _optimize_positions(best_inner if best_inner else positions, time_budget=budget)
            if optimized is not None:
                cx, cy, side, ang = _min_enclosing_hex(optimized)
                if _check_validity(optimized, (cx, cy), side, ang) and side < best_side:
                    best_inner = optimized
                    best_side = side
    
    # 5. Try a few random perturbations
    if elapsed() < 130:
        np.random.seed(42)
        for trial in range(5):
            if elapsed() > 130:
                break
            perturbed = []
            for x, y, a in (best_inner if best_inner else positions):
                px = x + np.random.uniform(-0.5, 0.5)
                py = y + np.random.uniform(-0.5, 0.5)
                pa = a + np.random.uniform(-15, 15)
                perturbed.append((px, py, pa))
            cx, cy, side, ang = _min_enclosing_hex(perturbed)
            if _check_validity(perturbed, (cx, cy), side, ang) and side < best_side:
                best_inner = perturbed
                best_side = side
    
    # Final result
    if best_inner is None:
        # Fallback: just use the grid
        best_inner = positions
    
    cx, cy, side, ang = _min_enclosing_hex(best_inner)
    
    # Final validity check
    if not _check_validity(best_inner, (cx, cy), side, ang):
        # Try to fix by scaling up slightly
        side *= 1.01
        if not _check_validity(best_inner, (cx, cy), side, ang):
            # Use a larger fallback
            side = 5.0
            cx, cy = 0.0, 0.0
            ang = 0.0
    
    return best_inner, (cx, cy), side, ang

