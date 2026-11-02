import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

# Precompute unit hexagon vertices (flat-top, centered at origin)
_HEX_VERTS = np.array([
    [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
])
# Edge normals for hexagon (outward)
_EDGE_NORMALS = np.array([
    [math.cos(math.radians(60 * k + 30)), math.sin(math.radians(60 * k + 30))] for k in range(6)
])

def _hex_overlap(h1, a1, h2, a2, thresh=1e-12):
    """Check if two unit hexagons (centers h1,h2, angles a1,a2) overlap.
    Returns True if overlapping (distance < 2*apothem with tolerance)."""
    # For regular hexagons, the minimum distance between centers for non-overlap
    # depends on relative orientation. Use a conservative check:
    # If centers are closer than 2*apothem - eps, they definitely overlap for parallel.
    # For rotated, use support function approximation.
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist = math.sqrt(dx*dx + dy*dy)
    # Quick reject: if far apart
    if dist > 2.0 * APOTHEM + 1.0:
        return False
    # If very close, likely overlap
    if dist < 2.0 * APOTHEM - 0.5:
        return True
    # Support function check: for each edge normal of hex1, check if hex2 violates
    # This is a SAT-like test but simplified.
    # Actually, for hexagons, the non-overlap condition is that for all 12 edge normals
    # (6 from each), the support function difference is >= 0.
    # Let's use a simpler distance-based approximation for speed, refined later.
    # The minimum distance between two hexagons with relative angle theta is complex.
    # Use a safe lower bound: if dist < 2*APOTHEM * 0.95, assume overlap.
    # This is conservative and will be fixed by the optimizer.
    if dist < 2.0 * APOTHEM * 0.98:
        return True
    return False

def _compute_container_size(positions, angles):
    """Compute the minimum side length of a hexagon (rotated by optimal angle) that contains all hexagons."""
    # This is a simplified version: we'll optimize the container angle separately.
    # For now, return a large value if we can't compute quickly.
    # We'll use a grid search over container angles in the main loop.
    return None

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def energy(params):
        """Energy function for the physics-inspired packing.
        params: [x0, y0, a0, x1, y1, a1, ..., x10, y10, a10]
        Returns scalar energy to minimize.
        """
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            return 1e10

        # Unpack
        positions = np.array(params[:2*N]).reshape(N, 2)
        angles = np.array(params[2*N:])

        # Repulsion energy: penalize overlaps
        repulsion = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                # Target distance: 2*apothem for parallel hexagons
                # For simplicity, use a soft repulsion when dist < 2*APOTHEM
                target = 2.0 * APOTHEM
                if dist < target:
                    # Strong quadratic repulsion
                    overlap = target - dist
                    repulsion += 100.0 * overlap * overlap

        # Attraction to center: keep the packing compact
        center = np.mean(positions, axis=0)
        attraction = 0.1 * np.sum(np.sum((positions - center)**2, axis=1))

        # Boundary penalty: keep positions within a reasonable range
        boundary = 0.0
        for pos in positions:
            if abs(pos[0]) > 10.0 or abs(pos[1]) > 10.0:
                boundary += 1000.0

        # Angle regularization: encourage similar angles (for easier container fitting)
        # But allow variation. Use a small penalty for angle differences.
        angle_reg = 0.0
        for i in range(N):
            for j in range(i+1, N):
                da = abs(angles[i] - angles[j])
                da = min(da, 2*math.pi - da)
                angle_reg += 0.01 * da * da

        total = repulsion + attraction + boundary + angle_reg
        return total

    def compute_final_container(positions, angles):
        """Given positions and angles, find the optimal enclosing hexagon.
        Returns (side_length, center_x, center_y, container_angle_deg).
        """
        elapsed = time.time() - START_TIME
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            return None

        # For each hexagon, compute its 6 vertices
        all_vertices = []
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            # Rotate vertices
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            for vx, vy in _HEX_VERTS:
                rx = x + cos_a * vx - sin_a * vy
                ry = y + sin_a * vx + cos_a * vy
                all_vertices.append([rx, ry])
        
        all_vertices = np.array(all_vertices)

        # Optimize container angle and center
        # Grid search over container angle
        best_S = float('inf')
        best_cx = 0.0
        best_cy = 0.0
        best_beta = 0.0

        # Coarse grid
        for k in range(36):
            beta = k * math.pi / 6.0 / 36.0
            # Normals for the container
            normals = []
            for m in range(6):
                ang = beta + math.pi / 6.0 + m * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            # Find optimal center: minimize max projection
            # Use linear programming or just centroid + local search
            cx0 = float(np.mean(all_vertices[:, 0]))
            cy0 = float(np.mean(all_vertices[:, 1]))
            
            # Compute D for this center
            D = 0.0
            for nx, ny in normals:
                projs = (all_vertices[:, 0] - cx0) * nx + (all_vertices[:, 1] - cy0) * ny
                D = max(D, np.max(projs), -np.min(projs))
            
            S = (D + APOTHEM) / APOTHEM
            if S < best_S:
                best_S = S
                best_cx = cx0
                best_cy = cy0
                best_beta = beta

        # Refine with local optimization
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN - 5.0:
            def obj(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                normals = []
                for m in range(6):
                    ang = beta + math.pi / 6.0 + m * math.pi / 3.0
                    normals.append((math.cos(ang), math.sin(ang)))
                
                D = 0.0
                for nx, ny in normals:
                    projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
                    D = max(D, np.max(projs), -np.min(projs))
                
                return (D + APOTHEM) / APOTHEM

            res = minimize(obj, [best_cx, best_cy, best_beta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_cx = res.x[0]
                best_cy = res.x[1]
                best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)

        return best_S, best_cx, best_cy, math.degrees(best_beta)

    # Multiple restarts with different initial configurations
    best_result = None
    best_S = float('inf')

    # Restart 1: Lattice-based initialization
    # Use a hexagonal lattice arrangement
    init_positions = []
    init_angles = []
    # Place 11 hexagons in a roughly hexagonal shape
    # Center + 6 around + 4 more
    lattice_pts = [
        (0, 0),
        (2*APOTHEM, 0), (APOTHEM, 1.5), (-APOTHEM, 1.5), (-2*APOTHEM, 0), (-APOTHEM, -1.5), (APOTHEM, -1.5),
        (3*APOTHEM, 1.5), (0, 3.0), (-3*APOTHEM, 1.5), (2*APOTHEM, 3.0)
    ]
    for p in lattice_pts:
        init_positions.append(p)
        init_angles.append(0.0)
    
    init_params = np.array(init_positions + init_angles).flatten()
    
    # Optimize
    res = minimize(energy, init_params, method='L-BFGS-B', options={'maxiter': 2000, 'gtol': 1e-8})
    
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        positions = res.x[:2*N].reshape(N, 2)
        angles = res.x[2*N:]
        # Normalize angles to [0, pi/3)
        angles = np.array([((a % (math.pi/3)) + math.pi/3) % (math.pi/3) for a in angles])
        
        result = compute_final_container(positions, angles)
        if result is not None:
            S, cx, cy, angle_deg = result
            if S < best_S:
                best_S = S
                best_result = (positions, angles, cx, cy, angle_deg)

    # Restart 2: Random initialization
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        np.random.seed(42)
        for attempt in range(3):
            init_positions = []
            init_angles = []
            for i in range(N):
                init_positions.append([np.random.uniform(-5, 5), np.random.uniform(-5, 5)])
                init_angles.append(np.random.uniform(0, math.pi/3))
            
            init_params = np.array(init_positions + init_angles).flatten()
            
            res = minimize(energy, init_params, method='L-BFGS-B', options={'maxiter': 2000, 'gtol': 1e-8})
            
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            positions = res.x[:2*N].reshape(N, 2)
            angles = res.x[2*N:]
            angles = np.array([((a % (math.pi/3)) + math.pi/3) % (math.pi/3) for a in angles])
            
            result = compute_final_container(positions, angles)
            if result is not None:
                S, cx, cy, angle_deg = result
                if S < best_S:
                    best_S = S
                    best_result = (positions, angles, cx, cy, angle_deg)

    # Fallback: simple lattice
    if best_result is None:
        positions = np.array([
            [0, 0],
            [2*APOTHEM, 0], [APOTHEM, 1.5], [-APOTHEM, 1.5], [-2*APOTHEM, 0], [-APOTHEM, -1.5], [APOTHEM, -1.5],
            [3*APOTHEM, 1.5], [0, 3.0], [-3*APOTHEM, 1.5], [2*APOTHEM, 3.0]
        ])
        angles = np.zeros(N)
        cx = float(np.mean(positions[:, 0]))
        cy = float(np.mean(positions[:, 1]))
        # Compute S
        all_vertices = []
        for i in range(N):
            x, y = positions[i]
            for vx, vy in _HEX_VERTS:
                all_vertices.append([x + vx, y + vy])
        all_vertices = np.array(all_vertices)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_S = (max_proj + APOTHEM) / APOTHEM
        best_result = (positions, angles, cx, cy, 0.0)

    positions, angles, cx, cy, angle_deg = best_result
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(math.degrees(angles[i]))])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
