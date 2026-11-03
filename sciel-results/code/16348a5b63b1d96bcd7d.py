import math
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: trivial configuration
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Force-directed relaxation to find dense packing
    # We model the problem as: place 11 unit hexagons to minimize the
    # side length of the minimal enclosing regular hexagon.
    
    # Strategy:
    # 1. Use a force-directed method to pack hexagons densely (minimize diameter-like metric)
    # 2. Then compute the minimal enclosing hexagon for the resulting cluster
    
    # The key insight: for a set of points (hexagon centers), the minimal enclosing
    # regular hexagon side S is determined by:
    #   S = (max_k max_i |projection of (p_i - c) onto n_k| + APOTHEM) / APOTHEM
    # where n_k are the 6 outward normals of the enclosing hexagon, c is its center.
    # Minimizing S over c and rotation angle theta.
    
    # But first, we need good positions for the 11 hexagons.
    
    # Force-directed approach:
    # - Attractive force pulling all hexagons toward a common center (minimizes spread)
    # - Repulsive force between pairs of hexagons that are too close (prevents overlap)
    # - We iterate until convergence
    
    def hex_overlap_check(p1, p2, s1=1.0, s2=1.0):
        """Check if two unit hexagons at positions p1, p2 overlap.
        For two regular hexagons of side 1, they don't overlap iff
        the distance between centers >= 2 (since the minimum distance
        between non-overlapping hexagon centers is 2*apothem = sqrt(3) for
        edge-to-edge, but actually for point-to-point it's 2.
        Wait: two unit hexagons (side=1) have inradius = sqrt(3)/2.
        The minimum center-to-center distance for non-overlap depends on orientation.
        For same orientation, the minimum distance is 2*sqrt(3)/2 * 2 = ... 
        Actually, two unit hexagons with same orientation: the minimum distance
        between centers for no overlap is 2 (vertex to vertex) or sqrt(3) (edge to edge).
        The critical distance is when a vertex of one touches an edge of the other.
        For identical orientation, the minimum non-overlapping distance is 2*apothem = sqrt(3) ≈ 1.732.
        Wait no: apothem = sqrt(3)/2. Two hexagons edge-to-edge: center distance = 2*apothem = sqrt(3).
        But if rotated, it could be different. For safety, let's use a conservative repulsion.
        """
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        # Conservative: minimum distance for non-overlap of two unit hexagons is sqrt(3) ≈ 1.732
        # (when they share an edge)
        return dist < math.sqrt(3) - 1e-9

    def repulsive_force(p1, p2, strength=100.0):
        """Compute repulsive force vector on p1 due to p2."""
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 1e-6:
            return (0.0, 0.0)
        # Minimum non-overlap distance for two unit hexagons
        d_min = math.sqrt(3)
        if dist >= d_min:
            return (0.0, 0.0)
        # Force proportional to penetration depth
        overlap = d_min - dist
        fx = strength * overlap * dx / dist
        fy = strength * overlap * dy / dist
        return (fx, fy)

    def force_directed_pack(n_iter=2000, dt=0.01, attract_strength=0.5):
        """Pack n hexagons using force-directed method."""
        # Initialize positions in a rough hexagonal pattern
        np.random.seed(42)
        positions = np.zeros((N, 2))
        # Start with a hexagonal lattice pattern
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (-2, 0), (0, 2), (0, -2)
        ]
        for i, (q, r) in enumerate(axial_coords):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions[i] = [x, y]
        
        # Add small random perturbation
        positions += np.random.uniform(-0.1, 0.1, (N, 2))
        
        for it in range(n_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            forces = np.zeros((N, 2))
            
            # Attractive force toward centroid
            centroid = np.mean(positions, axis=0)
            for i in range(N):
                dx = centroid[0] - positions[i][0]
                dy = centroid[1] - positions[i][1]
                forces[i][0] += attract_strength * dx
                forces[i][1] += attract_strength * dy
            
            # Repulsive forces between pairs
            for i in range(N):
                for j in range(i+1, N):
                    fx, fy = repulsive_force(positions[i], positions[j], strength=50.0)
                    forces[i][0] += fx
                    forces[i][1] += fy
                    forces[j][0] -= fx
                    forces[j][1] -= fy
            
            # Update positions
            positions += dt * forces
            
            # Decay learning rate
            dt *= 0.999
        
        return positions

    def compute_min_enclosing_hexagon(positions):
        """Compute the minimal enclosing regular hexagon for the given positions.
        Returns (side_length, center, rotation_angle)."""
        # The minimal enclosing hexagon is determined by:
        # S = (D + APOTHEM) / APOTHEM
        # where D = max over 6 directions of max_i |projection of (p_i - c) onto n_k|
        # We need to minimize D over center c and rotation angle theta.
        
        # For a fixed rotation theta, the optimal center c is the one that
        # minimizes the maximum projection in each of the 6 directions.
        # This is a linear programming problem.
        
        best_D = float('inf')
        best_theta = 0.0
        best_c = (0.0, 0.0)
        
        # Grid search over theta (0 to pi/6 due to 6-fold symmetry)
        n_theta = 36
        for i in range(n_theta):
            theta = i * (math.pi / 6) / n_theta
            
            # Compute the 6 normal directions
            normals = []
            for k in range(6):
                ang = theta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            # For this theta, find optimal center c that minimizes max projection
            # This is: minimize D subject to
            #   (p_i - c) · n_k <= D for all i, k
            #   -(p_i - c) · n_k <= D for all i, k
            # Equivalently: |p_i · n_k - c · n_k| <= D for all i, k
            
            # Use scipy linprog or just solve via the fact that optimal c
            # makes the max and min projections as balanced as possible.
            # Actually, for fixed theta, the optimal D is:
            #   D = 0.5 * max_k (max_i p_i·n_k - min_i p_i·n_k)
            # and optimal c·n_k = 0.5*(max_i p_i·n_k + min_i p_i·n_k)
            
            D_val = 0.0
            c_projections = np.zeros(6)
            for k in range(6):
                nx, ny = normals[k]
                projs = positions[:, 0] * nx + positions[:, 1] * ny
                max_p = np.max(projs)
                min_p = np.min(projs)
                D_val = max(D_val, 0.5 * (max_p - min_p))
                c_projections[k] = 0.5 * (max_p + min_p)
            
            # Now recover c from c_projections
            # c · n_k = c_projections[k] for k=0..5
            # We have 6 equations, 2 unknowns. Least squares.
            A = np.array(normals)
            b = c_projections
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            if D_val < best_D:
                best_D = D_val
                best_theta = theta
                best_c = (c[0], c[1])
        
        # Refine with local optimization
        def objective(params):
            cx, cy, theta = params
            theta = ((theta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            normals = []
            for k in range(6):
                ang = theta + math.pi / 6.0 + k * math.pi / 3.0
                normals.append((math.cos(ang), math.sin(ang)))
            
            max_proj = 0.0
            for nx, ny in normals:
                projs = (positions[:, 0] - cx) * nx + (positions[:, 1] - cy) * ny
                max_proj = max(max_proj, np.max(projs), -np.min(projs))
            return max_proj
        
        # Local refinement from the grid search result
        try:
            res = minimize(objective, [best_c[0], best_c[1], best_theta], method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_D:
                best_D = res.fun
                best_c = (res.x[0], res.x[1])
                best_theta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, best_c, best_theta

    # Run force-directed packing
    positions = force_directed_pack(n_iter=3000, dt=0.02, attract_strength=0.3)
    
    # Compute minimal enclosing hexagon
    S, center, theta = compute_min_enclosing_hexagon(positions)
    
    # Try a few different initial configurations and pick the best
    best_S = S
    best_positions = positions.copy()
    best_center = center
    best_theta = theta
    
    # Alternative initial configurations
    alt_configs = []
    
    # Config 1: ring + center
    alt1 = []
    alt1.append((0, 0))
    for k in range(6):
        ang = k * math.pi / 3
        alt1.append((math.sqrt(3) * math.cos(ang), math.sqrt(3) * math.sin(ang)))
    # Add 4 more in a second ring
    for k in range(4):
        ang = k * math.pi / 2 + math.pi/4
        r = 3.0
        alt1.append((r * math.cos(ang), r * math.sin(ang)))
    alt_configs.append(np.array(alt1))
    
    # Config 2: elongated
    alt2 = []
    for i in range(11):
        x = (i - 5) * math.sqrt(3)
        y = 0.0 if i % 2 == 0 else 1.5
        alt2.append((x, y))
    alt_configs.append(np.array(alt2))
    
    # Config 3: compact cluster
    alt3 = []
    axial3 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (-2, 2), (-1, 3), (1, -2)
    ]
    for q, r in axial3:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        alt3.append((x, y))
    alt_configs.append(np.array(alt3))
    
    for alt_pos in alt_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        # Run force-directed from this start
        alt_pos = alt_pos.copy()
        alt_pos += np.random.uniform(-0.05, 0.05, (N, 2))
        
        # Quick force-directed relaxation
        dt = 0.01
        for it in range(1000):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            forces = np.zeros((N, 2))
            centroid = np.mean(alt_pos, axis=0)
            for i in range(N):
                forces[i][0] += 0.3 * (centroid[0] - alt_pos[i][0])
                forces[i][1] += 0.3 * (centroid[1] - alt_pos[i][1])
            for i in range(N):
                for j in range(i+1, N):
                    dx = alt_pos[i][0] - alt_pos[j][0]
                    dy = alt_pos[i][1] - alt_pos[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-6:
                        continue
                    d_min = math.sqrt(3)
                    if dist < d_min:
                        overlap = d_min - dist
                        fx = 50.0 * overlap * dx / dist
                        fy = 50.0 * overlap * dy / dist
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy
            alt_pos += dt * forces
            dt *= 0.999
        
        S_alt, c_alt, th_alt = compute_min_enclosing_hexagon(alt_pos)
        if S_alt < best_S:
            best_S = S_alt
            best_positions = alt_pos
            best_center = c_alt
            best_theta = th_alt
    
    # Final result
    angle_deg = math.degrees(best_theta)
    cx, cy = best_center
    
    # Apply tiny safety scale
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_positions:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    # Recompute side with scaled points
    max_proj = 0.0
    for k in range(6):
        ang = best_theta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (np.array(pts_scaled)[:, 0] - cx) * nx + (np.array(pts_scaled)[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    S_final = (max_proj + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center_out = [float(cx), float(cy)]
    side_out = float(S_final)
    rotation_out = float(angle_deg)
    
    return inner, center_out, side_out, rotation_out
