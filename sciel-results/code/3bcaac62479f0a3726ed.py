"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    sqrt3 = np.sqrt(3)

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Vertices of regular hexagon centered at (cx,cy), side length, rotation in degrees."""
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def sat_disjoint(h1, h2):
        """SAT check for two convex hexagons. Returns True if disjoint."""
        v1 = hexagon_vertices(h1[0], h1[1], 1.0, h1[2])
        v2 = hexagon_vertices(h2[0], h2[1], 1.0, h2[2])
        
        # Quick distance check
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        if dx*dx + dy*dy > 4.5:  # > 2.12^2, definitely disjoint
            return True
        
        edges = []
        for i in range(6):
            edges.append(v1[i] - v1[(i+1) % 6])
            edges.append(v2[i] - v2[(i+1) % 6])
        
        for edge in edges:
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal /= norm
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return True
        return False

    def min_enclosing_hex_side(inner, center, outer_angle_deg):
        """Compute minimum side length of outer hexagon to contain all inner hexagons."""
        all_verts = np.vstack([hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)])
        offsets = all_verts - np.array(center)
        
        max_extent = 0.0
        outer_angle_rad = np.radians(outer_angle_deg)
        for k in range(6):
            normal_angle = outer_angle_rad + np.pi/6 + k * np.pi / 3
            normal = np.array([np.cos(normal_angle), np.sin(normal_angle)])
            extents = offsets @ normal
            max_e = extents.max()
            if max_e > max_extent:
                max_extent = max_e
        
        # For a regular hexagon with side L, the apothem is sqrt(3)/2 * L
        # The maximum projection onto any of the 6 edge normals must be <= apothem
        # So L = 2 * max_extent / sqrt(3)
        return 2.0 * max_extent / sqrt3

    def objective(params):
        """Smooth objective: outer side length + penalty for overlaps."""
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        center = (params[3*N], params[3*N+1])
        outer_angle = params[3*N+2]
        
        side = min_enclosing_hex_side(inner, center, outer_angle)
        
        # Penalty for overlapping pairs
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not sat_disjoint(inner[i], inner[j]):
                    # Compute overlap depth as penalty
                    v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                    v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                    # Use center distance as proxy for overlap severity
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    # If dist < sqrt(3) ~ 1.732, hexagons overlap
                    if dist < sqrt3:
                        penalty += (sqrt3 - dist) ** 2
        
        return side + 100.0 * penalty

    def initial_guess():
        """Hexagonal lattice initial configuration."""
        params = np.zeros(3*N + 3)
        positions = []
        # Center hexagon
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for i in range(6):
            theta = i * np.pi / 3
            positions.append((sqrt3 * np.cos(theta), 1.5 * np.sin(theta), 0.0))
        # Ring 2: 5 hexagons (partial ring)
        for i in range(5):
            theta = i * 2*np.pi / 5 + np.pi/10
            positions.append((2*sqrt3 * np.cos(theta), 3.0 * np.sin(theta), 0.0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0
        return params

    best_params = initial_guess()
    best_score = objective(best_params)

    # Multi-start L-BFGS-B optimization
    n_starts = 0
    max_starts = 15
    while time.time() < deadline - 15 and n_starts < max_starts:
        n_starts += 1
        if n_starts == 1:
            params0 = best_params.copy()
        else:
            # Perturbed initial guess
            params0 = initial_guess().copy()
            for i in range(3*N + 3):
                if random.random() < 0.3:
                    params0[i] += random.gauss(0, 0.5)
        
        try:
            result = minimize(objective, params0, method='L-BFGS-B',
                            options={'maxiter': 200, 'gtol': 1e-6, 'ftol': 1e-8})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass

    # Extract best result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    outer_angle = float(best_params[3*N+2])
    side = min_enclosing_hex_side(inner, center, outer_angle)

    # Final validation
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not sat_disjoint(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: spread hexagons far apart
        inner = [[float(4.0 * k), 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        center = (0.0, 0.0)
        side = 25.0
        outer_angle = 0.0

    return inner, center, side, outer_angle
# EVOLVE-BLOCK-END
