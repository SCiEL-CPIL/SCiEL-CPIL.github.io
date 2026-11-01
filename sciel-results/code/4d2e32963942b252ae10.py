"""Improved packing: differential evolution + local polish for 12 unit hexagons."""
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        verts.append([cx + side * math.cos(theta), cy + side * math.sin(theta)])
    return np.array(verts)

def _sat_disjoint(v1, v2, tol=1e-9):
    """SAT test: do two convex hexagons overlap? Returns True if disjoint."""
    all_edges = []
    for i in range(6):
        all_edges.append(v1[i] - v1[(i+1) % 6])
        all_edges.append(v2[i] - v2[(i+1) % 6])
    for edge in all_edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return True
    return False

def _hexagons_disjoint(h1, h2):
    """Check if two inner hexagons are disjoint."""
    v1 = _hex_verts(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_verts(h2[0], h2[1], 1.0, h2[2])
    return _sat_disjoint(v1, v2)

def _outer_hex_contains(inner_verts, outer_center, outer_side, outer_angle, tol=1e-9):
    """Check if inner hexagon is inside outer hexagon."""
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    all_edges = []
    for i in range(6):
        all_edges.append(inner_verts[i] - inner_verts[(i+1) % 6])
        all_edges.append(outer_verts[i] - outer_verts[(i+1) % 6])
    for edge in all_edges:
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        inner_proj = inner_verts @ normal
        outer_proj = outer_verts @ normal
        if inner_proj.max() > outer_proj.min() - tol or inner_proj.min() < outer_proj.max() + tol:
            # Check if inner is inside outer on this axis
            if inner_proj.max() > outer_proj.max() - tol or inner_proj.min() < outer_proj.min() + tol:
                return False
    return True

def _point_in_convex(p, verts):
    """Check if point p is inside convex polygon verts."""
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i+1) % n]
        cross = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
        if cross < -1e-9:
            return False
    return True

def _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Find minimum outer side such that all inner hexagons fit inside."""
    # For each inner hexagon, compute the maximum projection onto each of the 6 outer axes
    outer_verts_unit = _hex_verts(0, 0, 1.0, outer_angle)
    # The 6 edge normals of the outer hexagon
    normals = []
    for i in range(6):
        edge = outer_verts_unit[i] - outer_verts_unit[(i+1) % 6]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
            normals.append(normal)
    
    # For a regular hexagon with side s, the apothem is s * sqrt(3)/2
    # The projection of the hexagon onto any edge normal ranges from -s*sqrt(3)/2 to s*sqrt(3)/2
    # Actually, for a regular hexagon centered at origin with side s:
    # The vertices are at distance s from center, and the apothem is s*sqrt(3)/2
    
    # For each inner hexagon, find the max extent in each normal direction
    required = 0.0
    for i in range(N):
        iv = _hex_verts(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        for normal in normals:
            proj = iv @ normal
            proj_center = np.array(outer_center) @ normal
            extent = max(proj.max() - proj_center, proj_center - proj.min())
            # The outer hexagon has half-width = side * sqrt(3)/2 in the normal direction
            # So we need side * sqrt(3)/2 >= extent
            required = max(required, extent / (math.sqrt(3) / 2))
    
    return required

def _objective(params):
    """Objective function for optimization."""
    inner_positions = []
    inner_angles = []
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        angle = params[3*i+2]
        inner_positions.append((x, y))
        inner_angles.append(angle)
    
    outer_center = (params[3*N], params[3*N+1])
    outer_angle = params[3*N+2]
    
    # Check disjointness
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint((inner_positions[i][0], inner_positions[i][1], inner_angles[i]),
                                       (inner_positions[j][0], inner_positions[j][1], inner_angles[j])):
                # Overlap penalty
                v1 = _hex_verts(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
                v2 = _hex_verts(inner_positions[j][0], inner_positions[j][1], 1.0, inner_angles[j])
                # Approximate overlap by min distance
                min_dist = np.inf
                for vi in v1:
                    for vj in v2:
                        d = np.linalg.norm(vi - vj)
                        min_dist = min(min_dist, d)
                # Also check edge-edge
                return 1000.0 + min_dist * 100
    
    # Compute required outer side
    side = _min_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
    
    # Verify all hexagons are actually inside
    outer_verts = _hex_verts(outer_center[0], outer_center[1], side, outer_angle)
    for i in range(N):
        iv = _hex_verts(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        if not _outer_hex_contains(iv, outer_center, side, outer_angle):
            return side + 100.0
    
    return side

def _local_polish(params, deadline):
    """Local refinement using Nelder-Mead."""
    best_params = params.copy()
    best_score = _objective(best_params)
    
    # Try several local optimizations
    for attempt in range(3):
        if time.time() > deadline - 5:
            break
        try:
            result = minimize(_objective, best_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    return best_params, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Initial guess: hexagonal packing pattern
    def initial_guess():
        params = np.zeros(3*N + 3)
        # Center
        positions = [(0, 0, 0)]
        # Ring 1: 6 hexagons
        for i in range(6):
            theta = i * math.pi / 3
            positions.append((2*math.cos(theta), 2*math.sin(theta), 0))
        # Ring 2: 5 more
        for i in range(5):
            theta = i * 2*math.pi / 5 + math.pi/5
            positions.append((4*math.cos(theta), 4*math.sin(theta), 0))
        
        for i in range(N):
            params[3*i] = positions[i][0]
            params[3*i+1] = positions[i][1]
            params[3*i+2] = positions[i][2]
        
        params[3*N] = 0.0
        params[3*N+1] = 0.0
        params[3*N+2] = 0.0  # Outer angle
        return params
    
    best_params = initial_guess()
    best_score = _objective(best_params)
    
    # Differential evolution
    bounds = [(-10, 10)] * (3*N) + [(-5, 5), (-5, 5), (-180, 180)]
    
    if time.time() < deadline - 60:
        try:
            de_result = differential_evolution(
                _objective, bounds,
                maxiter=50, popsize=15, tol=1e-4,
                seed=42, polish=False,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    # Local polish
    if time.time() < deadline - 10:
        best_params, best_score = _local_polish(best_params, deadline)
    
    # Extract result
    inner = []
    for i in range(N):
        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
    
    center = (float(best_params[3*N]), float(best_params[3*N+1]))
    angle = float(best_params[3*N+2])
    
    # Compute final side
    side = _min_outer_side([(inner[i][0], inner[i][1]) for i in range(N)],
                          [inner[i][2] for i in range(N)], center, angle)
    
    # Final validation
    outer_verts = _hex_verts(center[0], center[1], side, angle)
    valid = True
    for i in range(N):
        iv = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
        if not _outer_hex_contains(iv, center, side, angle):
            valid = False
            break
        for j in range(i+1, N):
            if not _hexagons_disjoint(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: larger container
        side = max(side, 8.0)
        # Re-check
        outer_verts = _hex_verts(center[0], center[1], side, angle)
        valid = True
        for i in range(N):
            iv = _hex_verts(inner[i][0], inner[i][1], 1.0, inner[i][2])
            if not _outer_hex_contains(iv, center, side, angle):
                valid = False
                break
        if not valid:
            # Very large fallback
            inner = [[0.0, 0.0, 0.0]] * N
            center = (0.0, 0.0)
            side = 100.0
            angle = 0.0
    
    return inner, center, side, angle
