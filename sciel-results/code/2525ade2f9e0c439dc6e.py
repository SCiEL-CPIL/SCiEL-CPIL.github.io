import math
import time
import random
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def polygons_disjoint_sat(poly1, poly2):
        """
        SAT-based check for disjointness.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid_config(positions, angles, eps=0.0):
        """Check if all hexagons are disjoint."""
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not polygons_disjoint_sat(polys[i], polys[j]):
                    return False
        return True

    def local_refine(positions, angles, max_iters=200, step=0.05):
        """
        Local refinement using coordinate descent.
        Tries to reduce the outer side length by moving one hexagon at a time.
        """
        if not is_valid_config(positions, angles):
            return None, None, 1e9

        current_side = calc_min_outer_side(positions, angles, 0.0)
        current_positions = positions.copy()
        current_angles = angles.copy()
        best_positions = current_positions.copy()
        best_angles = current_angles.copy()
        best_side = current_side

        # Directions to try for movement: 8 directions + stay
        hex_dirs = []
        for k in range(8):
            ang = math.radians(45 * k)
            hex_dirs.append([math.cos(ang), math.sin(ang)])
        hex_dirs.append([0, 0])

        # Rotation steps
        rot_steps = [0, 0.05, -0.05, 0.1, -0.1]

        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                # Try moving position i
                for dx, dy in hex_dirs:
                    new_pos = current_positions.copy()
                    new_pos[i][0] += dx * step
                    new_pos[i][1] += dy * step

                    if is_valid_config(new_pos, current_angles):
                        new_side = calc_min_outer_side(new_pos, current_angles, 0.0)
                        if new_side < current_side - 1e-5:
                            current_positions = new_pos
                            current_side = new_side
                            improved = True
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = current_positions.copy()
                                best_angles = current_angles.copy()

                # Try rotating hexagon i
                for da in rot_steps:
                    new_ang = current_angles.copy()
                    new_ang[i] += da

                    if is_valid_config(current_positions, new_ang):
                        new_side = calc_min_outer_side(current_positions, new_ang, 0.0)
                        if new_side < current_side - 1e-5:
                            current_angles = new_ang
                            current_side = new_side
                            improved = True
                            if new_side < best_side:
                                best_side = new_side
                                best_positions = current_positions.copy()
                                best_angles = current_angles.copy()

            if not improved:
                # Try smaller step if stuck
                step *= 0.8
                if step < 0.001:
                    break

            # Random restart within local neighborhood to escape local minima
            if it % 50 == 0 and it > 0:
                if is_valid_config(best_positions, best_angles):
                    current_positions = best_positions + np.random.normal(0, 0.1, (N, 2))
                    current_angles = best_angles + np.random.normal(0, 0.05, N)
                    if is_valid_config(current_positions, current_angles):
                        current_side = calc_min_outer_side(current_positions, current_angles, 0.0)
                    else:
                        current_positions = best_positions
                        current_angles = best_angles
                        current_side = best_side

        return best_positions, best_angles, best_side

    # Generate candidate topologies
    sqrt3 = math.sqrt(3)

    # 1. Honeycomb cluster (standard)
    # Center + Ring 1 (6) + Ring 2 (5 selected) = 12
    # We use a lattice approach: generate a dense hex grid and pick 12 closest to origin
    lattice_points = []
    for q in range(-5, 6):
        for r in range(-5, 6):
            x = 1.5 * q
            y = sqrt3 * (r + q/2.0)
            lattice_points.append((x, y))

    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_honeycomb = np.array(lattice_points[:12])
    init_honeycomb -= np.mean(init_honeycomb, axis=0)
    init_angles = np.zeros(N)

    candidates = [
        ("Honeycomb", init_honeycomb, init_angles),
    ]

    # 2. Rectangular grid 3x4
    rect_pos = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * 1.5
            y = (j - 1.5) * sqrt3
            rect_pos.append((x, y))
    rect_pos = np.array(rect_pos)
    rect_pos -= np.mean(rect_pos, axis=0)
    candidates.append(("Rect3x4", rect_pos, np.zeros(N)))

    # 3. Perturbed Honeycomb
    np.random.seed(42)
    perturbed = init_honeycomb + np.random.normal(0, 0.2, (N, 2))
    candidates.append(("PerturbedHoneycomb", perturbed, np.zeros(N)))

    # 4. Circular arrangement
    circ_pos = []
    for i in range(N):
        ang = 2 * math.pi * i / N
        r = 3.0
        circ_pos.append((r * math.cos(ang), r * math.sin(ang)))
    circ_pos = np.array(circ_pos)
    circ_pos -= np.mean(circ_pos, axis=0)
    candidates.append(("Circular", circ_pos, np.zeros(N)))

    best_score = 1e9
    best_positions = None
    best_angles = None

    for name, pos, ang in candidates:
        if time.time() > deadline - 10:
            break

        # Ensure initial validity
        if not is_valid_config(pos, ang):
            continue

        res_pos, res_ang, res_score = local_refine(pos, ang, max_iters=150, step=0.1)

        if res_pos is not None and res_score < best_score:
            best_score = res_score
            best_positions = res_pos
            best_angles = res_ang

        # Also try a finer refinement on the best result so far
        if best_positions is not None and time.time() < deadline - 20:
             res_pos2, res_ang2, res_score2 = local_refine(best_positions, best_angles, max_iters=200, step=0.02)
             if res_pos2 is not None and res_score2 < best_score:
                 best_score = res_score2
                 best_positions = res_pos2
                 best_angles = res_ang2

    # If no valid config found, fallback to a very simple valid config
    if best_positions is None:
        best_positions = np.array([[math.cos(i*2*math.pi/N)*5, math.sin(i*2*math.pi/N)*5] for i in range(N)])
        best_angles = np.zeros(N)
        best_score = calc_min_outer_side(best_positions, best_angles, 0.0)

    final_positions = best_positions
    final_angles = best_angles
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])

    return inner, (0.0, 0.0), final_side, 0.0
