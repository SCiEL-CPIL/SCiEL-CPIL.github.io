import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.001):
        """Check if all hexagons are disjoint (center distance >= sqrt(3))."""
        n = len(centers)
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimum outer hexagon side for given configuration.
        The outer hexagon must contain all inner hexagon vertices.
        For each of the 6 edge normals, we need max projection >= 0 and min projection <= 0
        relative to center, OR equivalently, the max absolute projection in each normal direction.
        """
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = angles[i]
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    abs_proj = abs(proj)
                    if abs_proj > max_proj:
                        max_proj = abs_proj
        return 2.0 * max_proj / math.sqrt(3)

    def optimize_outer(centers, angles, ocx, ocy, otheta, max_iters=50):
        """Optimize outer hexagon center and angle to minimize side."""
        best_side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
        
        # Multi-scale optimization
        scales = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005]
        for scale in scales:
            if time.time() > deadline - 20:
                break
            improved = True
            while improved and (time.time() < deadline - 20):
                improved = False
                step = scale * 0.5
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        for da_deg in [-5, 0, 5]:
                            da = math.radians(da_deg)
                            side = compute_outer_side(centers, angles, best_ocx+dx, best_ocy+dy, best_otheta+da)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta += da
                                improved = True
        return best_side, best_ocx, best_ocy, best_otheta

    def optimize_config(centers, angles, max_iters=100, step_size=0.05):
        """Local optimization to minimize outer hexagon side."""
        n = len(centers)
        best_side, best_ocx, best_ocy, best_otheta = optimize_outer(centers, angles, 0, 0, 0)
        
        # Now optimize inner hexagon positions and angles
        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            improved = False
            for i in range(n):
                # Try position perturbations
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_centers = [c.copy() for c in centers]
                        new_centers[i][0] += dx
                        new_centers[i][1] += dy
                        if check_disjoint(new_centers, angles):
                            side, ocx, ocy, otheta = optimize_outer(new_centers, angles, best_ocx, best_ocy, best_otheta, max_iters=10)
                            if side < best_side - 1e-10:
                                best_side = side
                                best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
                                centers = new_centers
                                improved = True
                # Try angle perturbations
                for da_deg in [-10, 0, 10]:
                    if da_deg == 0:
                        continue
                    new_angles = angles.copy()
                    new_angles[i] += da_deg
                    if check_disjoint(centers, new_angles):
                        side, ocx, ocy, otheta = optimize_outer(centers, new_angles, best_ocx, best_ocy, best_otheta, max_iters=10)
                        if side < best_side - 1e-10:
                            best_side = side
                            best_ocx, best_ocy, best_otheta = ocx, ocy, otheta
                            angles = new_angles
                            improved = True
            if not improved:
                break
        
        return centers, angles, best_side, best_ocx, best_ocy, best_otheta

    # Lattice vectors for hexagonal packing
    eps = 0.01
    d = math.sqrt(3) + eps
    
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*math.sqrt(3)/2.0])
    
    # Ring 0
    centers = [np.array([0.0, 0.0])]
    
    # Ring 1
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())
        
    # Ring 2 Candidates
    ring2_candidates = []
    basis = [u, v, v-u, -u, -v, u-v]
    
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)
                    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    for i in range(5):
        centers.append(ring2_candidates[i].copy())
        
    assert len(centers) == 12
    
    # Convert to list format
    centers_list = [[c[0], c[1]] for c in centers]
    angles_list = [30.0] * 12
    
    # Run local optimization
    opt_centers, opt_angles, opt_side, opt_ocx, opt_ocy, opt_otheta = optimize_config(centers_list, angles_list)
    
    # Final high-precision polish: fine-tune outer hexagon parameters with finer grid
    best_outer_side = opt_side
    best_ocx, best_ocy, best_otheta_deg = opt_ocx, opt_ocy, math.degrees(opt_otheta)
    
    # Fine grid search around the optimized outer parameters
    for otheta_deg in np.linspace(best_otheta_deg - 15, best_otheta_deg + 15, 61):
        otheta = math.radians(otheta_deg)
        for ocx in np.linspace(best_ocx - 0.3, best_ocx + 0.3, 21):
            for ocy in np.linspace(best_ocy - 0.3, best_ocy + 0.3, 21):
                side = compute_outer_side(opt_centers, opt_angles, ocx, ocy, otheta)
                if side < best_outer_side - 1e-12:
                    best_outer_side = side
                    best_ocx, best_ocy, best_otheta_deg = ocx, ocy, otheta_deg
    
    # Use optimized configuration
    inner = [[c[0], c[1], a] for c, a in zip(opt_centers, opt_angles)]
    
    outer_center = (best_ocx, best_ocy)
    outer_side = best_outer_side
    outer_angle = best_otheta_deg
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
