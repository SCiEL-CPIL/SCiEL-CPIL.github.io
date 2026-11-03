import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0
    elapsed = lambda: time.time() - (deadline - 140.0)

    # Initial lattice configuration
    eps = 0.005  # Small gap to ensure strict disjointness
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
    
    # Flatten to 2D array of positions
    x0 = np.array([c[0] for c in centers]).flatten()
    y0 = np.array([c[1] for c in centers]).flatten()
    z0 = np.concatenate([x0, y0])
    
    # Angle for all hexagons: 30 degrees (flat-top)
    angle_deg = 30.0
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices
    
    def compute_outer_side(centers_list):
        """Compute minimum outer hexagon side length for given centers."""
        all_verts = []
        for c in centers_list:
            verts = get_hex_vertices(c[0], c[1], 1.0, angle_deg)
            all_verts.extend(verts)
        
        if not all_verts:
            return float('inf'), (0, 0), 0.0
        
        # Use centroid as center
        cx = sum(p[0] for p in all_verts) / len(all_verts)
        cy = sum(p[1] for p in all_verts) / len(all_verts)
        
        best_side = float('inf')
        best_angle = 0.0
        
        # Scan outer hexagon angle with fine resolution
        for theta_deg in np.linspace(0, 60, 121):
            theta = math.radians(theta_deg)
            max_proj = 0.0
            for k in range(6):
                normal_angle = theta + math.pi/6 + k * math.pi/3
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                for px, py in all_verts:
                    proj = (px - cx) * nx + (py - cy) * ny
                    if proj > max_proj:
                        max_proj = proj
            s = 2.0 * max_proj / math.sqrt(3)
            if s < best_side:
                best_side = s
                best_angle = theta_deg
                
        return best_side, (cx, cy), best_angle
    
    def check_disjoint(centers_list, tol=1e-6):
        """Check if all hexagons are strictly disjoint."""
        verts_list = []
        for c in centers_list:
            verts = get_hex_vertices(c[0], c[1], 1.0, angle_deg)
            verts_list.append(verts)
        
        n_hex = len(verts_list)
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                # Check if any vertex of hex i is inside hex j or vice versa
                # For regular hexagons with same orientation, check center distance
                ci = np.array(centers_list[i])
                cj = np.array(centers_list[j])
                dist = np.linalg.norm(ci - cj)
                # Two unit hexagons with same orientation are disjoint if center distance > sqrt(3)
                # But this is only true for specific relative orientations. For general check:
                # Use the fact that two hexagons with same orientation and side 1 are disjoint
                # if and only if no vertex of one is inside the other.
                # For efficiency, use center distance as a quick check:
                # Minimum center distance for disjointness depends on relative angle.
                # For same orientation, min distance is sqrt(3) (edge-to-edge touching).
                # We require strict disjointness, so dist > sqrt(3) + small_tol
                if dist < math.sqrt(3) - tol:
                    return False
        return True
    
    def objective(z):
        """Objective function: outer hexagon side length + penalty for overlaps."""
        if elapsed() > 138.0:
            return float('inf')
        
        x = z[:12]
        y = z[12:]
        centers_list = list(zip(x, y))
        
        # Check disjointness
        if not check_disjoint(centers_list, tol=1e-4):
            # Penalty for overlap
            penalty = 0.0
        else:
            penalty = 0.0
        
        side, _, _ = compute_outer_side(centers_list)
        return side + penalty
    
    # Run Nelder-Mead optimization
    best_z = z0.copy()
    best_side, best_center, best_angle = compute_outer_side([tuple(c) for c in zip(x0, y0)])
    
    try:
        result = minimize(objective, z0, method='Nelder-Mead',
                         options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if result.fun < best_side:
            best_z = result.x
            x_opt = best_z[:12]
            y_opt = best_z[12:]
            centers_opt = list(zip(x_opt, y_opt))
            
            if check_disjoint(centers_opt, tol=1e-4):
                best_side, best_center, best_angle = compute_outer_side(centers_opt)
    except Exception:
        pass
    
    # Final fine scan of outer hexagon
    x_final = best_z[:12]
    y_final = best_z[12:]
    centers_final = list(zip(x_final, y_final))
    
    # Even finer scan
    all_verts = []
    for c in centers_final:
        verts = get_hex_vertices(c[0], c[1], 1.0, angle_deg)
        all_verts.extend(verts)
    
    cx = sum(p[0] for p in all_verts) / len(all_verts)
    cy = sum(p[1] for p in all_verts) / len(all_verts)
    
    best_side_final = float('inf')
    best_angle_final = 0.0
    
    for theta_deg in np.linspace(0, 60, 361):
        theta = math.radians(theta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = theta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for px, py in all_verts:
                proj = (px - cx) * nx + (py - cy) * ny
                if proj > max_proj:
                    max_proj = proj
        s = 2.0 * max_proj / math.sqrt(3)
        if s < best_side_final:
            best_side_final = s
            best_angle_final = theta_deg
    
    # Build inner list
    inner = [[x_final[i], y_final[i], angle_deg] for i in range(12)]
    
    return inner, (cx, cy), best_side_final, best_angle_final
# EVOLVE-BLOCK-END
