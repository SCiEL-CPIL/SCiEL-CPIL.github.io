import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)
MIN_DIST = SQRT3 - 1e-4  # minimum center distance for disjoint hexagons

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin for 150s limit
    
    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side for given configuration."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi/6 + k * math.pi/3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / SQRT3

    def check_disjoint(centers, angles, min_dist=MIN_DIST):
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

    def polish_outer(centers, angles, initial_ocx, initial_ocy, initial_otheta_deg):
        """High-precision polish of outer hexagon parameters."""
        best_side = compute_outer_side(centers, angles, initial_ocx, initial_ocy, initial_otheta_deg)
        best_ocx, best_ocy, best_otheta = initial_ocx, initial_ocy, initial_otheta_deg
        
        # Multi-scale search
        scales = [0.5, 0.1, 0.01, 0.001, 0.0001]
        for scale in scales:
            improved = True
            while improved and time.time() < deadline - 20:
                improved = False
                for dx in [-scale, 0, scale]:
                    for dy in [-scale, 0, scale]:
                        for da_deg in [-scale*10, 0, scale*10]:
                            otheta = (best_otheta + da_deg) % 60
                            side = compute_outer_side(centers, angles, best_ocx + dx, best_ocy + dy, otheta)
                            if side < best_side - 1e-12:
                                best_side = side
                                best_ocx += dx
                                best_ocy += dy
                                best_otheta = otheta
                                improved = True
        
        return best_ocx, best_ocy, best_otheta, best_side

    def polish_positions(centers, angles, ocx, ocy, otheta_deg, max_iters=30):
        """Polish positions using Nelder-Mead with tight tolerances."""
        n = len(centers)
        x0 = []
        for i in range(n):
            x0.extend([centers[i][0], centers[i][1], angles[i]])
        
        def objective(x):
            if time.time() > deadline - 25:
                return 1e10
            c = [[x[3*i], x[3*i+1]] for i in range(n)]
            a = [x[3*i+2] for i in range(n)]
            return compute_outer_side(c, a, ocx, ocy, otheta_deg)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': max_iters, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < 1e9:
                c = [[result.x[3*i], result.x[3*i+1]] for i in range(n)]
                a = [result.x[3*i+2] for i in range(n)]
                return c, a
        except Exception:
            pass
        return centers, angles

    def optimize_full(centers, angles, max_iters=50):
        """Full optimization: positions + outer hexagon."""
        n = len(centers)
        
        # First polish outer hexagon
        ocx, ocy, otheta_deg, side = polish_outer(centers, angles, 0.0, 0.0, 0.0)
        
        # Then polish positions
        centers2, angles2 = polish_positions(centers, angles, ocx, ocy, otheta_deg)
        
        # Polish outer again
        ocx, ocy, otheta_deg, side = polish_outer(centers2, angles2, ocx, ocy, otheta_deg)
        
        # Check validity
        if not check_disjoint(centers2, angles2):
            return None
        
        # Verify containment
        R = side * SQRT3 / 2
        otheta = math.radians(otheta_deg)
        for i in range(n):
            cx, cy = centers2[i]
            angle = math.radians(angles2[i])
            for vi in range(6):
                vert_angle = angle + vi * math.pi / 3
                vx = cx + math.cos(vert_angle)
                vy = cy + math.sin(vert_angle)
                for k in range(6):
                    normal_angle = otheta + math.pi/6 + k * math.pi/3
                    proj = (vx - ocx) * math.cos(normal_angle) + (vy - ocy) * math.sin(normal_angle)
                    if proj > R + 1e-6:
                        return None
        
        return centers2, angles2, ocx, ocy, otheta_deg, side

    # Generate initial configurations using honeycomb lattice
    d = SQRT3 + 0.001  # minimum center distance
    u = np.array([d, 0.0])
    v = np.array([d/2.0, d*SQRT3/2.0])
    
    # Ring 1: 6 neighbors of origin
    ring1_vectors = [u, v, v-u, -u, -v, u-v]
    
    # Generate ring 2 candidates
    basis = [u, v, v-u, -u, -v, u-v]
    ring2_candidates = []
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            # Check not origin
            if np.linalg.norm(p2) < 0.01:
                continue
            # Check not in ring 1
            is_r1 = False
            for p1j in ring1_vectors:
                if np.linalg.norm(p2 - p1j) < 0.01:
                    is_r1 = True
                    break
            if is_r1:
                continue
            # Check not duplicate
            duplicate = False
            for pc in ring2_candidates:
                if np.linalg.norm(p2 - pc) < 0.01:
                    duplicate = True
                    break
            if not duplicate:
                ring2_candidates.append(p2)
    
    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    
    best_result = None
    best_side = float('inf')
    
    # Try different ring 2 selections
    num_configs_to_try = min(8, len(ring2_candidates))
    
    for config_idx in range(num_configs_to_try):
        if time.time() > deadline - 40:
            break
        
        # Select ring 2 vectors
        if config_idx == 0:
            ring2_sel = ring2_candidates[:5]
        elif config_idx == 1:
            ring2_sel = ring2_candidates[5:10]
        elif config_idx == 2:
            ring2_sel = [ring2_candidates[i] for i in [0, 2, 4, 6, 8]]
        elif config_idx == 3:
            ring2_sel = [ring2_candidates[i] for i in [1, 3, 5, 7, 9]]
        else:
            random.seed(config_idx * 12345)
            idxs = list(range(len(ring2_candidates)))
            random.shuffle(idxs)
            ring2_sel = [ring2_candidates[i] for i in idxs[:5]]
        
        centers = [[0.0, 0.0]]
        for vec in ring1_vectors:
            centers.append([vec[0], vec[1]])
        for vec in ring2_sel:
            centers.append([vec[0], vec[1]])
        
        angles = [30.0] * 12
        
        # Check if valid
        if not check_disjoint(centers, angles):
            continue
        
        # Optimize
        result = optimize_full(centers, angles)
        
        if result is not None:
            c, a, ocx, ocy, otheta_deg, side = result
            if side < best_side:
                best_side = side
                best_result = (c, a, ocx, ocy, otheta_deg)
    
    # Also try some random initial configurations
    for trial in range(5):
        if time.time() > deadline - 30:
            break
        
        random.seed(trial * 777)
        centers = [[0.0, 0.0]]
        angles = [30.0]
        
        for _ in range(N-1):
            # Try to place a new hexagon
            placed = False
            for attempt in range(50):
                # Random position near existing hexagons
                if centers:
                    base = random.choice(centers)
                    angle = random.uniform(0, 2*math.pi)
                    dist = d + random.uniform(0, 1.0)
                    new_x = base[0] + dist * math.cos(angle)
                    new_y = base[1] + dist * math.sin(angle)
                else:
                    new_x = random.uniform(-3, 3)
                    new_y = random.uniform(-3, 3)
                
                new_center = [new_x, new_y]
                new_angles = angles + [30.0]
                new_centers = centers + [new_center]
                
                if check_disjoint(new_centers, new_angles):
                    centers = new_centers
                    angles = new_angles
                    placed = True
                    break
            
            if not placed:
                break
        
        if len(centers) == N and check_disjoint(centers, angles):
            result = optimize_full(centers, angles)
            if result is not None:
                c, a, ocx, ocy, otheta_deg, side = result
                if side < best_side:
                    best_side = side
                    best_result = (c, a, ocx, ocy, otheta_deg)
    
    # Final fallback: linear arrangement
    if best_result is None:
        centers = [[i * d, 0.0] for i in range(N)]
        angles = [30.0] * N
        ocx, ocy, otheta_deg, side = polish_outer(centers, angles, 0.0, 0.0, 0.0)
        best_result = (centers, angles, ocx, ocy, otheta_deg)
        best_side = side
    
    # Final polish on best result
    if best_result is not None and time.time() < deadline - 15:
        c, a, ocx, ocy, otheta_deg = best_result
        # One more polish pass
        ocx, ocy, otheta_deg, side = polish_outer(c, a, ocx, ocy, otheta_deg)
        if side < best_side:
            best_side = side
            best_result = (c, a, ocx, ocy, otheta_deg)
    
    # Extract final result
    centers, angles, ocx, ocy, otheta_deg = best_result
    
    # Translate so outer hexagon is centered at origin
    # The outer hexagon is centered at (ocx, ocy), so we need to shift all centers
    final_centers = [[cx - ocx, cy - ocy] for cx, cy in centers]
    
    # Compute final side length
    final_side = compute_outer_side(final_centers, angles, 0.0, 0.0, otheta_deg)
    
    # Verify validity one more time
    if not check_disjoint(final_centers, angles):
        # Fallback to simple arrangement
        final_centers = [[i * d, 0.0] for i in range(N)]
        angles = [30.0] * N
        final_side = compute_outer_side(final_centers, angles, 0.0, 0.0, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_centers[i][0], final_centers[i][1], angles[i]])
    
    return inner, (0.0, 0.0), final_side, otheta_deg
