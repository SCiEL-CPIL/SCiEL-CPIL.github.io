import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions):
        """
        positions: (12, 2) array of (x, y) for unit hexagons (side=1, angle=0).
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            verts = hexagon_vertices(x, y, 1.0, 0.0)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def calculate_side_length(positions):
        """
        Calculate the exact minimum side length of the outer hexagon (centered at origin, angle 0)
        required to contain all inner hexagons.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x + y / sqrt3
        L2 = y * 2.0 / sqrt3
        L3 = -x + y / sqrt3
        L4 = -x - y / sqrt3
        L5 = -y * 2.0 / sqrt3
        L6 = x - y / sqrt3
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        return np.max(all_L)

    def objective(positions):
        """
        Objective function: 
        1. If invalid (overlapping), return a large penalty.
        2. If valid, return the exact outer side length.
        """
        positions = positions.reshape((N, 2))
        if not check_validity(positions):
            return 1000.0 + calculate_side_length(positions) # Large penalty
        return calculate_side_length(positions)

    def generate_initial_configs():
        configs = []
        
        # 1. Hexagonal lattice (compact)
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        init1 = []
        for q, r in coords:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            init1.append([x, y])
        configs.append(np.array(init1).flatten())
        
        # 2. Circular arrangement
        init2 = np.zeros(N * 2)
        for i in range(N):
            angle = 2 * np.pi * i / N
            r = 2.0
            init2[2*i] = r * np.cos(angle)
            init2[2*i+1] = r * np.sin(angle)
        configs.append(init2)
        
        # 3. Random starts (multiple)
        np.random.seed(42)
        for _ in range(5):
            init_rand = np.random.uniform(-4, 4, N * 2)
            configs.append(init_rand)
            
        return configs

    best_val = float('inf')
    best_params = None
    
    inits = generate_initial_configs()
    
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            # Use Nelder-Mead which is robust to non-smooth functions
            # and doesn't require gradients.
            res = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 500,
                    'xatol': 1e-6,
                    'fatol': 1e-6,
                    'adaptive': True
                }
            )
            
            # Verify the result is actually valid
            pos_res = res.x.reshape((N, 2))
            if check_validity(pos_res):
                val = calculate_side_length(pos_res)
                if val < best_val:
                    best_val = val
                    best_params = res.x
            else:
                # If Nelder-Mead landed in an invalid spot, try to repair or ignore
                # We might try a simple local repair by pushing apart overlapping hexagons
                # But for now, we just discard this candidate if invalid
                pass
        except Exception as e:
            pass

    # Fallback if optimization failed or returned invalid
    if best_params is None:
        # Try initial configs directly
        for init in inits:
            pos = init.reshape((N, 2))
            if check_validity(pos):
                val = calculate_side_length(pos)
                if val < best_val:
                    best_val = val
                    best_params = init
        if best_params is None:
            # Last resort: spread out
            best_params = np.array([i * 3.0 for i in range(N)] + [0.0 for i in range(N)])
            best_val = calculate_side_length(best_params.reshape((N, 2)))

    positions = best_params.reshape((N, 2))
    
    # Calculate exact outer hexagon side length
    exact_side = calculate_side_length(positions)
    
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(exact_side), 0.0

# EVOLVE-BLOCK-END
