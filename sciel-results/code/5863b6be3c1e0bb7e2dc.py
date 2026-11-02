import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 12

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

    def required_outer_side(positions):
        """
        Compute the exact side length of the smallest regular hexagon (angle 0, centered at origin)
        that contains all unit hexagons at given positions.
        """
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0)
        verts_flat = verts.reshape(-1, 2)
        
        x = verts_flat[:, 0]
        y = verts_flat[:, 1]
        sqrt3 = np.sqrt(3.0)
        
        L1 = x * (sqrt3/2) + y * (1/2)
        L2 = y
        L3 = x * (-sqrt3/2) + y * (1/2)
        L4 = x * (-sqrt3/2) + y * (-1/2)
        L5 = -y
        L6 = x * (sqrt3/2) + y * (-1/2)
        
        all_L = np.stack([L1, L2, L3, L4, L5, L6], axis=1)
        max_proj = np.max(all_L)
        return (2.0 / sqrt3) * max_proj

    def check_validity(positions, tol=1e-4):
        """
        Check if hexagons are disjoint using SAT.
        """
        verts_list = [hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Quick bounding box check
                if (np.max(v1[:,0]) + tol < np.min(v2[:,0]) or 
                    np.max(v2[:,0]) + tol < np.min(v1[:,0]) or
                    np.max(v1[:,1]) + tol < np.min(v2[:,1]) or
                    np.max(v2[:,1]) + tol < np.min(v1[:,1])):
                    continue

                # SAT check
                for k in range(6):
                    edge = v1[(k+1)%6] - v1[k]
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm < 1e-10: continue
                    normal /= norm
                    
                    proj1 = np.dot(v1, normal)
                    proj2 = np.dot(v2, normal)
                    
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    
                    if max1 + tol < min2 or max2 + tol < min1:
                        break
                else:
                    return False
        return True

    def generate_symmetric_positions(r1, r2, theta2):
        """
        Generate 12 hexagon positions using 6-fold symmetry:
        - 1 center at (0, 0)
        - 6 ring-1 at radius r1, angles 0, 60, 120, ...
        - 5 ring-2 at radius r2, angles theta2, theta2+72, ... (5 points, not 6-fold symmetric)
        
        Actually, for 12 hexagons with 6-fold symmetry, we need:
        - 1 center
        - 6 ring-1
        - 5 ring-2 (but 5 doesn't fit 6-fold symmetry)
        
        Alternative: 1 center + 6 ring-1 + 5 ring-2 is NOT 6-fold symmetric.
        Let's use: 1 center + 6 ring-1 + 5 ring-2 where ring-2 has 5 points at angles that break 6-fold symmetry slightly.
        
        Actually, a better approach: 12 hexagons can be arranged as:
        - 1 center
        - 6 ring-1 (6-fold symmetric)
        - 5 ring-2 (not 6-fold symmetric, so we can't enforce full symmetry)
        
        Let's try a different structure:
        - 1 center
        - 6 ring-1 at radius r1
        - 5 ring-2 at radius r2, with one gap
        
        Or: 12 hexagons in a 3-ring hexagonal lattice pattern:
        Ring 0: 1 hexagon (center)
        Ring 1: 6 hexagons
        Ring 2: 12 hexagons (but we only have 5 more)
        
        So: 1 + 6 + 5 = 12. The 5 ring-2 hexagons can be placed at 5 of the 6 ring-2 positions,
        leaving one gap. This breaks the 6-fold symmetry, but we can still use a reduced parameter space.
        
        Parameters:
        - r1: radius of ring-1
        - r2: radius of ring-2
        - theta2: angular offset of ring-2 (one of the 6 ring-2 positions is missing)
        - gap_index: which ring-2 position is missing (0-5)
        
        But gap_index is discrete. Let's treat it as a continuous parameter that we round.
        """
        positions = np.zeros((N, 2))
        
        # Center
        positions[0, 0] = 0.0
        positions[0, 1] = 0.0
        
        # Ring 1: 6 hexagons at radius r1, angles 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = np.radians(60 * k)
            positions[1 + k, 0] = r1 * np.cos(angle)
            positions[1 + k, 1] = r1 * np.sin(angle)
        
        # Ring 2: 5 hexagons at radius r2, angles theta2 + 72*k for k=0..4
        # This is 5-fold symmetric, not 6-fold.
        for k in range(5):
            angle = np.radians(theta2 + 72 * k)
            positions[7 + k, 0] = r2 * np.cos(angle)
            positions[7 + k, 1] = r2 * np.sin(angle)
        
        return positions

    def objective(params):
        """
        Objective function for optimization.
        Returns the outer hexagon side length, or a large penalty if invalid.
        """
        r1, r2, theta2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 100.0
        
        positions = generate_symmetric_positions(r1, r2, theta2)
        
        if not check_validity(positions):
            # Penalty for invalid configurations
            return 100.0 + required_outer_side(positions)
        
        return required_outer_side(positions)

    # Initial best
    best_positions = None
    best_side = float('inf')
    
    # Try multiple initial points for the optimizer
    initial_points = [
        (2.0, 4.0, 0.0),
        (2.2, 4.2, 30.0),
        (2.5, 4.5, 60.0),
        (1.8, 3.8, 15.0),
        (2.1, 4.1, 45.0),
        (2.3, 4.3, 20.0),
        (2.0, 4.0, 90.0),
        (2.4, 4.4, 10.0),
    ]
    
    for init in initial_points:
        if time.time() > deadline:
            break
        
        try:
            result = minimize(
                objective,
                init,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < best_side:
                r1, r2, theta2 = result.x
                if r1 > 0.5 and r2 > 0.0:
                    positions = generate_symmetric_positions(r1, r2, theta2)
                    if check_validity(positions):
                        side = required_outer_side(positions)
                        if side < best_side:
                            best_side = side
                            best_positions = positions.copy()
        except Exception:
            continue
    
    # Fallback: if no valid symmetric configuration found, use a simple lattice
    if best_positions is None:
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        best_positions = []
        for q, r in coords:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            best_positions.append([x, y])
        best_positions = np.array(best_positions)
        best_side = required_outer_side(best_positions)
    
    # Calculate final exact side
    final_side = required_outer_side(best_positions)
    
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), 0.0])
        
    return inner, (0.0, 0.0), float(final_side), 0.0
