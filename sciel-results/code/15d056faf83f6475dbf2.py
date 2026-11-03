"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def construct():
    start_time = time.time()
    
    # Minimum distance between centers of non-overlapping unit hexagons
    min_dist = 2.0 * math.sqrt(3.0)
    
    def hexagon_vertices(cx, cy, angle_deg):
        """Return 6 vertices of a unit hexagon centered at (cx,cy) with given rotation."""
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        verts = []
        for k in range(6):
            theta = angle + k * math.pi / 3
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append([x, y])
        return np.array(verts)
    
    def compute_outer_side(positions, angles):
        """Compute the minimum outer hexagon side length needed to contain all hexagons.
        Returns (side_length, center, outer_angle)"""
        # Collect all vertices
        all_verts = []
        for i in range(N):
            verts = hexagon_vertices(positions[i], angles[i], 0)  # angle already in positions
            all_verts.extend(verts.tolist())
        all_verts = np.array(all_verts)
        
        # Find center as mean of all vertices
        center = np.mean(all_verts, axis=0)
        centered_verts = all_verts - center
        
        # Search over outer hexagon orientation to minimize side length
        best_s = float('inf')
        best_outer_angle = 0.0
        
        # For a regular hexagon with side s and center at origin, the distance from center
        # to each edge (apothem) is s*sqrt(3)/2
        # The vertices are at distance s from center
        # A point (x,y) is inside the hexagon if for all 6 edge normals n_k,
        # dot(point, n_k) <= apothem
        # where n_k are the outward normals to the 6 edges
        
        # The 6 edge normals for a hexagon with rotation phi are at angles phi + pi/6 + k*pi/3
        # The constraint is: max_k dot(v, n_k) <= s*sqrt(3)/2 for all vertices v
        # So s >= (2/sqrt(3)) * max_k dot(v, n_k)
        
        # We want to minimize s over phi
        # s(phi) = (2/sqrt(3)) * max_k max_v dot(v, n_k(phi))
        
        phi_steps = np.linspace(0, math.pi/3, 121)
        for phi in phi_steps:
            max_dot = 0.0
            for k in range(6):
                ang = phi + math.pi/6 + k * math.pi/3
                n = np.array([math.cos(ang), math.sin(ang)])
                dots = centered_verts @ n
                max_dot = max(max_dot, np.max(np.abs(dots)))
            s = (2.0 / math.sqrt(3.0)) * max_dot
            if s < best_s:
                best_s = s
                best_outer_angle = phi
        
        return best_s, (center[0], center[1]), best_outer_angle
    
    def objective(params):
        """Objective: minimize outer hexagon side length with penalty for overlaps."""
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:3*N]
        
        # Check for overlaps: minimum center distance must be >= min_dist
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i,0] - positions[j,0]
                dy = positions[i,1] - positions[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        # Compute outer side length
        s, _, _ = compute_outer_side(positions, angles)
        
        return s + 10.0 * penalty
    
    # Initial configurations to try
    def make_initial_config(idx):
        """Generate different initial configurations."""
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        
        if idx == 0:
            # Hexagonal cluster
            positions[0] = [0, 0]
            for k in range(6):
                theta = k * math.pi / 3
                positions[k+1] = [min_dist * math.cos(theta), min_dist * math.sin(theta)]
            # Add 5 more around
            for k in range(5):
                theta = k * 2 * math.pi / 5
                positions[k+7] = [2 * min_dist * math.cos(theta), 2 * min_dist * math.sin(theta)]
            angles[:] = 0.0
        elif idx == 1:
            # Rectangular grid 4x3
            for i in range(4):
                for j in range(3):
                    idx_flat = i * 3 + j
                    positions[idx_flat] = [i * min_dist, j * min_dist * math.sqrt(3)]
            angles[:] = 0.0
        elif idx == 2:
            # Linear arrangement
            for i in range(N):
                positions[i] = [i * min_dist, 0]
            angles[:] = 0.0
        elif idx == 3:
            # Two rows of 6
            for i in range(6):
                positions[i] = [i * min_dist, 0]
                positions[i+6] = [i * min_dist, min_dist * math.sqrt(3)]
            angles[:] = 0.0
        else:
            # Random
            np.random.seed(idx)
            positions = np.random.randn(N, 2) * 10
            angles = np.random.randn(N) * math.pi / 3
        
        # Center the configuration
        center = np.mean(positions, axis=0)
        positions -= center
        
        return np.concatenate([positions.flatten(), angles])
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial configurations
    n_restarts = 8
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 30:
            break
        
        x0 = make_initial_config(restart)
        
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_score:
                params = result.x
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:3*N]
                
                # Verify no overlaps
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        dx = positions[i,0] - positions[j,0]
                        dy = positions[i,1] - positions[j,1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    s, center, outer_angle = compute_outer_side(positions, angles)
                    if s < best_score:
                        best_score = s
                        best_result = (positions, angles, center, outer_angle)
                        
        except Exception:
            continue
    
    # If no valid result found, use fallback
    if best_result is None:
        # Fallback: simple linear arrangement
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        for i in range(N):
            positions[i] = [i * min_dist, 0]
        center = np.mean(positions, axis=0)
        s, center, outer_angle = compute_outer_side(positions, angles)
        best_result = (positions, angles, center, outer_angle)
        best_score = s
    
    positions, angles, center, outer_angle = best_result
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([positions[i,0], positions[i,1], math.degrees(angles[i])])
    
    return inner, (center[0], center[1]), best_score, math.degrees(outer_angle)

# EVOLVE-BLOCK-END
