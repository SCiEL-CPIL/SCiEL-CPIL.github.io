import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Define normals for the outer hexagon (vertex at angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300
    # Normals to sides are at 30, 90, 150, 210, 270, 330
    angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    
    # Precompute inner hexagon vertices relative to center
    # Unit hexagon vertices (side 1)
    # Vertices at 0, 60, ... 300
    inner_angles = np.deg2rad(np.arange(0, 360, 60))
    cos_inner = np.cos(inner_angles)
    sin_inner = np.sin(inner_angles)
    
    # Initial positions: Hexagonal lattice
    # Axial coords (q, r)
    # (0,0)
    # (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # (2,-1), (2,-2), (1,-2), (-1,2), (-2,1)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (-1, 2), (-2, 1)
    ]
    
    initial_params = []
    for q, r in axial_coords:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        initial_params.extend([x, y, 0.0]) # angle 0
    
    initial_params = np.array(initial_params)
    
    def objective(params):
        # Check time
        if time.time() - start_time > 140:
            return 1e10 # Huge penalty to stop if possible, though minimize doesn't support this directly
            
        positions = params.reshape((N, 3))
        xs = positions[:, 0]
        ys = positions[:, 1]
        thetas = positions[:, 2]
        
        max_S = 0.0
        
        # Calculate required S for containment
        # For each hexagon, get vertices
        for i in range(N):
            cx = xs[i]
            cy = ys[i]
            theta = thetas[i]
            
            # Rotate inner vertices
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            
            # Local vertices (x', y')
            vx_local = cos_inner
            vy_local = sin_inner
            
            # Rotate
            vx_rot = vx_local * cos_t - vy_local * sin_t
            vy_rot = vx_local * sin_t + vy_local * cos_t
            
            # Global vertices
            vx = cx + vx_rot
            vy = cy + vy_rot
            
            # For each vertex, check containment
            # Projection onto normals
            # proj = vx * cos_a + vy * sin_a
            # S_req = max(0, (2/sqrt(3)) * proj)
            projs = vx[0]*cos_a + vy[0]*sin_a # Wait, vx is array of 6, cos_a is array of 6.
            # We need to check ALL 6 vertices against ALL 6 normals.
            # This is a 6x6 matrix multiplication.
            # V (6x1) * N (1x6) -> 6x6 matrix of projections
            # Actually, for each vertex v, we need max over normals n of v.n.
            # Then S_req for that vertex is (2/sqrt(3)) * max(v.n).
            # Then global S is max over all vertices.
            
            # Efficient way:
            # Matrix V (6x2) of vertices.
            # Matrix N (6x2) of normals.
            # Projections = V @ N.T (6x6 matrix, P[i,j] is dot product of vertex i and normal j)
            # Max projection for each vertex i is max_j P[i,j]
            # S_req_i = (2/sqrt(3)) * max_j P[i,j]
            # S_global = max_i S_req_i
            
            V = np.vstack((vx, vy)).T # Shape (6, 2)
            N_mat = np.vstack((cos_a, sin_a)).T # Shape (6, 2)
            
            Projs = V @ N_mat.T # Shape (6, 6)
            MaxProjs = np.max(Projs, axis=1) # Shape (6,)
            S_reqs = (2.0 / np.sqrt(3)) * MaxProjs
            S_max_local = np.max(S_reqs)
            
            if S_max_local > max_S:
                max_S = S_max_local
        
        # Overlap Penalty
        penalty = 0.0
        min_dist = np.sqrt(3) # Distance between centers for non-overlap (approx)
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx*dx + dy*dy)
                if d < min_dist:
                    penalty += (min_dist - d)**2
        
        # Weight for penalty
        # If penalty is high, S should be pushed up? No, penalty is added to objective.
        # We want to minimize S + penalty.
        # If they overlap, penalty increases objective, so optimizer moves them apart.
        # However, if they are far apart, S increases.
        # Trade-off.
        
        return max_S + 100.0 * penalty

    # Run optimization
    # Use L-BFGS-B
    try:
        result = minimize(objective, initial_params, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-6})
        best_params = result.x
    except Exception:
        best_params = initial_params

    # Extract best solution
    best_positions = best_params.reshape((N, 3))
    inner = [[float(x), float(y), float(t)] for x, y, t in best_positions]
    
    # Recalculate final S
    final_S = 0.0
    for i in range(N):
        cx = best_positions[i, 0]
        cy = best_positions[i, 1]
        theta = best_positions[i, 2]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        vx_local = cos_inner
        vy_local = sin_inner
        vx_rot = vx_local * cos_t - vy_local * sin_t
        vy_rot = vx_local * sin_t + vy_local * cos_t
        vx = cx + vx_rot
        vy = cy + vy_rot
        V = np.vstack((vx, vy)).T
        N_mat = np.vstack((cos_a, sin_a)).T
        Projs = V @ N_mat.T
        MaxProjs = np.max(Projs, axis=1)
        S_reqs = (2.0 / np.sqrt(3)) * MaxProjs
        S_max_local = np.max(S_reqs)
        if S_max_local > final_S:
            final_S = S_max_local

    return inner, (0.0, 0.0), float(final_S), 0.0
# EVOLVE-BLOCK-END
