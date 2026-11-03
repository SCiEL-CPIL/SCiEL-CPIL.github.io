import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    # Lattice basis for hexagonal packing of unit hexagons
    # Distance between centers of touching hexagons is sqrt(3)
    a = np.sqrt(3)
    v1 = np.array([a, 0])
    v2 = np.array([a/2, 1.5])
    
    # Generate a grid of candidate points
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # Select N closest to origin
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)[:N]
    x0 = points[indices].flatten()
    
    # Precompute normals for the container (flat-top hexagon)
    # Vertices at 0, 60, ...
    # Normals at 30, 90, ...
    n_angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    r_in = np.sqrt(3) / 2.0
    
    def objective(x):
        if time.time() > deadline:
            return 1e6
        pos = x.reshape(N, 2)
        
        # Calculate required S
        # Projection of each hexagon center onto each normal
        # proj = pos @ normals.T + r_in
        # R_out = max(proj)
        # S = 2 / sqrt(3) * R_out
        projs = pos @ normals.T + r_in
        R_out = np.max(projs)
        S = (2.0 / np.sqrt(3.0)) * R_out
        
        # Penalty for overlap
        # Constraint: ||p_i - p_j|| >= sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3):
                    penalty += (np.sqrt(3) - d) ** 2
        
        return S + 100.0 * penalty

    # Optimization
    # Use L-BFGS-B for better handling of large N
    try:
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-8})
    except Exception:
        res = None

    if time.time() > deadline or res is None:
        best_x = x0
    else:
        best_x = res.x
    
    pos = best_x.reshape(N, 2)
    
    # Post-processing: Ensure validity by resolving any remaining overlaps
    # If overlaps exist, push points apart
    # This is a simple iterative relaxation
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        moved = False
        for i in range(N):
            for j in range(i+1, N):
                d_vec = pos[i] - pos[j]
                d = np.linalg.norm(d_vec)
                min_dist = np.sqrt(3)
                if d < min_dist:
                    if d < 1e-6:
                        # Random nudge
                        d_vec = np.array([1.0, 0.0])
                        d = 1.0
                    # Push apart by half the overlap
                    overlap = min_dist - d
                    d_unit = d_vec / d
                    pos[i] += 0.5 * overlap * d_unit
                    pos[j] -= 0.5 * overlap * d_unit
                    moved = True
        if not moved:
            break

    # Recalculate S based on final positions
    projs = pos @ normals.T + r_in
    R_out = np.max(projs)
    S = (2.0 / np.sqrt(3.0)) * R_out
    
    # Final validity check
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(pos[i] - pos[j]) < np.sqrt(3) - 1e-5:
                valid = False
    
    if not valid:
        # Fallback: Return the lattice configuration which is guaranteed valid
        pos = points[indices]
        projs = pos @ normals.T + r_in
        R_out = np.max(projs)
        S = (2.0 / np.sqrt(3.0)) * R_out

    # Format output
    inner = []
    for p in pos:
        inner.append([p[0], p[1], 0.0])
        
    return inner, (0.0, 0.0), S, 0.0
