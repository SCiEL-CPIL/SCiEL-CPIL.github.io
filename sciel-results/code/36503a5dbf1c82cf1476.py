import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s limit, 10s margin
    
    # Directions for the outer hexagon normals
    n0 = np.array([1.0, 0.0])
    n1 = np.array([0.5, np.sqrt(3.0)/2.0])
    n2 = np.array([-0.5, np.sqrt(3.0)/2.0])
    normals = np.array([n0, n1, n2])
    
    min_dist = np.sqrt(3.0)
    
    def calc_side_length(positions):
        # positions: (N, 2)
        # Calculate max projection in each normal direction
        # proj: (3, N)
        proj = positions @ normals.T
        # max abs proj for each normal
        max_abs_proj = np.max(np.abs(proj), axis=1)
        # M is the max of these
        M = np.max(max_abs_proj)
        # Side length L
        L = (2.0 / np.sqrt(3.0)) * (M + np.sqrt(3.0)/2.0)
        return L
    
    def calc_penalty(positions):
        # Calculate pairwise distances
        # diff: (N, N, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        # Fill diagonal with inf to ignore
        np.fill_diagonal(dist, np.inf)
        # Find min dist
        min_d = np.min(dist)
        if min_d < min_dist:
            # Penalty is sum of squared violations
            violations = np.clip(min_dist - dist, 0, None)
            # Exclude diagonal
            violations = violations[~np.eye(N, dtype=bool)]
            return np.sum(violations**2)
        return 0.0

    def objective(x):
        positions = x.reshape(N, 2)
        L = calc_side_length(positions)
        P = calc_penalty(positions)
        # Weight for penalty
        # If L is around 4, and violations are small, we want P to be significant
        # Let's use a large multiplier
        return L + 10.0 * P

    best_L = float('inf')
    best_positions = None
    
    # Initial guess generation
    # Hexagonal lattice basis
    v1 = np.array([np.sqrt(3.0), 0.0])
    v2 = np.array([np.sqrt(3.0)/2.0, 1.5])
    
    # Generate a few random initial configurations
    num_restarts = 50
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Create initial positions
        # Start with 7 central, add 4 random ones nearby
        positions = np.zeros((N, 2))
        # 7 central
        central_pts = [
            np.array([0.0, 0.0]),
            v1, -v1,
            v2, -v2,
            v1 - v2, v2 - v1
        ]
        for i in range(7):
            positions[i] = central_pts[i]
        
        # 4 outer points
        # Place them in the next shell, slightly perturbed
        # Next shell points are 2*v1, 2*v2, etc.
        # Let's pick 4 random directions from the next shell
        # Or just place them at random positions with radius ~ 2*sqrt(3)
        for i in range(7, N):
            angle = np.random.uniform(0, 2*np.pi)
            radius = 2.0 * np.sqrt(3.0) + np.random.uniform(-0.5, 0.5)
            positions[i] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            
        # Add small random noise to break symmetry
        positions += np.random.normal(0, 0.1, (N, 2))
        
        # Optimize
        res = minimize(objective, positions.flatten(), method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success or True: # Nelder-Mead often doesn't set success=True but converges
            pos_opt = res.x.reshape(N, 2)
            # Check if it's a valid packing (penalty close to 0)
            P_opt = calc_penalty(pos_opt)
            if P_opt < 1e-3:
                L_opt = calc_side_length(pos_opt)
                if L_opt < best_L:
                    best_L = L_opt
                    best_positions = pos_opt
            else:
                # If penalty is high, maybe try to repair or just ignore
                # For now, ignore invalid packs
                pass

    if best_positions is None:
        # Fallback to the seed solution if optimization failed
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct output
    # inner: (x, y, angle_degrees)
    # center: (x, y) of outer hexagon
    # side: outer side length
    # angle: outer hexagon rotation
    
    # The outer hexagon is centered at (0,0) in our optimization frame.
    # We should center the inner hexagons relative to the outer one.
    # Our optimization assumed the bounding box is centered at (0,0).
    # So the outer center is (0,0).
    # The inner positions are already relative to (0,0).
    
    # However, we should verify the centering.
    # The min/max projections determine the bounding box.
    # If the cluster is not perfectly centered, the "M" calculation still works 
    # because we used abs projections.
    # But the "center" of the outer hexagon should be the center of the bounding box.
    # Since we used abs, the box is symmetric around 0.
    
    inner_list = []
    for i in range(N):
        x, y = best_positions[i]
        inner_list.append([x, y, 0.0])
        
    outer_center = [0.0, 0.0]
    outer_side = best_L
    outer_angle = 0.0
    
    return inner_list, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
