import numpy as np
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Pre-compute geometry constants
    # Inner hexagons have side length 1.
    # Vertices relative to center (angle 0):
    # (1, 0), (1/2, sqrt(3)/2), (-1/2, sqrt(3)/2), (-1, 0), (-1/2, -sqrt(3)/2), (1/2, -sqrt(3)/2)
    # These are unit vectors.
    base_angles = np.deg2rad(np.arange(6) * 60)
    base_verts = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Outer hexagon normals.
    # For a regular hexagon with side S, the inradius (distance from center to edge midpoint) is R_in = S * sqrt(3) / 2.
    # The condition for containment is: for all vertices v of inner hexagons, 
    # dot(v, n_k) <= R_in for all 6 normals n_k of the outer hexagon.
    # The outer hexagon is fixed at rotation 0 (pointy-right) for simplicity, or we can allow rotation.
    # Let's fix outer rotation to 0 to reduce search space dimensionality, as symmetry suggests optimal might align.
    # Normals for outer hexagon (pointy-right, vertex at 0 deg):
    # Edges are perpendicular to angles 30, 90, 150, 210, 270, 330.
    outer_normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
    
    # To contain a point p, we need dot(p, n) <= R_in for all n.
    # So R_in must be >= max_k ( max_i ( dot(p_i, n_k) ) )
    # S = R_in * 2 / sqrt(3)

    # State representation: 
    # x[0:24] : positions (x, y) of 12 hexagons
    # x[24:36] : rotations (radians) of 12 hexagons
    
    def get_state(x):
        """Extract positions and rotations from flat vector."""
        pos = x[0:24].reshape(12, 2)
        rots = x[24:36]
        return pos, rots

    def compute_S_and_violation(x):
        """
        Computes the minimal S required to contain the configuration defined by x.
        Also computes a penalty for overlaps to guide the search towards valid packings.
        Returns (S, overlap_penalty).
        """
        pos, rots = get_state(x)
        
        # 1. Compute S based on containment
        # For each inner hexagon i, generate its vertices
        # Vertices of hex i: pos[i] + R(rot_i) @ base_verts
        # We need max projection of ALL vertices onto ALL outer normals.
        
        # Vectorized approach:
        # rots: (12,)
        # pos: (12, 2)
        # base_verts: (6, 2)
        
        # Rotation matrices for each hexagon
        cos_r = np.cos(rots)
        sin_r = np.sin(rots)
        
        # Rotated vertices for all hexagons: (12, 6, 2)
        # v_local = R @ base_verts
        # R = [[c, -s], [s, c]]
        # v_local[:, k, 0] = c * base_verts[k, 0] - s * base_verts[k, 1]
        # v_local[:, k, 1] = s * base_verts[k, 0] + c * base_verts[k, 1]
        
        bv_x = base_verts[:, 0] # (6,)
        bv_y = base_verts[:, 1] # (6,)
        
        # (12, 6)
        vx = cos_r[:, None] * bv_x[None, :] - sin_r[:, None] * bv_y[None, :]
        vy = sin_r[:, None] * bv_x[None, :] + cos_r[:, None] * bv_y[None, :]
        
        # Add center positions: (12, 6, 2)
        px = pos[:, 0:1] # (12, 1)
        py = pos[:, 1:2] # (12, 1)
        
        vx = vx + px
        vy = vy + py
        
        # Stack into (12, 6, 2)
        all_verts = np.column_stack((vx, vy))
        
        # Flatten to (72, 2) for projection
        flat_verts = all_verts.reshape(-1, 2)
        
        # Project onto outer normals (6, 2)
        # Projections: (72, 6)
        projections = flat_verts @ outer_normals.T
        
        # Max projection for each normal direction
        max_projections = np.max(projections, axis=0) # (6,)
        
        # R_in must be at least the max of these
        R_in = np.max(max_projections)
        
        # S = R_in * 2 / sqrt(3)
        S = R_in * 2.0 / np.sqrt(3.0)
        
        # 2. Compute Overlap Penalty
        # We need to ensure hexagons do not overlap.
        # Approximation: Check distance between centers.
        # For two unit hexagons, if distance < 2 * apothem = sqrt(3), they *might* overlap depending on rotation.
        # A stricter check: minimum distance between boundaries.
        # For speed in SA, we use a center-distance penalty with a threshold that is safe for worst-case rotation.
        # Worst-case "width" of hexagon is 2 (vertex to vertex).
        # If centers are < 2, they definitely overlap? No, if aligned flat-to-flat, min dist is sqrt(3).
        # If aligned point-to-point, min dist is 2.
        # So if dist < sqrt(3), they always overlap? 
        # Let's use a soft penalty for dist < 1.8 (conservative) to push them apart.
        # Actually, a better proxy for "disjoint" is checking if the Minkowski difference contains origin.
        # Too slow for SA.
        
        # Let's use a pairwise distance penalty.
        # Threshold: The minimal distance between centers for non-overlapping unit hexagons is sqrt(3) (approx 1.732).
        # If we allow rotation, the "effective radius" varies between 0.866 (apothem) and 1.0 (vertex).
        # So min center dist can be as low as 0.866 + 0.866 = 1.732 if flat-to-flat.
        # Max min-center-dist is 2.0 if point-to-point.
        # To be safe, we penalize if dist < 1.7.
        
        penalty = 0.0
        # Pairwise distances
        # pos is (12, 2)
        # cdist would be fast, but let's do it manually or with numpy broadcasting
        # dist_matrix (12, 12)
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Only consider upper triangle
        iu = np.triu_indices(12, k=1)
        pair_dists = dists[iu]
        
        # Threshold for overlap risk
        # If we assume optimal packing, they will likely be close to touching.
        # We want to push them apart if they are too close.
        # Let's use a threshold of 1.75 to be safe.
        threshold = 1.75
        violations = np.maximum(0, threshold - pair_dists)
        penalty = np.sum(violations ** 2) * 10.0
        
        return S, penalty

    # Objective function for SA
    # We want to minimize S + lambda * penalty
    # But S is derived from the configuration.
    # If penalty > 0, the configuration is invalid. We should heavily penalize invalid configurations.
    # Score = S if penalty == 0 else S + 100 * penalty
    
    def objective(x):
        S, penalty = compute_S_and_violation(x)
        if penalty > 1e-6:
            return S + 100.0 * penalty
        return S

    # Initial State
    # Start with a grid layout
    x0 = np.zeros(36)
    # Grid 4x3
    # Spacing approx 2.0
    for i in range(12):
        row = i // 4
        col = i % 4
        x0[2*i] = (col - 1.5) * 2.1
        x0[2*i+1] = (row - 1.0) * 2.0
        x0[24+i] = 0.0 # rotation 0

    # Simulated Annealing Parameters
    T = 5.0  # Initial temperature
    cooling_rate = 0.999
    T_min = 0.01
    max_iter = 200000
    
    current_x = x0.copy()
    current_score = objective(current_x)
    
    best_x = current_x.copy()
    best_score = current_score
    
    # Pre-compute random steps? No, generate on fly.
    # Step size: Gaussian noise.
    # We need to tune the step size. 
    # Positions range roughly [-5, 5]. Rotations range [0, 2pi].
    # Let's use different scales for pos and rot.
    pos_scale = 0.5
    rot_scale = 0.5
    
    np.random.seed(42) # For reproducibility in debugging, but maybe remove later? 
    # Actually, random is fine.
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Generate neighbor
        new_x = current_x.copy()
        # Add noise to positions
        new_x[0:24] += np.random.normal(0, pos_scale, 24)
        # Add noise to rotations
        new_x[24:36] += np.random.normal(0, rot_scale, 12)
        
        new_score = objective(new_x)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Better
            current_x = new_x
            current_score = new_score
            if current_score < best_score:
                best_x = current_x.copy()
                best_score = current_score
        else:
            # Worse
            # Accept with probability exp(-delta / T)
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.random() < prob:
                    current_x = new_x
                    current_score = new_score
        
        # Cool down
        T *= cooling_rate
        
        # Adaptive step size? 
        # If we are stuck (no improvement for a while), increase step size.
        # If we are jumping too much (score increasing), decrease step size.
        # Simple heuristic:
        if i % 1000 == 0:
            # Check recent improvement
            pass # Keeping it simple for now.

    # Final check on best solution
    # We might have a solution with small penalty but high S, or low S but high penalty.
    # We want the lowest S among valid (or near-valid) solutions.
    # Let's refine the best solution found with a few iterations of local search (Nelder-Mead) to polish?
    # Or just return the best SA result.
    
    # Let's do a quick polish with Nelder-Mead on the best solution found by SA
    # This helps fix small overlaps or tighten S.
    
    from scipy.optimize import minimize
    
    def polish_objective(x):
        S, penalty = compute_S_and_violation(x)
        return S + 1000.0 * penalty # High penalty for overlap to ensure validity
        
    # Run NM for a short time
    try:
        res = minimize(polish_objective, best_x, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score + 1e-6: # Only accept if it didn't get worse significantly
             # Check validity
             S_final, pen_final = compute_S_and_violation(res.x)
             if pen_final < 1e-3: # Valid enough
                 best_x = res.x
                 best_score = S_final
    except Exception:
        pass

    # Extract final parameters
    pos, rots = get_state(best_x)
    
    inner = []
    for i in range(12):
        angle_deg = np.rad2deg(rots[i])
        inner.append([pos[i, 0], pos[i, 1], angle_deg])
        
    # Outer hexagon center (0,0), side length best_score, rotation 0
    return inner, (0.0, 0.0), best_score, 0.0

# Run construct to test
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Best Side Length: {side}")
