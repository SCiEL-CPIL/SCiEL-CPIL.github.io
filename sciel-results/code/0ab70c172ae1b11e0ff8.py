# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Pack N equal circles inside the unit circle to maximize the common radius.
    
    Uses a physics-based relaxation approach with repulsive springs and a 
    harmonic boundary potential. A binary search is used to find the maximum 
    feasible radius, where feasibility is determined by whether the system 
    can relax to a non-overlapping state within a fixed number of steps.
    """
    N = 50
    
    def simulate_packing(r, steps=2000, dt=0.01, k_rep=50.0, k_bound=50.0, damping=0.98, init_type='rings'):
        """
        Simulate the relaxation of N circles of radius r.
        Returns the final positions and the maximum violation.
        """
        # Initialize positions
        if init_type == 'rings':
            # Approximate ring structure
            # 1, 6, 12, 18, 13
            rings = [1, 6, 12, 18, 13]
            radii = [0.0, 0.25, 0.5, 0.75, 0.9]
            points = []
            for k, (n_pts, rad) in enumerate(zip(rings, radii)):
                for i in range(n_pts):
                    angle = 2 * np.pi * i / n_pts + (np.pi / n_pts if k % 2 == 0 else 0)
                    points.append([rad * np.cos(angle), rad * np.sin(angle)])
            P = np.array(points[:N], dtype=float)
        else:
            P = np.random.uniform(-0.5, 0.5, (N, 2))
            
        V = np.zeros((N, 2))
        
        # Precompute pair indices for efficiency? 
        # For N=50, i<j is 1225 pairs. 
        # We can just do full matrix or loop. 
        # Full matrix in numpy is fast.
        
        max_violation = np.inf
        
        for step in range(steps):
            # Calculate Forces
            F = np.zeros((N, 2))
            
            # Pairwise repulsion
            # diff[i,j] = P[i] - P[j]
            diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0) # Avoid division by zero
            
            # Overlap amount: max(0, 2r - d)
            overlap = np.maximum(0, 2 * r - dists)
            
            # Force magnitude: k_rep * overlap
            # Direction: diff / dist
            # F_ij = k_rep * overlap * diff / dist
            # Note: This force pushes i away from j.
            # Safe division
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            np.fill_diagonal(safe_dists, 1.0)
            
            force_vecs = k_rep * overlap / safe_dists
            np.fill_diagonal(force_vecs, 0)
            
            # Sum forces from all j on i
            # F[i] = sum_j (force_vecs[i,j] * diff[i,j])
            F += np.sum(force_vecs[:, :, np.newaxis] * diff, axis=1)
            
            # Boundary force
            radii = np.linalg.norm(P, axis=1)
            bound_violation = np.maximum(0, radii - (1.0 - r))
            
            # Force is inward: -k_bound * violation * (P / |P|)
            safe_radii = np.where(radii > 1e-10, radii, 1.0)
            F -= k_bound * bound_violation[:, np.newaxis] * P / safe_radii[:, np.newaxis]
            
            # Integration (Velocity Verlet / Leapfrog style with damping)
            # V_new = (V + F * dt) * damping
            # P_new = P + V_new * dt
            
            V = (V + F * dt) * damping
            P += V * dt
            
            # Check convergence/violation every 100 steps to save time
            if step % 100 == 0 or step == steps - 1:
                dists_check = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
                np.fill_diagonal(dists_check, np.inf)
                min_dist = np.min(dists_check)
                max_radius = np.max(np.linalg.norm(P, axis=1))
                
                # Violation is max(0, 2r - min_dist, max_radius - (1-r))
                v1 = 2 * r - min_dist
                v2 = max_radius - (1.0 - r)
                current_violation = max(0, v1, v2)
                
                if current_violation < max_violation:
                    max_violation = current_violation
                    if current_violation < 1e-5:
                        # Converged
                        break
                        
        # Final check
        dists_final = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists_final, np.inf)
        min_dist_final = np.min(dists_final)
        max_radius_final = np.max(np.linalg.norm(P, axis=1))
        
        final_violation = max(0, 2 * r - min_dist_final, max_radius_final - (1.0 - r))
        
        return P, final_violation

    np.random.seed(42)
    
    r_low = 0.1
    r_high = 0.126 # Slightly above known best to be safe, but we know 0.1258 is best known
    
    best_centers = None
    best_r = r_low
    
    # Binary search
    for _ in range(15): # 15 iterations gives precision ~ 0.026 / 32768 ~ 1e-6
        r_mid = (r_low + r_high) / 2
        
        # Try to solve for r_mid
        # We might need multiple restarts if it gets stuck, but rings init is usually good
        P, violation = simulate_packing(r_mid)
        
        if violation < 1e-5:
            r_low = r_mid
            best_r = r_mid
            best_centers = P
        else:
            r_high = r_mid
            
    # If we have a best center, do a final high-precision polish
    if best_centers is not None:
        # Run a longer simulation at the best r found to ensure it's tight
        # Or just return the one we have. 
        # Let's do a quick polish with more steps
        P_polished, v_polished = simulate_packing(best_r, steps=5000, damping=0.99)
        if v_polished < 1e-5:
            best_centers = P_polished
        else:
            # If polish failed, keep the previous one
            pass

    if best_centers is None:
        # Fallback: just return the rings init for r_low
        P, _ = simulate_packing(r_low)
        best_centers = P

    return best_centers
# EVOLVE-BLOCK-END
