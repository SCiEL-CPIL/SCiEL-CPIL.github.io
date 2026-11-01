"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s kill
    
    best_points = None
    best_radius = 0.0
    
    # Helper to calculate the score (radius) for a set of points
    def calc_radius(pts):
        # Ensure points are inside
        # If any point is outside, return 0 (invalid or very bad)
        # Actually, the score formula handles it: min(1 - |c|) becomes negative.
        # But we want the largest r such that circles are disjoint and inside.
        # r = min(min_dist/2, min(1 - dist_center))
        
        if len(pts) < N:
            return 0.0
            
        # Distances to center
        dists_center = np.linalg.norm(pts, axis=1)
        min_dist_center = np.min(dists_center)
        max_r_container = 1.0 - min_dist_center
        
        if max_r_container <= 0:
            return 0.0
            
        # Pairwise distances
        # To save memory/time for N=50, we can use broadcasting
        # Diff matrix
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Zero out diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist_pair = np.min(dists)
        
        max_r_pairs = min_dist_pair / 2.0
        
        return min(max_r_container, max_r_pairs)

    # Local Search Step: Force-based relaxation
    def local_search(pts, max_iters=1000):
        pts = pts.copy()
        # We want to maximize the minimum distance.
        # This is equivalent to minimizing a potential energy where overlapping circles repel.
        # We use a simple gradient ascent on the score, or a force-directed layout.
        
        # Let's use a soft-min approximation for the gradient to be smooth, 
        # or just use random perturbations + hill climbing?
        # Hill climbing on discrete swaps is hard. 
        # Let's use a continuous optimization: 
        # Objective: maximize min(d_ij/2, 1-|c_i|)
        # This is non-smooth. 
        # Strategy: 
        # 1. Initialize with random points inside (1-r) disk? No, r is unknown.
        # 2. Use a repulsive force model: F = k / d^2 if d < d_target, else 0.
        #    But d_target depends on r.
        #    
        # Better: Use the "soft-min" gradient ascent idea from inspirations but simpler.
        # Or: Use L-BFGS-B on a smooth approximation of the violation.
        
        # Let's try a simpler stochastic local search:
        # 1. Compute current score.
        # 2. Generate a small random perturbation.
        # 3. If score improves, accept.
        # 4. If stuck, try larger jumps (simulated annealing-like).
        
        # However, N=50 is small enough for L-BFGS-B on a smooth proxy.
        # Let's define a smooth objective:
        # V(pts) = sum_{i<j} max(0, 2*r_target - dist_ij)^2 + sum_i max(0, |c_i| + r_target - 1)^2
        # We want to find pts such that V(pts) = 0 for a given r_target.
        # We can binary search on r_target.
        
        # But for now, let's implement a robust gradient ascent on the "min distance" using a soft-min.
        # f(x) = -log(sum(exp(-beta * d_ij))) approximates -min(d_ij) / beta for large beta.
        # This is differentiable.
        
        # We will use scipy.optimize.minimize with L-BFGS-B.
        # Objective to minimize: -soft_min_distance + penalty_for_outside
        
        from scipy.optimize import minimize
        
        def soft_min_dist(pts_flat, beta=100.0):
            pts = pts_flat.reshape(N, 2)
            # Calculate all pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Soft min of pairwise distances
            # We also need to account for the boundary.
            # Boundary constraint: dist_center <= 1 - r.
            # Let's combine them.
            # We want to maximize r.
            # Let's define the objective as:
            # Minimize: - (soft_min of (d_ij, 2*(1-|c_i|)))
            # Wait, the radius is min( min_ij d_ij / 2, min_i (1-|c_i|) ).
            # Let A = min_ij d_ij / 2
            # Let B = min_i (1-|c_i|)
            # Score = min(A, B)
            
            # We can approximate min(A, B) by -soft_max(-A, -B) or similar.
            # Or just minimize a penalty function that penalizes if d_ij < 2r or |c_i| > 1-r.
            
            # Let's stick to the feasibility approach for the local search.
            # Given a target radius r, minimize Violation(pts, r).
            # Violation = sum_{i<j} max(0, 2r - d_ij)^2 + sum_i max(0, |c_i| + r - 1)^2
            
            # But we don't know r. 
            # Let's use a heuristic: 
            # Start with a guess for r (e.g., from current best).
            # Optimize pts to minimize violation for that r.
            # If violation == 0, we have a valid packing for r.
            # Then try to increase r slightly and re-optimize.
            
            # This is complex to implement robustly in a short time.
            # Let's use a simpler "Repulsive Force" simulation which is often effective for packing.
            
            return 0 # Placeholder

        # Actually, let's use the gradient ascent on the score directly with a soft-min.
        # Score S = min( A, B )
        # Approx S_smooth = - (1/beta) * log( sum( exp(-beta * A), exp(-beta * B) ) ) ? No, that's soft-min of A and B.
        # We want to maximize S. So minimize -S.
        
        # Let's define the objective function for L-BFGS-B.
        # We need gradients.
        
        def objective_and_grad(pts_flat):
            pts = pts_flat.reshape(N, 2)
            
            # 1. Pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # 2. Distances to center
            dists_center = np.linalg.norm(pts, axis=1)
            
            # Candidates for radius:
            # A = min(dists) / 2
            # B = 1 - max(dists_center)
            
            # We want to maximize min(A, B).
            # Let's use a soft-min with beta.
            beta = 50.0
            
            # We need to be careful with the log-sum-exp for stability.
            # Let's just use the actual min for the score calculation, but for gradient,
            # we can use a soft approximation.
            
            # Actually, for L-BFGS-B, we can just use the "violation" approach with a fixed r.
            # But r changes.
            
            # Let's try a different tactic: 
            # Run a "Molecular Dynamics" style simulation.
            # Forces:
            # 1. Repulsion between circles: F_ij = k * (1/d_ij^2) * (c_i - c_j) if d_ij < 2*r_est
            # 2. Containment: F_i = k * (1/d_center^2) * (-c_i) if |c_i| > 1 - r_est
            # 3. Centering: Weak pull to origin to prevent drift?
            
            # We need to estimate r. Let's start with r_est = 0.1.
            # We will run this simulation for a fixed number of steps.
            # We will use multiple restarts with different initial positions.
            
            return 0, np.zeros_like(pts_flat)

        # Let's implement the MD simulation directly in the local_search function.
        
        r_est = 0.12 # Initial guess
        # We'll refine r_est by checking the actual min distance later.
        
        # Actually, let's just use a simple iterative improvement:
        # 1. Calculate current min distance d_min and min boundary dist b_min.
        # 2. r = min(d_min/2, b_min).
        # 3. If r < target, apply forces to increase r.
        
        # Let's use a generic "push apart" algorithm.
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            # Calculate forces
            forces = np.zeros_like(pts)
            
            # Update r_est to current best possible radius
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist_pair = np.min(dists)
            
            dists_center = np.linalg.norm(pts, axis=1)
            min_dist_center = np.min(dists_center)
            max_r_container = 1.0 - min_dist_center
            
            current_r = min(min_dist_pair / 2.0, max_r_container)
            
            # We want to increase current_r.
            # Identify the "tight" constraints.
            # If min_dist_pair/2 < max_r_container, we need to increase pairwise distances.
            # If max_r_container < min_dist_pair/2, we need to move points inward.
            
            # Force 1: Pairwise repulsion
            # Only apply to pairs that are closer than 2 * (current_r + epsilon)?
            # Or just apply to all pairs with a strength that decays with distance.
            # F_ij = k * (c_i - c_j) / dist^2  for dist < 2*current_r + margin
            
            margin = 0.05
            threshold = 2.0 * current_r + margin
            
            # Vectorized force calculation is tricky with the threshold.
            # For N=50, N^2 = 2500 pairs. Loop is fine.
            
            for i_idx in range(N):
                for j_idx in range(i_idx + 1, N):
                    d = dists[i_idx, j_idx]
                    if d < threshold and d > 1e-6:
                        # Repulsion force
                        # Strength proportional to violation?
                        # Let's use a constant strength for simplicity, scaled by 1/d^2
                        f = (pts[i_idx] - pts[j_idx]) / (d * d) * 0.01
                        forces[i_idx] += f
                        forces[j_idx] -= f
            
            # Force 2: Boundary containment
            # If a point is too close to the boundary, push it inward.
            boundary_threshold = 1.0 - (current_r + margin)
            for i_idx in range(N):
                d_c = dists_center[i_idx]
                if d_c > boundary_threshold and d_c > 1e-6:
                    # Push towards center
                    # f = -c_i / d_c^2 * strength
                    f = -pts[i_idx] / (d_c * d_c) * 0.01
                    forces[i_idx] += f
            
            # Apply forces with a small step size
            # Use a simple Euler step with damping
            step_size = 0.005
            pts += forces * step_size
            
            # Optional: Normalize or constrain to stay within bounds?
            # If a point goes outside, the next iteration's boundary force will push it back.
            # But if it goes far outside, it might be hard to recover.
            # Let's clip to the unit circle for stability? No, that breaks the gradient.
            # Let's just let the forces handle it.
            
            # Check for improvement?
            # We can calculate the score periodically.
            if i % 10 == 0:
                new_r = calc_radius(pts)
                if new_r > best_radius:
                    best_radius = new_r
                    best_points = pts.copy()
        
        return pts

    # Main Loop: Multiple Restarts
    num_restarts = 500 # We will break early if time runs out
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialization: Random points inside a slightly smaller disk
        # To avoid immediate overlaps, we can use a Poisson Disk Sampling or just random and let MD fix it.
        # Random uniform in disk of radius 0.8
        angles = np.random.uniform(0, 2 * np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 0.8**2, N))
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        
        # Run local search
        optimized_pts = local_search(pts, max_iters=200)
        
        # Check result
        r = calc_radius(optimized_pts)
        if r > best_radius:
            best_radius = r
            best_points = optimized_pts.copy()
            
        # If we found a very good radius, we might want to stop or refine.
        # Best known is ~0.1258.
        if best_radius > 0.125:
            # We can try to do more restarts or refine.
            pass

    if best_points is None:
        # Fallback: Return a valid but poor packing (e.g. hex lattice)
        # Re-use the logic from the seed program as fallback
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                return points[np.argsort(radius, kind="stable")[:N]]
            spacing *= 0.98

    return best_points
# EVOLVE-BLOCK-END
