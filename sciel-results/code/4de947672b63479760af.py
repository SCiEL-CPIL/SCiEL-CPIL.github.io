import numpy as np

def construct():
    N = 50
    # Parameters:
    # R: radius of the sphere on which centers lie
    # theta, phi: spherical coordinates for each of the N points
    # Total params: 1 + 2*N = 101
    
    # Initialization
    # Start with a reasonable R. The max possible R is 0.5 (touching walls).
    # If R is too large, points must be far apart.
    # If R is too small, wall distance is large but pair distance is small.
    # Optimal R is likely around 0.3-0.4.
    R = 0.35
    # Fibonacci sphere for initial angles
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    thetas = np.zeros(N)
    phis = np.zeros(N)
    for i in range(N):
        cos_theta = 1.0 - 2.0 * (i + 0.5) / N
        sin_theta = np.sqrt(1.0 - cos_theta**2)
        theta = np.arccos(cos_theta)
        phi = i * golden_angle
        thetas[i] = theta
        phis[i] = phi
    
    # Optimization
    max_iter = 5000
    step_size = 0.05
    best_r = 0.0
    best_R = R
    best_thetas = thetas.copy()
    best_phis = phis.copy()
    
    # Helper to compute centers and radius
    def get_centers(R_val, th, ph):
        x = R_val * np.sin(th) * np.cos(ph)
        y = R_val * np.sin(th) * np.sin(ph)
        z = R_val * np.cos(th)
        return np.column_stack((x, y, z))
    
    def compute_score(R_val, th, ph):
        centers = get_centers(R_val, th, ph)
        # Wall distance
        # Since centers are on a sphere of radius R_val centered at origin,
        # max coord is <= R_val.
        # min wall dist = 0.5 - R_val
        wall_dist = 0.5 - R_val
        
        # Pair distances
        # dist(i,j) = R_val * |u_i - u_j|
        # We want min dist / 2
        # |u_i - u_j|^2 = 2 - 2 cos(angle_diff)
        # cos(angle_diff) = cos(t_i - t_j) if same phi? No.
        # Dot product of unit vectors:
        # u_i . u_j = cos(t_i)cos(t_j) + sin(t_i)sin(t_j)cos(p_i - p_j)
        
        # Compute all pairwise dot products efficiently
        # u_i = [sin(t_i)cos(p_i), sin(t_i)sin(p_i), cos(t_i)]
        # Let's just compute centers and distances, it's fast enough for N=50
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        pair_r = min_pair_dist / 2.0
        
        return min(wall_dist, pair_r)

    current_r = compute_score(R, thetas, phis)
    if current_r > best_r:
        best_r = current_r
        best_R = R
        best_thetas = thetas.copy()
        best_phis = phis.copy()

    for i in range(max_iter):
        # Compute gradient numerically or analytically?
        # Analytical is better.
        # Let's use numerical gradient for simplicity and robustness, 
        # but N=50 means 101 variables. 101 * 2 * (50*50) operations per step?
        # 101 * 2 * 2500 = 500k operations. 5000 iters = 2.5 billion. Might be too slow in Python loops.
        # Let's try to be smarter or reduce iterations.
        
        # Alternative: Use a simpler optimization.
        # Since the prompt asks to improve the current program's approach (gradient ascent) 
        # but with the new design (spherical constraint), let's stick to the spherical model 
        # but use a more efficient update.
        
        # Actually, let's just do a simple local search or gradient ascent with a smaller step and fewer iterations to stay within time limits.
        # Or, we can just do a random restart / simulated annealing style?
        # No, the current program uses gradient ascent. Let's use gradient ascent on the 101 params.
        
        # To speed up, we can compute the score and its gradient.
        # Score S = min(W, P)
        # W = 0.5 - R
        # P = 0.5 * min_{i<j} |c_i - c_j|
        
        # Let's just use finite difference gradient for the 101 parameters.
        # 101 params. Step size h=1e-5.
        # This requires 101 * 2 evaluations of compute_score.
        # Each eval is O(N^2). 2500 ops.
        # 202 * 2500 = 500k ops per iteration.
        # 5000 iterations = 2.5e9 ops. In numpy, vectorized, this might be ok?
        # But compute_score creates arrays.
        
        # Let's optimize the inner loop.
        
        # Actually, a simpler approach:
        # The optimal packing is likely symmetric.
        # We can fix the symmetry and optimize fewer variables.
        # But let's just do the general spherical optimization.
        
        # Let's reduce max_iter to 1000 to be safe.
        # And use a slightly larger step.
        
        # Compute gradient
        h = 1e-5
        grad = np.zeros(101)
        
        # We need to evaluate score at R, thetas, phis
        # And at R+h, R-h, etc.
        
        # This is too slow in pure Python loops if not careful.
        # Let's use a different strategy: 
        # Use the fact that the best known packing is not perfectly spherical.
        # But the directive says "constrained to lie on the surface of an inscribed sphere".
        # So I must follow that.
        
        # Let's try to do a steepest descent on the 101 variables using numpy vectorization where possible.
        
        # Compute current score
        centers = get_centers(R, thetas, phis)
        wall_dist = 0.5 - R
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        min_pair_dist = np.min(dist[~np.eye(N, dtype=bool)])
        pair_r = min_pair_dist / 2.0
        current_r = min(wall_dist, pair_r)
        
        if current_r > best_r:
            best_r = current_r
            best_R = R
            best_thetas = thetas.copy()
            best_phis = phis.copy()

        # Determine which constraint is active
        if wall_dist < pair_r:
            # Active constraint: Wall
            # Gradient of wall_dist w.r.t R is -1.
            # Gradient of wall_dist w.r.t angles is 0.
            # We want to maximize wall_dist, so decrease R.
            # But decreasing R increases pair_r (usually).
            # So we should decrease R.
            dR = -1.0
            dthetas = np.zeros(N)
            dphis = np.zeros(N)
        else:
            # Active constraint: Pair
            # We want to increase min_pair_dist.
            # Find the pair (i,j) with minimum distance.
            # There might be multiple.
            # Let's find the min distance pair.
            min_dist_val = min_pair_dist
            # Find indices
            # This is tricky to do efficiently in a loop without vectorization.
            # Let's just pick the first min pair.
            idx = np.unravel_index(np.argmin(dist[~np.eye(N, dtype=bool)]), (N, N-1))
            # This indexing is messy.
            
            # Let's just do a random perturbation / simulated annealing?
            # No, let's stick to gradient.
            
            # Let's compute gradient of min_pair_dist.
            # If only one pair is active, gradient is direction to separate them.
            # c_i - c_j vector.
            # Move c_i away from c_j, c_j away from c_i.
            # In spherical coords:
            # c_i = R u_i.
            # dc_i = dR u_i + R du_i.
            # du_i is tangent to sphere.
            
            # This is getting complex.
            # Let's use a simpler heuristic: 
            # If pair constraint is active, we want to maximize the minimum distance.
            # This is equivalent to maximizing the minimum angular separation for a given R.
            # For fixed R, maximizing min dist is maximizing min angle.
            # This is the spherical code problem.
            # We can use a repulsion force model.
            
            # Force on i from j: (u_i - u_j) / |u_i - u_j|^2
            # But we only care about the min distance pair?
            # No, we should consider all pairs close to the min.
            
            # Let's compute forces from all pairs with dist < min_dist + epsilon.
            eps = 0.01
            active_pairs = np.argwhere(dist < min_dist_val + eps)
            active_pairs = active_pairs[active_pairs[:, 0] < active_pairs[:, 1]]
            
            forces_u = np.zeros((N, 3)) # Forces in Cartesian space on unit vectors?
            # Actually, let's work in Cartesian space for the gradient, then project?
            # No, let's just compute the gradient of the distance in Cartesian space, 
            # then convert to spherical gradients.
            
            # Gradient of dist(i,j) w.r.t c_i is (c_i - c_j)/dist.
            # Gradient w.r.t c_j is (c_j - c_i)/dist.
            
            # We want to maximize min dist.
            # So we move in direction of sum of gradients of active distances.
            
            grad_c = np.zeros((N, 3))
            for i_idx, j_idx in active_pairs:
                v = centers[i_idx] - centers[j_idx]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-9:
                    v_norm = v / norm_v
                    grad_c[i_idx] += v_norm
                    grad_c[j_idx] -= v_norm
            
            # Now project grad_c onto the tangent space of the sphere at each point.
            # Tangent space is orthogonal to u_i.
            # u_i = centers[i] / R
            # grad_theta, grad_phi from grad_c?
            # This is a coordinate transform.
            
            # Alternatively, just do a line search in the direction of the projected gradient.
            
            # Let's simplify: 
            # Instead of complex gradient, let's do a "relaxation" step.
            # Move each point slightly away from its closest neighbor on the sphere.
            
            # For each i, find closest j.
            # Move i away from j.
            
            # Let's just use the Cartesian gradient and convert to angle changes.
            # c = R [sin(t)cos(p), sin(t)sin(p), cos(t)]
            # dc/dt = R [cos(t)cos(p), cos(t)sin(p), -sin(t)]
            # dc/dp = R [-sin(t)sin(p), sin(t)cos(p), 0]
            
            # We want dt, dp such that dc = dt * (dc/dt) + dp * (dc/dp) approximates grad_c.
            # This is a least squares problem for each i.
            # [dc/dt, dc/dp] is a 3x2 matrix.
            # We want to find [dt, dp] to project grad_c onto the span of these vectors.
            
            dthetas = np.zeros(N)
            dphis = np.zeros(N)
            
            for i_idx in range(N):
                t = thetas[i_idx]
                p = phis[i_idx]
                
                # Basis vectors
                v_t = R * np.array([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)])
                v_p = R * np.array([-np.sin(t)*np.sin(p), np.sin(t)*np.cos(p), 0])
                
                # If sin(t) is 0, v_p is 0. Handle poles.
                if np.abs(np.sin(t)) < 1e-6:
                    # At poles, phi is undefined. Just move in theta?
                    # Or just skip phi update.
                    # Project grad_c[i] onto v_t
                    dot_t = np.dot(grad_c[i_idx], v_t)
                    norm_t_sq = np.dot(v_t, v_t)
                    if norm_t_sq > 1e-9:
                        dthetas[i_idx] = dot_t / norm_t_sq
                    # dphis is 0
                else:
                    # Project grad_c[i] onto span(v_t, v_p)
                    # A = [v_t, v_p] (3x2)
                    # x = A^T (A A^T)^-1 grad_c ? No.
                    # x = (A^T A)^-1 A^T grad_c
                    A = np.column_stack((v_t, v_p))
                    A_t_A = A.T @ A
                    A_t_g = A.T @ grad_c[i_idx]
                    try:
                        x = np.linalg.solve(A_t_A, A_t_g)
                        dthetas[i_idx] = x[0]
                        dphis[i_idx] = x[1]
                    except:
                        pass
            
            # Also, if pair constraint is active, we might want to adjust R?
            # If we increase R, wall dist decreases.
            # If we decrease R, wall dist increases, but pair dist decreases (linearly with R).
            # Since pair is active, we are limited by pairs.
            # Changing R scales all distances.
            # If we increase R, pair dist increases, but wall dist decreases.
            # If wall_dist > pair_r, we can increase R until wall_dist == pair_r.
            # So if pair is active, we should increase R to reduce the "slack" in wall constraint?
            # No, if pair is active, r = pair_r.
            # r = min(0.5-R, 0.5*R*min_angle).
            # If we increase R, 0.5*R*min_angle increases, but 0.5-R decreases.
            # So r increases until they meet.
            # So if pair is active, we should increase R.
            dR = 1.0
            
            # Normalize gradients?
            # The magnitudes of dthetas, dphis, dR might be very different.
            # Let's just use step_size.
            
            R += step_size * dR
            thetas += step_size * dthetas
            phis += step_size * dphis
            
        # If wall was active, we decreased R.
        # If pair was active, we increased R and moved angles.
        
        # Clamp R
        R = np.clip(R, 0.0, 0.5)
        
        # Decay
        step_size *= 0.999
        
    # Final evaluation
    final_centers = get_centers(best_R, best_thetas, best_phis)
    return final_centers
