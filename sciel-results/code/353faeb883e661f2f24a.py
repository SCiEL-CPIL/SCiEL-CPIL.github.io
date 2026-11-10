import numpy as np
import time
import scipy.spatial.distance
import scipy.optimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Maximizes the min/max distance ratio for 16 points in 2D.
    
    Strategy:
    1. Bisection search on the minimum distance r (normalized such that max distance is 1).
    2. For a fixed r, solve the convex feasibility problem:
       Minimize D (max pairwise distance)
       Subject to: ||p_i - p_j|| >= r for all i < j
                   ||p_i - p_j|| <= D for all i < j
    3. This is solved via Sequential Convex Programming (SCP).
       The constraints ||p_i - p_j|| >= r are non-convex, but we linearize them
       using the current point estimate:
       (p_i - p_j)^T (p_i^{k} - p_j^{k}) >= r * ||p_i^{k} - p_j^{k}||
       The constraints ||p_i - p_j|| <= D are also non-convex (intersection of disks),
       but can be handled by minimizing the max of convex functions ||p_i - p_j||,
       which is a convex optimization problem if we treat D as a variable and 
       use epigraph form, OR we can just maximize the min distance directly?
       
       Actually, a cleaner SCP approach for "maximize min distance subject to max distance <= 1" is:
       Maximize r
       Subject to: ||p_i - p_j|| >= r  (Non-convex)
                   ||p_i - p_j|| <= 1   (Non-convex, intersection of disks)
       
       Let's stick to the prompt's suggested framing:
       "minimize the maximum pairwise distance subject to a hard lower-bound constraint on all pairwise distances"
       
       So for a fixed r:
       Minimize D
       Subject to: ||p_i - p_j|| >= r
       
       To make this convex, we linearize the lower bound constraints around the current point.
       The upper bound (minimize max distance) is naturally handled by minimizing the convex function max(||p_i - p_j||).
       Wait, max of convex functions is convex. So minimizing max(||p_i - p_j||) is a convex problem.
       The only non-convex part is ||p_i - p_j|| >= r.
       
       SCP Iteration:
       1. Start with a feasible point (or random).
       2. Linearize ||p_i - p_j|| >= r at current points p^k:
          (p_i - p_j)^T (p_i^k - p_j^k) >= r * ||p_i^k - p_j^k||
          This is a linear constraint.
       3. Solve the convex QP:
          Minimize D
          Subject to:
             ||p_i - p_j|| <= D  for all i<j  (Convex constraints)
             (p_i - p_j)^T (p_i^k - p_j^k) >= r * ||p_i^k - p_j^k|| for all i<j (Linear constraints)
       
       If the optimal D <= 1 + epsilon, then r is feasible.
       If D > 1, then r is not feasible (we couldn't pack them within diameter 1).
       
       We bisect on r to find the max feasible r.
    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 30  # 570 seconds to be safe

    # Pre-compute indices for all pairs
    pair_indices = np.triu_indices(n, k=1)
    num_pairs = len(pair_indices[0])
    i_idx, j_idx = pair_indices
    
    def get_pairwise_distances(points):
        return np.linalg.norm(points[i_idx] - points[j_idx], axis=1)

    def solve_scp(r_target, init_points, max_iters=50):
        """
        Solves the SCP subproblem for a fixed r_target.
        Returns (D_opt, points_opt, converged)
        """
        pts = init_points.copy()
        # Normalize to ensure we start somewhat reasonable, though SCP should handle it
        dists = get_pairwise_distances(pts)
        d_max_init = np.max(dists)
        if d_max_init > 0:
            pts /= d_max_init
        
        # If initial min dist is already less than r_target, we might be infeasible or need to move
        # But SCP will try to fix it.
        
        D_prev = np.inf
        for k in range(max_iters):
            if time.time() > deadline:
                return D_prev, pts, False
            
            # Calculate linearization terms
            # v_ij = p_i^k - p_j^k
            # norm_v_ij = ||v_ij||
            v = pts[i_idx] - pts[j_idx]
            norm_v = np.linalg.norm(v, axis=1)
            
            # Handle zero distance (coincident points) in linearization
            # If norm_v is 0, the gradient of distance is undefined.
            # We can add a small random perturbation or skip the constraint?
            # If points are coincident and r > 0, they are infeasible.
            # Let's add a small epsilon to norm to avoid division by zero in direction,
            # but the constraint itself (p_i - p_j)^T v >= r * norm_v becomes 0 >= 0 if v=0.
            # This doesn't help separate them.
            # Better: if any norm_v < 1e-6, perturb points slightly.
            mask_zero = norm_v < 1e-6
            if np.any(mask_zero):
                # Perturb coincident points
                for idx in np.where(mask_zero)[0]:
                    i, j = i_idx[idx], j_idx[idx]
                    noise = np.random.randn(2) * 0.01
                    pts[i] += noise
                    pts[j] -= noise
                # Recalculate v and norm_v
                v = pts[i_idx] - pts[j_idx]
                norm_v = np.linalg.norm(v, axis=1)
            
            # Build A and b for linear constraints:
            # (p_i - p_j)^T v_ij >= r * norm_v_ij
            # p_i^T v_ij - p_j^T v_ij >= r * norm_v_ij
            # In standard form A x <= b, this is:
            # -p_i^T v_ij + p_j^T v_ij <= -r * norm_v_ij
            # Or: p_j^T v_ij - p_i^T v_ij <= -r * norm_v_ij
            
            A_lin = np.zeros((num_pairs, n * d))
            b_lin = np.zeros(num_pairs)
            
            for idx in range(num_pairs):
                i, j = i_idx[idx], j_idx[idx]
                v_ij = v[idx]
                rhs = -r_target * norm_v[idx]
                
                # Coeff for p_i: -v_ij
                A_lin[idx, i*d : (i+1)*d] = -v_ij
                # Coeff for p_j: v_ij
                A_lin[idx, j*d : (j+1)*d] = v_ij
                b_lin[idx] = rhs
            
            # Objective: Minimize D
            # Subject to:
            # 1. Linear constraints from SCP
            # 2. Convex constraints: ||p_i - p_j|| <= D
            #    This is equivalent to:
            #    p_i^2 + p_j^2 - 2 p_i^T p_j <= D^2
            #    This is a quadratic constraint. scipy.optimize.minimize doesn't handle QP natively with SLSQP easily for many constraints?
            #    SLSQP can handle quadratic constraints? No, SLSQP handles linear and quadratic objectives, but constraints are generally linear or quadratic?
            #    Actually, SLSQP supports 'eq' and 'ineq' constraints which are scalar functions.
            #    We can define the constraints as functions.
            
            def obj(x):
                # x is [p_0...p_15, D]
                # We want to minimize D
                return x[-1]
            
            # Constraints for SLSQP:
            # 1. Linear constraints: A_lin x_lin + b_lin <= 0 (where x_lin is x[:-1])
            #    SLSQP constraint format: {'type': 'ineq', 'fun': lambda x: A_lin @ x[:-1] + b_lin}
            #    Wait, SLSQP requires fun(x) >= 0 for ineq.
            #    So we need: -A_lin @ x[:-1] - b_lin >= 0
            
            def cons_linear(x):
                return -A_lin @ x[:-1] - b_lin
            
            # 2. Distance constraints: D^2 - ||p_i - p_j||^2 >= 0
            #    D^2 - (p_i - p_j)^T (p_i - p_j) >= 0
            
            def cons_dist(x):
                p = x[:-1].reshape(n, d)
                D = x[-1]
                dists_sq = np.sum((p[i_idx] - p[j_idx])**2, axis=1)
                return D**2 - dists_sq
            
            # Initial guess for x
            x0 = np.concatenate([pts.flatten(), [np.max(get_pairwise_distances(pts))]])
            
            # Bounds: D should be positive. Points can be anywhere, but let's bound them to a box to help convergence?
            # Maybe [-10, 10] is safe.
            bounds = [(-10, 10)] * (n * d) + [(0, 10)]
            
            try:
                res = scipy.optimize.minimize(
                    obj,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=[
                        {'type': 'ineq', 'fun': cons_linear},
                        {'type': 'ineq', 'fun': cons_dist}
                    ],
                    options={'maxiter': 200, 'ftol': 1e-9}
                )
                
                if not res.success:
                    # If SLSQP fails, return previous best
                    return D_prev, pts, False
                
                D_new = res.x[-1]
                pts_new = res.x[:-1].reshape(n, d)
                
                # Check convergence
                if abs(D_new - D_prev) < 1e-6:
                    return D_new, pts_new, True
                
                D_prev = D_new
                pts = pts_new
                
            except Exception as e:
                # If optimization fails, break
                return D_prev, pts, False
        
        return D_prev, pts, False

    # Generate initial candidates
    candidates = []
    
    # 1. Hexagonal lattice
    x_coords = []
    y_coords = []
    spacing = 1.0
    for row in range(4):
        for col in range(4):
            x = col * spacing
            if row % 2 == 1:
                x += 0.5 * spacing
            y = row * spacing * np.sqrt(3) / 2
            x_coords.append(x)
            y_coords.append(y)
    hex_pts = np.column_stack((x_coords, y_coords))
    center = np.mean(hex_pts, axis=0)
    hex_pts -= center
    dists_hex = get_pairwise_distances(hex_pts)
    hex_pts /= np.max(dists_hex)
    candidates.append(hex_pts)
    
    # 2. Circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(circle_pts)
    
    # 3. Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists_grid = get_pairwise_distances(grid_pts)
    grid_pts /= np.max(dists_grid)
    candidates.append(grid_pts)

    # We will run the bisection search on the best candidate from a quick preliminary check
    # Or we can run bisection on each candidate and keep the best r.
    
    best_r = 0.0
    best_points = candidates[0]
    
    for init_pts in candidates:
        if time.time() > deadline - 60:
            break
            
        # Bisection search for r
        r_low = 0.0
        r_high = 1.0  # Max possible min dist is 1 (if all dists are 1, which is impossible for 16 pts in 2D, but safe upper bound)
        
        # Actually, a better upper bound is the min dist of the initial config, or just start high and let it fail.
        # Let's start with r_high = 0.6 (a reasonable guess for 16 pts)
        
        best_r_for_init = 0.0
        best_pts_for_init = init_pts.copy()
        
        # Run bisection for a fixed number of steps or until convergence
        for _ in range(20): # 20 steps gives precision ~ 1/2^20 which is plenty
            if time.time() > deadline - 30:
                break
                
            r_mid = (r_low + r_high) / 2.0
            
            D_opt, pts_opt, converged = solve_scp(r_mid, init_pts, max_iters=10)
            
            # If D_opt <= 1.0, then r_mid is feasible
            if D_opt <= 1.0 + 1e-6:
                r_low = r_mid
                best_r_for_init = r_mid
                best_pts_for_init = pts_opt.copy()
            else:
                r_high = r_mid
        
        if best_r_for_init > best_r:
            best_r = best_r_for_init
            best_points = best_pts_for_init
            
        # If we found a good r, we might try to refine it further with more SCP iterations
        # But the bisection loop already does that by calling solve_scp.
        
    # Final refinement: Run SCP one more time on the best point found with the best r
    if best_r > 0:
        D_final, pts_final, _ = solve_scp(best_r, best_points, max_iters=50)
        if D_final <= 1.0 + 1e-6:
            best_points = pts_final
    
    # Ensure shape is (16, 2)
    return best_points.reshape(16, 2)

if __name__ == "__main__":
    pts = min_max_dist_dim2_16()
    print(pts.shape)
