import numpy as np
import time
import scipy.optimize
import scipy.spatial.distance

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = 600 - 10  # 590 seconds to be safe

    def get_score(points):
        """Calculate the actual score R = (d_min/d_max)^2"""
        pts = points.reshape(n, d)
        dists = scipy.spatial.distance.pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    # Precompute indices for all pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pair_indices = np.array(pairs, dtype=np.int32)
    n_pairs = len(pair_indices)

    def rigid_body_energy(x, r_min, r_max, k_rep=1000.0, k_att=100.0):
        """
        Rigid-body mechanics simulation energy.
        Points are repulsive particles.
        Energy penalizes distances < r_min (repulsion) and > r_max (attraction).
        This approximates hard constraints with steep power-law potentials.
        """
        pts = x.reshape(n, d)
        # Vectorized distance calculation
        # diff[i, j, :] = pts[i] - pts[j]
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Mask out self-distances
        np.fill_diagonal(dists, np.inf)
        
        # We only care about upper triangle for energy, but symmetry is fine
        # Let's just sum over all pairs i < j to avoid double counting and self terms
        # Using pdist for efficiency if needed, but manual is fine for n=16
        # Actually, let's stick to the pair_indices loop for clarity and gradient computation
        
        energy = 0.0
        grad = np.zeros_like(x)
        
        for k in range(n_pairs):
            i, j = pair_indices[k]
            d_ij = np.linalg.norm(pts[i] - pts[j])
            
            if d_ij < 1e-12:
                # Singularity handling: strong repulsion
                energy += k_rep * 1e6
                # Random kick or zero gradient? Zero gradient is safer for stability
                continue
            
            if d_ij < r_min:
                # Repulsive force: pushes apart
                # Potential: k_rep * (r_min - d_ij)^2
                # Force: -2 * k_rep * (r_min - d_ij) * (x_i - x_j) / d_ij
                # Gradient of potential w.r.t x_i: 2 * k_rep * (r_min - d_ij) * (x_i - x_j) / d_ij
                # Wait, potential V = k (r_min - d)^2. dV/dx = 2k(r_min-d) * d(r_min-d)/dx
                # d(r_min-d)/dx_i = - (x_i - x_j)/d
                # So dV/dx_i = -2k(r_min-d) (x_i - x_j)/d
                # We want to minimize V, so we move in direction -grad V
                # grad V = -2k(r_min-d) (x_i - x_j)/d
                # So force on i is 2k(r_min-d) (x_i - x_j)/d ? No.
                # Let's just compute gradient of energy.
                # E_rep = k_rep * (r_min - d_ij)^2
                # dE_rep/dx_i = 2 * k_rep * (r_min - d_ij) * (-1) * (x_i - x_j) / d_ij
                # dE_rep/dx_j = 2 * k_rep * (r_min - d_ij) * (1) * (x_i - x_j) / d_ij
                coeff = 2.0 * k_rep * (r_min - d_ij) / d_ij
                vec = pts[i] - pts[j]
                grad[i*d:(i+1)*d] -= coeff * vec
                grad[j*d:(j+1)*d] += coeff * vec
                energy += k_rep * (r_min - d_ij)**2
            elif d_ij > r_max:
                # Attractive force: pulls together
                # Potential: k_att * (d_ij - r_max)^2
                # dE_att/dx_i = 2 * k_att * (d_ij - r_max) * (x_i - x_j) / d_ij
                # dE_att/dx_j = 2 * k_att * (d_ij - r_max) * (x_j - x_i) / d_ij
                coeff = 2.0 * k_att * (d_ij - r_max) / d_ij
                vec = pts[i] - pts[j]
                grad[i*d:(i+1)*d] += coeff * vec
                grad[j*d:(j+1)*d] -= coeff * vec
                energy += k_att * (d_ij - r_max)**2
            else:
                # In the well [r_min, r_max], no force
                pass
                
        return energy, grad

    def rigid_body_relax(initial_points, r_min, r_max, max_iter=1000, tol=1e-12):
        """
        Perform gradient-based relaxation on the rigid-body energy landscape.
        """
        x = initial_points.flatten().copy()
        
        # Simple gradient descent with adaptive step size
        step_size = 0.01
        prev_energy = np.inf
        
        for i in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            energy, grad = rigid_body_energy(x, r_min, r_max)
            
            if np.abs(prev_energy - energy) < tol:
                break
            prev_energy = energy
            
            # Adaptive step size
            # Try to take a step
            new_x = x - step_size * grad
            new_energy, _ = rigid_body_energy(new_x, r_min, r_max)
            
            if new_energy < energy:
                x = new_x
                # Increase step size slightly if improving significantly
                if energy - new_energy > 1e-6:
                    step_size *= 1.1
            else:
                # Decrease step size
                step_size *= 0.5
                
            if step_size < 1e-10:
                break
                
        return x.reshape(n, d)

    # Generate initial candidates
    candidates = []
    
    # 1. 4x4 Grid
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5, 1.5, 4), np.linspace(-1.5, 1.5, 4))
    grid_pts = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    dists = scipy.spatial.distance.pdist(grid_pts)
    grid_pts = grid_pts / np.max(dists)
    candidates.append(grid_pts)
    
    # 2. Regular 16-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    circle_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    circle_pts = circle_pts / 2.0
    candidates.append(circle_pts)
    
    # 3. Hexagonal packing approximation
    hex_pts = grid_pts.copy()
    for i in range(1, 4, 2):
        hex_pts[i*4:(i+1)*4, 0] += 0.5
    dists = scipy.spatial.distance.pdist(hex_pts)
    hex_pts = hex_pts / np.max(dists)
    candidates.append(hex_pts)
    
    # 4. Random
    np.random.seed(42)
    rand_pts = np.random.uniform(-1, 1, (n, d))
    dists = scipy.spatial.distance.pdist(rand_pts)
    rand_pts = rand_pts / np.max(dists)
    candidates.append(rand_pts)
    
    best_points = None
    best_score = -np.inf
    
    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_points = c.copy()
            
    # Bisection search for the optimal ratio using rigid-body relaxation
    # We search for the maximum r = d_min / d_max such that a configuration exists
    # with d_min >= r * d_max.
    # We normalize d_max to 1, so we search for max d_min.
    
    low_r = 0.0
    high_r = 1.0
    
    for c in candidates:
        if time.time() > deadline - 10:
            break
            
        c_norm = c.copy()
        d_max_init = np.max(scipy.spatial.distance.pdist(c_norm))
        if d_max_init > 0:
            c_norm = c_norm / d_max_init
        else:
            continue
            
        low_r = 0.0
        high_r = 1.0
        
        for _ in range(15): # 15 iterations of bisection
            if time.time() > deadline - 10:
                break
                
            mid_r = (low_r + high_r) / 2.0
            
            # Try to find a configuration with d_min >= mid_r and d_max <= 1.0
            # Using rigid body relaxation with r_min = mid_r, r_max = 1.0
            relaxed_pts = rigid_body_relax(c_norm, mid_r, 1.0, max_iter=200)
            
            dists = scipy.spatial.distance.pdist(relaxed_pts)
            d_min_found = np.min(dists)
            d_max_found = np.max(dists)
            
            # Normalize to d_max = 1 for consistent comparison
            if d_max_found > 0:
                relaxed_pts_norm = relaxed_pts / d_max_found
                d_min_norm = d_min_found / d_max_found
            else:
                d_min_norm = 0.0
                relaxed_pts_norm = relaxed_pts
                
            # Check feasibility: d_min >= mid_r
            if d_min_norm >= mid_r - 1e-5:
                low_r = mid_r
                s = get_score(relaxed_pts_norm)
                if s > best_score:
                    best_score = s
                    best_points = relaxed_pts_norm.copy()
            else:
                high_r = mid_r
                
    # Final local refinement on the best points found using standard optimization
    if best_points is not None and time.time() < deadline - 5:
        try:
            def smoothed_objective(x, beta=50.0):
                points = x.reshape(n, d)
                dists = scipy.spatial.distance.pdist(points)
                max_d = np.max(dists)
                term_min = np.exp(-beta * (dists - max_d))
                d_min_soft = max_d - (1.0 / beta) * np.log(np.sum(term_min))
                term_max = np.exp(beta * (dists - max_d))
                d_max_soft = max_d + (1.0 / beta) * np.log(np.sum(term_max))
                if d_max_soft == 0:
                    return 0.0
                return -(d_min_soft / d_max_soft)
                
            res2 = scipy.optimize.minimize(
                smoothed_objective, 
                best_points.flatten(), 
                args=(50.0,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12}
            )
            if res2.success:
                polished_pts = res2.x.reshape(n, d)
                polished_score = get_score(polished_pts)
                if polished_score > best_score:
                    best_score = s
                    best_points = polished_pts.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = candidates[0]
        
    return best_points
