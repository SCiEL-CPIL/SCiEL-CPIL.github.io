import numpy as np
from scipy.optimize import minimize
import time

N = 300
SYMMETRY = 12
N_PER_SECTOR = N // SYMMETRY  # 25 circles per sector

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def compute_score(points):
        """Compute the actual packing score for full N points."""
        points = points.reshape(-1, 2)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists) / 2.0
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary = np.min(boundary_dists)
        return min(min_pair_dist, min_boundary)

    def expand_symmetry(sector_points):
        """
        Expand a single sector's points to full 12-fold symmetric configuration.
        sector_points: array of shape (N_PER_SECTOR, 2) in polar coordinates (r, theta)
        where theta is in [0, 2*pi/12]
        Returns: array of shape (N, 2) in Cartesian coordinates
        """
        all_points = []
        sector_angle = 2 * np.pi / SYMMETRY
        for k in range(SYMMETRY):
            base_angle = k * sector_angle
            for pt in sector_points:
                r, theta = pt[0], pt[1]
                # Mirror within sector for better symmetry
                # Primary: rotate
                angle1 = base_angle + theta
                x1 = r * np.cos(angle1)
                y1 = r * np.sin(angle1)
                all_points.append([x1, y1])
                # Mirror: reflect across sector bisector
                angle2 = base_angle + sector_angle - theta
                x2 = r * np.cos(angle2)
                y2 = r * np.sin(angle2)
                all_points.append([x2, y2])
        
        return np.array(all_points[:N])

    def compute_score_symmetric(sector_points):
        """Compute score using symmetry expansion."""
        full_points = expand_symmetry(sector_points)
        return compute_score(full_points)

    # Initialize with hexagonal lattice, then extract one sector
    def generate_hex_grid(N_target, r_init):
        spacing = 2.0 * r_init
        R_est = spacing * np.sqrt(N_target * np.sqrt(3.0) / (2.0 * np.pi))
        R_est *= 1.1
        
        k = int(np.ceil(R_est / spacing)) + 2
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            if abs(y) > R_est:
                continue
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if np.hypot(x, y) <= R_est:
                    points.append((x, y))
        
        points = np.array(points)
        if len(points) == 0:
            return np.zeros((N_target, 2))
            
        if len(points) > N_target:
            radii = np.hypot(points[:, 0], points[:, 1])
            indices = np.argsort(radii)
            points = points[indices[:N_target]]
        elif len(points) < N_target:
            while len(points) < N_target:
                angle = np.random.uniform(0, 2*np.pi)
                rad = np.random.uniform(0, 1)
                points = np.vstack([points, [rad*np.cos(angle), rad*np.sin(angle)]])
        
        return points

    r_guess = 0.05
    hex_points = generate_hex_grid(N, r_guess)
    
    max_rad = np.max(np.hypot(hex_points[:, 0], hex_points[:, 1]))
    if max_rad > 1.0:
        scale = 0.99 / max_rad
        hex_points *= scale

    # Convert hex points to polar and select one sector
    def cartesian_to_polar(points):
        r = np.linalg.norm(points, axis=1)
        theta = np.arctan2(points[:, 1], points[:, 0])
        theta = np.mod(theta, 2 * np.pi)
        return np.column_stack([r, theta])

    hex_polar = cartesian_to_polar(hex_points)
    sector_angle = 2 * np.pi / SYMMETRY
    
    # Select points in the first sector [0, sector_angle]
    sector_mask = (hex_polar[:, 1] >= 0) & (hex_polar[:, 1] <= sector_angle)
    sector_points_cart = hex_points[sector_mask]
    
    if len(sector_points_cart) < N_PER_SECTOR:
        # Add more points if needed
        while len(sector_points_cart) < N_PER_SECTOR:
            angle = np.random.uniform(0, sector_angle)
            rad = np.random.uniform(0, 0.95)
            sector_points_cart = np.vstack([sector_points_cart, [rad*np.cos(angle), rad*np.sin(angle)]])
    elif len(sector_points_cart) > N_PER_SECTOR:
        # Keep the innermost points
        radii = np.linalg.norm(sector_points_cart, axis=1)
        indices = np.argsort(radii)
        sector_points_cart = sector_points_cart[indices[:N_PER_SECTOR]]
    
    # Convert to polar coordinates (r, theta) for optimization
    sector_polar = cartesian_to_polar(sector_points_cart)
    # Ensure theta is in [0, sector_angle]
    sector_polar[:, 1] = np.mod(sector_polar[:, 1], sector_angle)
    
    best_sector = sector_polar.copy()
    best_score = compute_score_symmetric(best_sector)
    best_full_points = expand_symmetry(best_sector)

    # Precompute pair indices within and across sectors for efficient computation
    # For the symmetric case, we need to compute distances between all pairs
    # But we can exploit symmetry: only compute for one sector and its neighbors
    
    def make_objective(beta):
        def objective(x):
            sector = x.reshape(N_PER_SECTOR, 2)
            full_points = expand_symmetry(sector)
            
            diff = full_points[:, np.newaxis, :] - full_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists) / 2.0
            boundary_dists = 1.0 - np.linalg.norm(full_points, axis=1)
            min_boundary = np.min(boundary_dists)
            
            all_vals = np.concatenate([np.full(1, min_pair_dist), np.full(1, min_boundary)])
            m = np.min(all_vals)
            
            if m < 0:
                return -m * 10.0
            
            # Use soft-min over a subset of critical constraints
            # For efficiency, use the minimum and a few nearby values
            dists_flat = dists[~np.eye(N, dtype=bool)].ravel() / 2.0
            boundary_flat = boundary_dists.ravel()
            all_constr = np.concatenate([dists_flat, boundary_flat])
            
            # Use top-k smallest for soft-min to keep it manageable
            k = min(100, len(all_constr))
            idx = np.argpartition(all_constr, k)[:k]
            critical = all_constr[idx]
            
            m_c = np.min(critical)
            exp_vals = np.exp(-beta * (critical - m_c))
            soft_min = m_c - (1.0/beta) * np.log(np.sum(exp_vals))
            return -soft_min
        return objective

    def make_gradient(beta):
        def objective_grad(x):
            sector = x.reshape(N_PER_SECTOR, 2)
            full_points = expand_symmetry(sector)
            
            diff = full_points[:, np.newaxis, :] - full_points[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            boundary_dists = 1.0 - np.linalg.norm(full_points, axis=1)
            
            dists_flat = dists[~np.eye(N, dtype=bool)].ravel() / 2.0
            boundary_flat = boundary_dists.ravel()
            all_constr = np.concatenate([dists_flat, boundary_flat])
            
            k = min(100, len(all_constr))
            idx = np.argpartition(all_constr, k)[:k]
            critical = all_constr[idx]
            
            m_c = np.min(critical)
            exp_vals = np.exp(-beta * (critical - m_c))
            S = np.sum(exp_vals)
            weights = exp_vals / S
            
            # Compute gradient with respect to full_points
            grad_full = np.zeros_like(full_points)
            
            # Boundary gradient
            norms = np.linalg.norm(full_points, axis=1)
            norms[norms == 0] = 1.0
            w_bounds = weights[N*(N-1):]  # boundary weights
            grad_full += -full_points / norms[:, np.newaxis] * w_bounds[:, np.newaxis]
            
            # Distance gradient
            w_dists = weights[:N*(N-1)]
            # Reshape weights to matrix
            W = np.zeros((N, N))
            mask = ~np.eye(N, dtype=bool)
            W[mask] = w_dists
            W = (W + W.T) / 2.0  # Symmetrize
            
            inv_dists = 1.0 / (2.0 * dists)
            np.fill_diagonal(inv_dists, 0.0)
            
            coeff = W * inv_dists
            grad_full += np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
            
            # Now map gradient back to sector parameters
            # This requires the Jacobian of expand_symmetry
            grad_sector = np.zeros_like(sector)
            
            for k_sym in range(SYMMETRY):
                base_angle = k_sym * sector_angle
                for i in range(N_PER_SECTOR):
                    r, theta = sector[i]
                    # Primary point
                    angle1 = base_angle + theta
                    # d/dtheta of (r*cos(angle), r*sin(angle))
                    d1 = np.array([-r * np.sin(angle1), r * np.cos(angle1)])
                    # d/dr
                    d2 = np.array([np.cos(angle1), np.sin(angle1)])
                    
                    # Mirror point
                    angle2 = base_angle + sector_angle - theta
                    d1_m = np.array([r * np.sin(angle2), r * np.cos(angle2)])
                    d2_m = np.array([np.cos(angle2), np.sin(angle2)])
                    
                    # Map indices
                    idx1 = k_sym * 2 * N_PER_SECTOR + i
                    idx2 = idx1 + 1
                    
                    grad_sector[i, 1] += np.dot(grad_full[idx1], d1) + np.dot(grad_full[idx2], d1_m)
                    grad_sector[i, 0] += np.dot(grad_full[idx1], d2) + np.dot(grad_full[idx2], d2_m)
            
            return grad_sector.ravel()
        return objective_grad

    # Optimize in the reduced 50-dimensional space
    bounds_sector = [(0.0, 1.0), (0.0, 2*np.pi/SYMMETRY)] * N_PER_SECTOR
    
    beta_schedule = [(20.0, 150), (50.0, 100), (100.0, 80), (200.0, 60), (500.0, 50)]
    
    for beta, max_iter in beta_schedule:
        if time.time() > deadline - 20:
            break
            
        obj_func = make_objective(beta)
        grad_func = make_gradient(beta)
        
        try:
            res = minimize(obj_func, best_sector.flatten(), jac=grad_func,
                          method='L-BFGS-B', bounds=bounds_sector,
                          options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False})
            
            if res.x is not None:
                candidate_sector = res.x.reshape(N_PER_SECTOR, 2)
                # Validate
                if np.all(np.isfinite(candidate_sector)) and np.all(candidate_sector[:, 0] >= 0) and np.all(candidate_sector[:, 0] <= 1.0):
                    score = compute_score_symmetric(candidate_sector)
                    if score > best_score:
                        best_sector = candidate_sector
                        best_score = score
                        best_full_points = expand_symmetry(best_sector)
                        
        except Exception as e:
            pass

    # Final polish with high beta
    if time.time() < deadline - 10:
        beta_final = 1000.0
        obj_final = make_objective(beta_final)
        grad_final = make_gradient(beta_final)
        
        try:
            res_final = minimize(obj_final, best_sector.flatten(), jac=grad_final,
                                method='L-BFGS-B', bounds=bounds_sector,
                                options={'maxiter': 100, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False})
            
            if res_final.x is not None:
                candidate_sector = res_final.x.reshape(N_PER_SECTOR, 2)
                if np.all(np.isfinite(candidate_sector)) and np.all(candidate_sector[:, 0] >= 0) and np.all(candidate_sector[:, 0] <= 1.0):
                    score = compute_score_symmetric(candidate_sector)
                    if score > best_score:
                        best_sector = candidate_sector
                        best_score = score
                        best_full_points = expand_symmetry(best_sector)
                        
        except Exception as e:
            pass

    # Final validity check
    full_points = expand_symmetry(best_sector)
    radii = np.hypot(full_points[:, 0], full_points[:, 1])
    if np.any(radii >= 1.0):
        max_r = np.max(radii)
        if max_r > 0:
            scale = 0.9999 / max_r
            full_points *= scale

    return full_points
