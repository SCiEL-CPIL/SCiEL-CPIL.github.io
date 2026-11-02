import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def generate_hex_grid(spacing, rotation=0.0):
        """Generate hexagonal lattice points inside unit circle with optional rotation."""
        points = []
        max_j = int(2.0 / (spacing * np.sqrt(3.0))) + 3
        max_i = int(2.0 / spacing) + 3
        
        for j in range(-max_j, max_j + 1):
            for i in range(-max_i, max_i + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if rotation != 0.0:
                    xr = x * np.cos(rotation) - y * np.sin(rotation)
                    yr = x * np.sin(rotation) + y * np.cos(rotation)
                else:
                    xr, yr = x, y
                if xr * xr + yr * yr <= 1.0:
                    points.append([xr, yr])
        return np.array(points)

    def compute_score(pts):
        """Compute the packing score (minimum radius)."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_bound = np.min(1.0 - radii)
        
        return min(min_pair / 2.0, min_bound)

    def soft_min_and_grad(x, beta):
        """Compute soft-min and its gradient."""
        x_max = np.max(x)
        exp_neg_beta_x = np.exp(-beta * (x - x_max))
        sum_exp = np.sum(exp_neg_beta_x)
        s_val = -1.0 / beta * (np.log(sum_exp) - beta * x_max)
        grad = exp_neg_beta_x / sum_exp
        return s_val, grad

    def objective_and_grad(flat_pts, beta):
        """Objective function and gradient for L-BFGS-B."""
        pts = flat_pts.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d_matrix = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(d_matrix, np.inf)
        
        # Upper triangle mask
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_flat = d_matrix[mask]
        
        # Boundary distances
        radii = np.hypot(pts[:, 0], pts[:, 1])
        d_bound = 1.0 - radii
        
        # Effective radii: pairwise/2 and boundary
        eff_rad = np.concatenate([d_flat / 2.0, d_bound])
        
        # Soft min
        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        obj = -s_val
        
        # Backpropagate gradient
        grad_pair = grad_eff[:len(d_flat)]
        grad_bound = grad_eff[len(d_flat):]
        
        # Build symmetric gradient matrix
        G = np.zeros((N, N))
        G[mask] = grad_pair
        G.T[mask] = grad_pair
        
        safe_d = np.where(d_matrix < 1e-9, 1e-9, d_matrix)
        weight = G / safe_d
        F_pair = weight @ diff
        
        # Boundary gradient
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        F_bound = -grad_bound[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        
        total_grad = F_pair + F_bound
        
        return obj, total_grad.flatten()

    def local_polish(pts, max_iters=200, tol=1e-10):
        """Local polish using projected gradient on exact minimum."""
        pts = pts.copy()
        for _ in range(max_iters):
            if time.time() > deadline - 3.0:
                break
            
            # Compute current score
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists)
            
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_bound = np.min(1.0 - radii)
            
            current_score = min(min_pair / 2.0, min_bound)
            
            # Find which constraint is active
            if min_pair / 2.0 <= min_bound:
                # Pairwise constraint active
                # Find all pairs at minimum distance
                close_pairs = np.argwhere(dists < min_pair + 1e-8)
                i_idx = close_pairs[:, 0]
                j_idx = close_pairs[:, 1]
                
                # Gradient: push apart points in close pairs
                grad = np.zeros_like(pts)
                for i, j in zip(i_idx, j_idx):
                    if i == j:
                        continue
                    d = pts[i] - pts[j]
                    norm_d = np.linalg.norm(d)
                    if norm_d > 1e-10:
                        d_unit = d / norm_d
                        grad[i] += d_unit
                        grad[j] -= d_unit
            else:
                # Boundary constraint active
                close_boundary = np.where(radii > 1.0 - min_bound - 1e-8)[0]
                grad = np.zeros_like(pts)
                for i in close_boundary:
                    r = radii[i]
                    if r > 1e-10:
                        grad[i] -= pts[i] / r
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-15:
                break
            grad = grad / grad_norm
            
            # Try small steps
            improved = False
            for step in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
                new_pts = pts + step * grad
                # Project back inside unit circle
                new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out = new_radii > 1.0
                if np.any(out):
                    new_pts[out] *= (0.99999 / new_radii[out])[:, np.newaxis]
                
                new_score = compute_score(new_pts)
                if new_score > current_score + 1e-12:
                    pts = new_pts
                    improved = True
                    break
            
            if not improved:
                break
        
        return pts

    # 1. Initialization
    best_init = None
    best_init_score = -np.inf
    
    spacings = [0.10, 0.105, 0.106, 0.107, 0.108, 0.11, 0.112, 0.115]
    rotations = [0.0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 8, np.pi / 5]
    
    for sp in spacings:
        for rot in rotations:
            pts = generate_hex_grid(sp, rot)
            if len(pts) < N:
                continue
            
            dists_to_center = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
            idx = np.argsort(dists_to_center)[:N]
            init_pts = pts[idx].copy()
            
            score = compute_score(init_pts)
            if score > best_init_score:
                best_init_score = score
                best_init = init_pts
    
    if best_init is None:
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9
        best_init_score = compute_score(best_init)

    current_points = best_init.copy()
    current_best_score = best_init_score
    current_best_points = best_init.copy()

    # 2. Optimization with annealed beta schedule
    max_time = deadline - start_time - 10.0
    beta_start = 50.0
    beta_end = 50000.0
    
    num_rounds = 10
    time_per_round = max_time / num_rounds
    
    for round_idx in range(num_rounds):
        if time.time() > deadline - 10.0:
            break
            
        t = round_idx / max(1, num_rounds - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        flat_pts = current_points.flatten()
        round_deadline = time.time() + time_per_round * 0.9
        
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 1500,
                    'maxfun': 3000,
                    'ftol': 1e-14,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            
            if result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_of_bounds = radii > 1.0
                if np.any(out_of_bounds):
                    new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                
                new_score = compute_score(new_pts)
                
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
                    current_points = new_pts.copy()
                    
        except Exception:
            pass

    # 3. Multiple polish passes with increasing precision
    for polish_iter in range(5):
        if time.time() > deadline - 5.0:
            break
        
        current_points = local_polish(current_points, max_iters=300, tol=1e-12)
        new_score = compute_score(current_points)
        
        if new_score > current_best_score:
            current_best_score = new_score
            current_best_points = current_points.copy()
        
        # Also try another L-BFGS-B pass with very high beta
        if time.time() < deadline - 5.0:
            beta_final = 100000.0
            flat_pts = current_points.flatten()
            try:
                result = minimize(
                    objective_and_grad,
                    flat_pts,
                    args=(beta_final,),
                    method='L-BFGS-B',
                    jac=True,
                    options={
                        'maxiter': 800,
                        'maxfun': 1500,
                        'ftol': 1e-15,
                        'gtol': 1e-15,
                        'maxls': 50
                    }
                )
                
                if result.fun < 0:
                    new_pts = result.x.reshape(N, 2)
                    radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                    out_of_bounds = radii > 1.0
                    if np.any(out_of_bounds):
                        new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                    
                    new_score = compute_score(new_pts)
                    
                    if new_score > current_best_score:
                        current_best_score = new_score
                        current_best_points = new_pts.copy()
                        current_points = new_pts.copy()
                        
            except Exception:
                pass

    # 4. Final polish
    if time.time() < deadline - 2.0:
        current_best_points = local_polish(current_best_points, max_iters=500, tol=1e-13)
        final_score = compute_score(current_best_points)
        if final_score > current_best_score:
            current_best_score = final_score

    return current_best_points
