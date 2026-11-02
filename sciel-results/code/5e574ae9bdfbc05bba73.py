import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

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

    def generate_ring_init(n_rings=15):
        """Generate concentric ring initialization."""
        pts = [[0.0, 0.0]]
        for ring in range(1, n_rings + 1):
            r = ring * 0.065
            if r > 0.95:
                break
            n_points = int(2 * np.pi * r / 0.11)
            if n_points < 1:
                n_points = 1
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            for a in angles:
                pts.append([r * np.cos(a), r * np.sin(a)])
        
        pts = np.array(pts)
        if len(pts) < N:
            # Pad with random points
            extra = np.random.randn(N - len(pts), 2)
            extra *= 0.5
            pts = np.vstack([pts, extra])
        else:
            # Select N closest to center
            dists = np.hypot(pts[:, 0], pts[:, 1])
            idx = np.argsort(dists)[:N]
            pts = pts[idx]
        return pts

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
        
        # Effective radii
        eff_rad = np.concatenate([d_flat / 2.0, d_bound])
        
        # Soft min
        s_val, grad_eff = soft_min_and_grad(eff_rad, beta)
        obj = -s_val
        
        grad_pair = grad_eff[:len(d_flat)]
        grad_bound = grad_eff[len(d_flat):]
        
        # Gradient via np.add.at for efficiency
        safe_d = np.where(d_matrix < 1e-9, 1e-9, d_matrix)
        
        # Build symmetric gradient matrix
        G = np.zeros((N, N))
        G[mask] = grad_pair
        G.T[mask] = grad_pair
        
        # F_i = sum_j G[i,j] * (p_i - p_j) / d_ij
        weight = G / safe_d
        F_pair = weight @ diff
        
        # Boundary gradient
        safe_radii = np.where(radii < 1e-9, 1e-9, radii)
        F_bound = -grad_bound[:, np.newaxis] * pts / safe_radii[:, np.newaxis]
        
        total_grad = F_pair + F_bound
        
        return obj, total_grad.flatten()

    # 1. Initialization: Try multiple hexagonal lattices and ring init
    best_init = None
    best_init_score = -np.inf
    
    spacings = [0.10, 0.104, 0.105, 0.106, 0.107, 0.108, 0.11, 0.112, 0.115]
    rotations = [0.0, np.pi / 12, np.pi / 8, np.pi / 6, np.pi / 5, np.pi / 4]
    
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
    
    # Also try ring initialization
    ring_pts = generate_ring_init(15)
    ring_score = compute_score(ring_pts)
    if ring_score > best_init_score:
        best_init_score = ring_score
        best_init = ring_pts
    
    if best_init is None:
        best_init = np.random.randn(N, 2)
        best_init /= np.linalg.norm(best_init, axis=1, keepdims=True)
        best_init *= 0.9
        best_init_score = compute_score(best_init)

    current_points = best_init.copy()
    current_best_score = best_init_score
    current_best_points = best_init.copy()

    # 2. Optimization with annealed beta schedule
    max_time = deadline - start_time - 5.0
    beta_start = 30.0
    beta_end = 50000.0
    
    num_rounds = 10
    time_per_round = max_time / num_rounds
    
    for round_idx in range(num_rounds):
        if time.time() > deadline - 5.0:
            break
            
        t = round_idx / max(1, num_rounds - 1)
        beta = beta_start * (beta_end / beta_start) ** t
        
        flat_pts = current_points.flatten()
        round_deadline = time.time() + time_per_round * 0.85
        
        try:
            result = minimize(
                objective_and_grad,
                flat_pts,
                args=(beta,),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 800,
                    'maxfun': 1500,
                    'ftol': 1e-12,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:
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
        
        if time.time() > round_deadline:
            continue

    # 3. Final polish pass with high beta
    if time.time() < deadline - 3.0:
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
                    'maxiter': 1000,
                    'maxfun': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-14,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                
                radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_of_bounds = radii > 1.0
                if np.any(out_of_bounds):
                    new_pts[out_of_bounds] *= (0.9999 / radii[out_of_bounds])[:, np.newaxis]
                
                new_score = compute_score(new_pts)
                
                if new_score > current_best_score:
                    current_best_score = new_score
                    current_best_points = new_pts.copy()
                    
        except Exception:
            pass

    # 4. Local refinement: push apart close pairs
    if time.time() < deadline - 2.0:
        pts = current_best_points.copy()
        for iteration in range(50):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d < 2 * current_best_score + 0.001:
                        if d < 1e-10:
                            pts[i] += 0.001 * np.random.randn(2)
                        else:
                            direction = (pts[i] - pts[j]) / d
                            push = (2 * current_best_score + 0.001 - d) / 2
                            pts[i] += push * direction
                            pts[j] -= push * direction
                            improved = True
                # Keep inside circle
                r = np.linalg.norm(pts[i])
                if r > 1 - current_best_score:
                    pts[i] *= (1 - current_best_score) / r
            if not improved:
                break
        
        new_score = compute_score(pts)
        if new_score > current_best_score:
            current_best_score = new_score
            current_best_points = pts.copy()

    return current_best_points
