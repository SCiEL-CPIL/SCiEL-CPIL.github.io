"""Optimized sphere packing in unit cube."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def compute_radius(pts):
        if pts is None:
            return 0.0
        pts = np.asarray(pts).reshape(N, 3)
        if not np.all(np.isfinite(pts)):
            return 0.0
        if np.any(pts < -0.5 - 1e-12) or np.any(pts > 0.5 + 1e-12):
            return 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        boundary = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary)
        return min(min_pair / 2.0, min_boundary)
    
    def soft_min_obj(x, P):
        pts = x.reshape(N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        pair_term = np.sum(1.0 / (dists_safe ** P))
        
        abs_pts = np.abs(pts)
        b_dists = 0.5 - abs_pts
        min_b = np.min(b_dists, axis=1)
        min_b_safe = np.where(min_b < 1e-10, 1e-10, min_b)
        bound_term = np.sum(1.0 / (min_b_safe ** P))
        
        return pair_term + bound_term
    
    def soft_min_grad(x, P):
        pts = x.reshape(N, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dists_sq, np.inf)
        dists = np.sqrt(dists_sq)
        dists_safe = np.where(dists < 1e-10, 1e-10, dists)
        
        # Gradient of sum 1/d^P = -P * sum d^(-P-2) * (p_i - p_j)
        W = dists_safe ** (-P - 2.0)
        np.fill_diagonal(W, 0.0)
        # grad_i_pair = -P * sum_j W_ij * (p_i - p_j) = -P * (row_sum_i * p_i - sum_j W_ij * p_j)
        row_sums = np.sum(W, axis=1)
        Wp = W @ pts
        grad_pair = -P * (row_sums[:, np.newaxis] * pts - Wp)
        
        # Boundary gradient
        abs_pts = np.abs(pts)
        b_dists = 0.5 - abs_pts
        min_idx = np.argmin(b_dists, axis=1)
        min_b = np.min(b_dists, axis=1)
        min_b_safe = np.where(min_b < 1e-10, 1e-10, min_b)
        
        grad_bound = np.zeros_like(pts)
        for i in range(N):
            k = min_idx[i]
            s = np.sign(pts[i, k])
            if s == 0:
                s = 1.0
            grad_bound[i, k] = P * (min_b_safe[i] ** (-P - 1.0)) * s
        
        return (grad_pair + grad_bound).flatten()
    
    def polish_repulsion(pts, P=40.0, max_iter=2000):
        """Polish using L-BFGS-B with soft-min objective."""
        x0 = pts.flatten()
        bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (N * 3)
        try:
            res = minimize(
                lambda x: soft_min_obj(x, P),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                jac=lambda x: soft_min_grad(x, P),
                options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
            )
            if res.fun < 1e15:
                return res.x.reshape(N, 3)
        except Exception:
            pass
        return pts
    
    def build_grid_init(nx, ny, nz):
        """Build a grid initialization with nx*ny*nz = N points."""
        tx = (np.arange(nx) + 0.5) / nx - 0.5
        ty = (np.arange(ny) + 0.5) / ny - 0.5
        tz = (np.arange(nz) + 0.5) / nz - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        if len(grid) == N:
            return grid.flatten()
        return None
    
    def build_bcc_init():
        """Body-centered cubic lattice initialization."""
        # 5x5x5 = 125 is too many, use 4x5x5=100
        tx = (np.arange(4) + 0.5) / 4 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def build_halton_init():
        """Halton sequence initialization."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            i = index
            while i > 0:
                f /= base
                r += f * (i % base)
                i //= base
            return r
        
        pts = np.zeros((N, 3))
        for i in range(N):
            pts[i, 0] = halton(i + 1, 2) - 0.5
            pts[i, 1] = halton(i + 1, 3) - 0.5
            pts[i, 2] = halton(i + 1, 5) - 0.5
        return pts.flatten()
    
    def build_random_init(seed=42):
        rng = np.random.RandomState(seed)
        return rng.uniform(-0.45, 0.45, size=(N, 3)).flatten()
    
    def simulated_annealing(pts, T_start=0.05, T_end=0.001, n_steps=2000, cool_rate=0.9995):
        """Simulated annealing for global search."""
        pts = pts.copy()
        current_r = compute_radius(pts)
        best_pts = pts.copy()
        best_r = current_r
        T = T_start
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
            
            # Perturb a random subset of points
            n_perturb = max(1, N // 10)
            idx = np.random.choice(N, size=n_perturb, replace=False)
            perturbation = np.random.normal(0, T, size=(n_perturb, 3))
            new_pts = pts.copy()
            new_pts[idx] += perturbation
            new_pts = np.clip(new_pts, -0.49, 0.49)
            
            new_r = compute_radius(new_pts)
            delta_r = new_r - current_r
            
            if delta_r > 0 or np.random.random() < np.exp(delta_r / max(T, 1e-10)):
                pts = new_pts
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
            
            T = max(T * cool_rate, T_end)
        
        return best_pts
    
    def optimize_from_initial(x0):
        """Full optimization pipeline from an initial point."""
        pts = x0.reshape(N, 3)
        
        # Phase 1: Simulated annealing for global exploration
        pts = simulated_annealing(pts, T_start=0.03, T_end=0.0005, n_steps=1500)
        
        # Phase 2: L-BFGS polishing with increasing P
        for P in [20, 40, 60]:
            if time.time() > deadline - 5:
                break
            pts = polish_repulsion(pts, P=P, max_iter=1500)
            pts = np.clip(pts, -0.5, 0.5)
        
        # Phase 3: Final fine polish
        if time.time() < deadline - 3:
            pts = polish_repulsion(pts, P=80, max_iter=1000)
            pts = np.clip(pts, -0.5, 0.5)
        
        return pts
    
    # Generate multiple diverse initializations
    inits = []
    
    # Grid-based
    grid_455 = build_grid_init(4, 5, 5)
    if grid_455 is not None:
        inits.append(grid_455)
    grid_545 = build_grid_init(5, 4, 5)
    if grid_545 is not None:
        inits.append(grid_545)
    grid_554 = build_grid_init(5, 5, 4)
    if grid_554 is not None:
        inits.append(grid_554)
    
    # BCC
    inits.append(build_bcc_init())
    
    # Halton
    inits.append(build_halton_init())
    
    # Random with different seeds
    for seed in [1, 2, 3]:
        inits.append(build_random_init(seed=seed))
    
    best_score = -1.0
    best_points = None
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 10:
            break
        
        try:
            pts = optimize_from_initial(x0)
            pts = np.clip(pts, -0.5, 0.5)
            score = compute_radius(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            continue
    
    # Final refinement on best
    if best_points is not None and time.time() < deadline - 3:
        try:
            pts = polish_repulsion(best_points, P=100, max_iter=500)
            pts = np.clip(pts, -0.5, 0.5)
            score = compute_radius(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = build_bcc_init().reshape(N, 3)
    
    return best_points
