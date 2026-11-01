"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_score = -1.0
    best_points = None
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r
    
    def init_grid_4x5x5():
        tx = (np.arange(4) + 0.5) / 4 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_5x4x5():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(4) + 0.5) / 4 - 0.5
        tz = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_grid_5x5x4():
        tx = (np.arange(5) + 0.5) / 5 - 0.5
        ty = (np.arange(5) + 0.5) / 5 - 0.5
        tz = (np.arange(4) + 0.5) / 4 - 0.5
        grid = np.array([(x, y, z) for z in tz for y in ty for x in tx])
        return grid.flatten()
    
    def init_bcc():
        """Body-centered cubic lattice initialization."""
        pts = []
        # Simple cubic sublattice
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        for z in ticks:
            for y in ticks:
                for x in ticks:
                    pts.append((x, y, z))
        # Add BCC centers if we need more
        while len(pts) < N:
            # Add body-center positions of the 5x5x5 grid cells
            cell_size = 1.0 / (m - 1) if m > 1 else 1.0
            # Actually use a different approach: add midpoints
            pass
        # Fallback: use 4x5x5
        return init_grid_4x5x5()
    
    def init_random():
        return np.random.uniform(-0.45, 0.45, size=(N, 3)).flatten()
    
    def force_directed_refine(pts, n_steps=500, cooling=True):
        """Repulsion-based refinement to spread points apart."""
        pts = pts.copy()
        N_local = len(pts)
        
        for step in range(n_steps):
            if time.time() > deadline - 5:
                break
                
            # Compute forces: repulsion between close pairs + boundary push
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Only consider pairs closer than 2*target (use current min as proxy)
            # Use a cutoff to save time
            cutoff = 0.25
            mask = dists < cutoff
            mask &= ~np.eye(N_local, dtype=bool)
            
            # Repulsion force: F = (1/d^2) * direction
            forces = np.zeros_like(pts)
            for i in range(N_local):
                for j in range(N_local):
                    if mask[i, j]:
                        d = dists[i, j]
                        if d > 1e-10:
                            f = 1.0 / (d * d)
                            forces[i] += f * (pts[i] - pts[j])
            
            # Boundary forces
            abs_pts = np.abs(pts)
            b_dists = 0.5 - abs_pts
            for i in range(N_local):
                for k in range(3):
                    bd = b_dists[i, k]
                    if bd < 0.15:
                        s = np.sign(pts[i, k])
                        if s == 0: s = 1.0
                        forces[i, k] += 10.0 / max(bd, 0.01) * s
            
            # Update with cooling
            if cooling:
                temp = 0.001 * (1.0 - step / n_steps)
            else:
                temp = 0.001
            
            pts += temp * forces
            pts = np.clip(pts, -0.49, 0.49)
        
        return pts
    
    def penalty_objective(x, r_target, mu):
        """Penalty objective for feasibility at radius r_target."""
        pts = x.reshape(N, 3)
        
        # Pairwise violations: dist < 2*r_target
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Penalty for pairs too close
        violations_pair = np.maximum(0, 2 * r_target - dists)
        pair_penalty = np.sum(violations_pair ** 2)
        
        # Boundary violations: dist to boundary < r_target
        b_dists = 0.5 - np.abs(pts)
        violations_bound = np.maximum(0, r_target - b_dists)
        bound_penalty = np.sum(violations_bound ** 2)
        
        total = pair_penalty + bound_penalty
        return total
    
    def penalty_grad(x, r_target, mu):
        """Gradient of penalty objective."""
        pts = x.reshape(N, 3)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        grad = np.zeros_like(pts)
        
        # Pairwise gradient
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                d = dists[i, j]
                if d < 2 * r_target and d > 1e-10:
                    # d(0.5*(2r - d)^2)/dxi = -(2r-d) * (xi-xj)/d
                    factor = -(2 * r_target - d) / d
                    grad[i] += factor * (pts[i] - pts[j])
        
        # Boundary gradient
        b_dists = 0.5 - np.abs(pts)
        for i in range(N):
            for k in range(3):
                bd = b_dists[i, k]
                if bd < r_target:
                    s = np.sign(pts[i, k])
                    if s == 0: s = 1.0
                    grad[i, k] += 2 * (r_target - bd) * s
        
        return grad.flatten()
    
    def try_radius(pts, r_target, max_iter=2000):
        """Try to find a feasible configuration for radius r_target."""
        x0 = pts.flatten()
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        try:
            res = minimize(
                lambda x: penalty_objective(x, r_target, 1.0),
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                jac=lambda x: penalty_grad(x, r_target, 1.0),
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
            )
            if res.fun < 1e-8:
                return res.x.reshape(N, 3)
        except Exception:
            pass
        return None
    
    def binary_search_radius(pts, r_low, r_high, n_iters=15):
        """Binary search for maximum feasible radius."""
        best_pts = pts.copy()
        best_r = r_low
        
        for _ in range(n_iters):
            if time.time() > deadline - 5:
                break
                
            r_mid = (r_low + r_high) / 2.0
            result = try_radius(best_pts, r_mid, max_iter=1000)
            if result is not None:
                best_r = r_mid
                best_pts = result
                r_low = r_mid
            else:
                r_high = r_mid
        
        return best_r, best_pts
    
    def full_optimize_from_start(x0):
        """Full optimization pipeline from an initial point."""
        pts = x0.reshape(N, 3)
        
        # Step 1: Force-directed refinement
        pts = force_directed_refine(pts, n_steps=200)
        
        # Step 2: Get initial radius estimate
        r_init = calculate_score(pts)
        
        # Step 3: Binary search for better radius
        r_low = max(r_init * 0.95, 0.05)
        r_high = min(r_init * 1.15, 0.45)
        
        # Ensure r_low is feasible
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            r_test = r_low
            result = try_radius(pts, r_test, max_iter=500)
            if result is not None:
                pts = result
                break
            else:
                r_low *= 0.9
        
        # Binary search
        r_final, pts_final = binary_search_radius(pts, r_low, r_high, n_iters=12)
        
        # Step 4: Final polish with soft-min objective at the found radius
        P = 30.0
        def soft_min_obj(x):
            p = x.reshape(N, 3)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            dists_safe = np.where(dists < 1e-10, 1e-10, dists)
            sum_pair = np.sum(1.0 / (dists_safe ** P))
            
            abs_p = np.abs(p)
            b_dists = 0.5 - abs_p
            min_b = np.min(b_dists, axis=1)
            min_b_safe = np.where(min_b < 1e-10, 1e-10, min_b)
            sum_bound = np.sum(1.0 / (min_b_safe ** P))
            
            return sum_pair + sum_bound
        
        def soft_min_grad(x):
            p = x.reshape(N, 3)
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            dists_safe = np.where(dists < 1e-10, 1e-10, dists)
            
            W = dists_safe ** (-P - 2.0)
            np.fill_diagonal(W, 0.0)
            row_sums = np.sum(W, axis=1)
            grad_pair = -P * (row_sums[:, np.newaxis] * p - (W @ p))
            
            abs_p = np.abs(p)
            b_dists = 0.5 - abs_p
            min_idx = np.argmin(b_dists, axis=1)
            min_b = np.min(b_dists, axis=1)
            min_b_safe = np.where(min_b < 1e-10, 1e-10, min_b)
            
            grad_bound = np.zeros_like(p)
            for i in range(N):
                k = min_idx[i]
                s = np.sign(p[i, k])
                if s == 0: s = 1.0
                grad_bound[i, k] = P * (min_b_safe[i] ** (-P - 1.0)) * s
            
            return (grad_pair + grad_bound).flatten()
        
        bounds = [(-0.5, 0.5)] * (N * 3)
        try:
            res = minimize(soft_min_obj, pts_final.flatten(), method='L-BFGS-B',
                          bounds=bounds, jac=soft_min_grad,
                          options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
            if res.fun < 1e15:
                pts_final = res.x.reshape(N, 3)
                pts_final = np.clip(pts_final, -0.5, 0.5)
        except Exception:
            pass
        
        return pts_final
    
    # Multiple starts
    starts = [
        init_grid_4x5x5(),
        init_grid_5x4x5(),
        init_grid_5x5x4(),
        init_random(),
        init_random(),
    ]
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        try:
            pts = full_optimize_from_start(x0)
            pts = np.clip(pts, -0.5, 0.5)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            continue
    
    # Final refinement on best
    if best_points is not None and time.time() < deadline - 5:
        try:
            pts = force_directed_refine(best_points, n_steps=100)
            score = calculate_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = init_grid_4x5x5().reshape(N, 3)
        
    return best_points
# EVOLVE-BLOCK-END
