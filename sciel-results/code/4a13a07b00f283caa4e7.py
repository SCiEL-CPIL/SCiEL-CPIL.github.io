import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hex_init():
        """Generate hexagonal lattice initialization with slight noise."""
        # Estimate optimal spacing
        # Area of hex lattice cell is s^2 * sqrt(3)/2
        # N * s^2 * sqrt(3)/2 = pi * 1^2 => s = sqrt(2*pi / (N*sqrt(3)))
        s = np.sqrt(2 * np.pi / (np.sqrt(3) * N)) * 0.98
        y_step = s * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / s) + 2
        
        points = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else s / 2
            for i in range(-max_i, max_i + 1):
                x = i * s + offset
                if np.hypot(x, y) < 1.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback: random
            return np.random.randn(N, 2) * 0.5
        
        # Sort by distance to origin, take N closest
        dists = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(dists)
        pts = points[idx[:N]].copy()
        
        # Add small noise to break symmetry
        pts += np.random.randn(N, 2) * 1e-5
        return pts

    def calc_radius(pts):
        """Compute the maximum radius for given centers."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_boundary)

    def md_optimize(x0, time_limit):
        """
        Molecular Dynamics simulation to find a dense packing.
        Uses repulsive forces between particles and a soft wall.
        """
        x = x0.copy()
        v = np.zeros_like(x)
        
        # MD Parameters
        dt = 0.005
        damping = 0.99
        k_repel = 50.0
        k_wall = 100.0
        r_target = calc_radius(x) # Initial estimate
        
        t0 = time.time()
        
        # Pre-calculate neighbor lists? For N=300, O(N^2) is 90k pairs. 
        # 90k * 1000 steps = 90M ops. In numpy, this is fast.
        
        while time.time() - t0 < time_limit:
            # Calculate forces
            F = np.zeros_like(x)
            
            # Pairwise repulsion
            # diff[i, j] = x[i] - x[j]
            diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Force is proportional to (d_min - d) if d < d_min
            # We want to maximize min dist, so we repel strongly if close.
            # Let's use a potential U = k * (d - 2*r)^2 for d < 2*r
            # But r changes. 
            # Simpler: Repel if d < threshold.
            # Let's use a soft repulsion: F = k * (1 - d/d0)^2 * (d0-d) / d * u
            # Or just simple spring force if overlapping.
            
            # Let's define "overlap" relative to current best radius.
            # Actually, standard MD for packing uses a repulsive potential 
            # that goes to infinity at d=0.
            # F_ij = k * (1/d^2 - 1/d0^2) * u_ij ?
            
            # Let's stick to a simple repulsive force that pushes them apart.
            # F = k_repel * (d0 - d) * u  if d < d0
            # We want d0 to be large.
            # Let's set d0 dynamically based on current min dist?
            # No, let's just use a fixed strong repulsion.
            
            # Vectorized force calculation
            # dists is (N, N)
            # We want force on i from j.
            # F_i = sum_j k * (d0 - d_ij) * u_ij  for d_ij < d0
            
            # Let's use d0 = 2 * r_current * 1.1
            d0 = 2.0 * r_target * 1.05
            mask = dists < d0
            # Avoid division by zero
            dists_safe = np.where(dists > 1e-8, dists, 1.0)
            
            # Force magnitude: k * (d0 - d)
            # Direction: (x_i - x_j) / d
            # F_i += sum_j k * (d0 - d_ij) * (x_i - x_j) / d_ij
            # Note: (x_i - x_j) / d_ij is u_ij.
            # So F_i = sum_j k * (d0 - d_ij) * u_ij
            
            # We can compute this as:
            # F_i = sum_j k * (d0 - d_ij) * (x_i - x_j) / d_ij
            # Let's create a matrix of forces.
            # W_ij = k * (d0 - d_ij) / d_ij if d_ij < d0 else 0
            # F_i = sum_j W_ij * (x_i - x_j)
            # F_i = x_i * sum_j W_ij - sum_j W_ij * x_j
            
            W = np.zeros((N, N))
            W[mask] = k_repel * (d0 - dists[mask]) / dists[mask]
            
            # Sum of weights for each i
            sum_W = np.sum(W, axis=1)
            # Sum of W_ij * x_j
            weighted_x = W.dot(x)
            
            F = x * sum_W[:, np.newaxis] - weighted_x
            
            # Wall force
            # If |x_i| > 1 - r_target, push in
            norms = np.linalg.norm(x, axis=1)
            r_wall = 1.0 - r_target * 0.9
            wall_mask = norms > r_wall
            if np.any(wall_mask):
                # F_wall = -k_wall * (norm - r_wall) * u_r
                # u_r = x / norm
                F[wall_mask] -= k_wall * (norms[wall_mask] - r_wall)[:, np.newaxis] * (x[wall_mask] / norms[wall_mask, np.newaxis])
            
            # Update velocities and positions (Velocity Verlet)
            v = v * damping + F * dt
            x += v * dt
            
            # Update r_target
            r_new = calc_radius(x)
            if r_new > r_target:
                r_target = r_new
            
            # Check convergence (low kinetic energy)
            ke = 0.5 * np.sum(v**2)
            if ke < 1e-10:
                break

        return x, r_target

    def lbfgs_polish(x0, r0, time_limit):
        """Polish using L-BFGS-B on a smooth objective with annealed sharpness."""
        x = x0.copy()
        best_x = x.copy()
        best_r = r0
        
        t0 = time.time()
        
        # We will run multiple L-BFGS passes with increasing beta
        # Beta controls the sharpness of the soft-min
        betas = [20.0, 50.0, 100.0, 200.0]
        
        for beta in betas:
            if time.time() - t0 > time_limit:
                break
                
            def objective(flat_x):
                pts = flat_x.reshape(N, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                norms = np.linalg.norm(pts, axis=1)
                boundary = 1.0 - norms
                
                all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
                
                # Soft min: -log-sum-exp(-beta * dists) / beta
                scaled = -beta * all_dists
                max_scaled = np.max(scaled)
                log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
                return log_sum_exp / beta
            
            def gradient(flat_x):
                pts = flat_x.reshape(N, 2)
                diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
                dists = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dists, np.inf)
                norms = np.linalg.norm(pts, axis=1)
                boundary = 1.0 - norms
                
                all_dists = np.concatenate([dists[np.triu_indices(N, k=1)] / 2.0, boundary])
                scaled = -beta * all_dists
                max_scaled = np.max(scaled)
                exp_scaled = np.exp(scaled - max_scaled)
                sum_exp = np.sum(exp_scaled)
                weights = exp_scaled / sum_exp
                
                grad_flat = np.zeros(2 * N)
                
                # Pairwise gradients
                mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                pair_dists = dists[mask]
                pair_weights = weights[:len(pair_dists)]
                
                W = np.zeros((N, N))
                W[mask] = pair_weights
                
                safe_dists = np.where(dists > 1e-8, dists, 1.0)
                unit_vecs = diff / safe_dists[:, :, np.newaxis]
                S = np.sum(W[:, :, np.newaxis] * unit_vecs, axis=1)
                pair_grads = -beta / 2.0 * S
                
                # Boundary gradients
                bnd_weights = weights[len(pair_dists):]
                safe_norms = np.where(norms > 1e-8, norms, 1.0)
                bnd_unit = pts / safe_norms[:, np.newaxis]
                bnd_grads = beta * bnd_weights[:, np.newaxis] * bnd_unit
                
                total_grad = pair_grads + bnd_grads
                grad_flat = total_grad.flatten()
                
                return grad_flat / beta
            
            try:
                res = minimize(objective, x.flatten(), method='L-BFGS-B', jac=gradient,
                              options={'maxiter': 200, 'maxfun': 2000, 'disp': False})
                x_new = res.x.reshape(N, 2)
                r_new = calc_radius(x_new)
                if r_new > best_r:
                    best_r = r_new
                    best_x = x_new.copy()
                x = x_new.copy() # Continue from here
            except Exception:
                pass

        return best_x, best_r

    # Main execution
    x0 = hex_init()
    r0 = calc_radius(x0)
    best_x = x0.copy()
    best_r = r0
    
    # Phase 1: MD
    md_time = 40.0
    x_md, r_md = md_optimize(x0, md_time)
    if r_md > best_r:
        best_r = r_md
        best_x = x_md.copy()
    
    # Phase 2: L-BFGS Polish
    polish_time = 90.0
    x_polish, r_polish = lbfgs_polish(x_md, r_md, polish_time)
    if r_polish > best_r:
        best_r = r_polish
        best_x = x_polish.copy()
    
    # Final validation
    if np.any(~np.isfinite(best_x)):
        best_x = x0
        best_r = r0
    
    # Ensure all points are inside
    norms = np.linalg.norm(best_x, axis=1)
    if np.any(norms > 1.0):
        # Project back
        for i in range(N):
            if norms[i] > 1.0:
                best_x[i] = best_x[i] / norms[i] * 0.999
        best_r = calc_radius(best_x)
    
    return best_x
