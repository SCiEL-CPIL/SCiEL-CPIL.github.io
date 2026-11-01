"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator kills at 150s)
    
    best_score = -1.0
    best_points = None
    
    # Precompute upper triangle indices for pairwise distances
    iu = np.triu_indices(N, k=1)
    num_pairs = len(iu[0])
    
    def calculate_score(points):
        if points is None:
            return 0.0
        pts = points.reshape(N, 3)
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
            
        # Pairwise distances using precomputed indices
        diff = pts[iu[0]] - pts[iu[1]]
        dists = np.sqrt(np.sum(diff * diff, axis=1))
        min_pair = np.min(dists)
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        r = min(min_pair / 2.0, min_boundary)
        return r
    
    def soft_min_objective(x, beta):
        """
        Soft-min objective: approximate min of pairwise distances/2 and boundary distances
        using log-sum-exp for smoothness. Lower is better (we minimize negative of soft-min).
        """
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[iu[0]] - pts[iu[1]]
        dists_sq = np.sum(diff * diff, axis=1)
        dists = np.sqrt(dists_sq + 1e-20)
        half_dists = dists / 2.0
        
        # Boundary distances
        b_dists = 0.5 - np.abs(pts)
        
        # Combine all constraints into one array
        all_dists = np.concatenate([half_dists, b_dists])
        
        # Soft-min via log-sum-exp: soft_min = -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        log_sum = np.log(np.sum(np.exp(-beta * shifted))) - beta * max_d
        soft_min = -log_sum / beta
        
        # We want to MAXIMIZE soft_min, so minimize -soft_min
        return -soft_min
    
    def soft_min_grad(x, beta):
        """Gradient of the soft-min objective."""
        pts = x.reshape(N, 3)
        
        # Pairwise distances
        diff = pts[iu[0]] - pts[iu[1]]
        dists_sq = np.sum(diff * diff, axis=1)
        dists = np.sqrt(dists_sq + 1e-20)
        half_dists = dists / 2.0
        
        # Boundary distances
        b_dists = 0.5 - np.abs(pts)
        
        # Combine all constraints
        all_dists = np.concatenate([half_dists, b_dists])
        
        # Soft-min gradient
        max_d = np.max(all_dists)
        shifted = all_dists - max_d
        exp_vals = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp  # softmax weights
        
        # Gradient of -soft_min = sum(weights * grad(-d_k))
        # For pairwise: d(d_ij/2)/dx_i = (x_i - x_j)/(2*d_ij)
        # For boundary: d(0.5 - |x_i|)/dx_i = -sign(x_i)
        
        grad = np.zeros_like(pts)
        
        # Pairwise part
        pair_weights = weights[:num_pairs]
        active_pairs = np.where(dists > 1e-15)[0]
        for k in active_pairs:
            i, j = iu[0][k], iu[1][k]
            w = pair_weights[k]
            d = dists[k]
            factor = w / (2.0 * d)
            grad[i] += factor * diff[k]
            grad[j] -= factor * diff[k]
        
        # Boundary part
        bound_weights = weights[num_pairs:]
        for i in range(N):
            w = bound_weights[i]
            # For each coordinate, the gradient of (0.5 - |x_ik|) w.r.t. x_ik is -sign(x_ik)
            # But we use the minimum coordinate (closest face)
            min_coord = np.argmin(b_dists[i])
            coord_val = pts[i, min_coord]
            if coord_val > 0:
                sign = 1.0
            elif coord_val < 0:
                sign = -1.0
            else:
                sign = 0.0
            if sign != 0:
                grad[i, min_coord] -= w * sign
        
        return grad.flatten()
    
    def dynamics_optimize(pts_init, max_time=30.0):
        """
        Rigid-body dynamics with velocity-Verlet integration.
        Spheres experience repulsive forces when separation < target.
        """
        pts = pts_init.copy()
        velocities = np.zeros_like(pts)
        
        # Target radius for repulsion
        r_target = 0.112
        dt = 0.002
        damping = 0.995
        max_iter = int(max_time / dt)
        
        # Precompute
        iu_local = iu
        
        def compute_forces(pts):
            forces = np.zeros_like(pts)
            
            # Pairwise repulsive forces
            diff = pts[iu_local[0]] - pts[iu_local[1]]
            dists_sq = np.sum(diff * diff, axis=1)
            dists = np.sqrt(dists_sq + 1e-20)
            
            # Repulsion: F = k * (r_target - d) * (x_i - x_j)/d for d < r_target
            active = dists < r_target
            if np.any(active):
                active_idx = np.where(active)[0]
                for k in active_idx:
                    i, j = iu_local[0][k], iu_local[1][k]
                    d = dists[k]
                    if d > 1e-15:
                        f_mag = (r_target - d) / d
                        forces[i] += f_mag * diff[k]
                        forces[j] -= f_mag * diff[k]
            
            # Boundary forces: push away from walls
            b_dists = 0.5 - np.abs(pts)
            for i in range(N):
                min_coord = np.argmin(b_dists[i])
                coord_val = pts[i, min_coord]
                if coord_val > 0:
                    forces[i, min_coord] -= max(0, r_target - b_dists[i, min_coord])
                elif coord_val < 0:
                    forces[i, min_coord] += max(0, r_target - b_dists[i, min_coord])
            
            return forces
        
        # Initial forces
        forces = compute_forces(pts)
        
        for step in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Velocity-Verlet: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            acc = forces  # mass = 1
            pts += velocities * dt + 0.5 * acc * dt * dt
            
            # New forces
            new_forces = compute_forces(pts)
            
            # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            velocities += 0.5 * (acc + new_forces) * dt
            
            # Damping
            velocities *= damping
            
            # Clip to boundary
            pts = np.clip(pts, -0.499, 0.499)
            
            forces = new_forces
        
        return pts
    
    def init_grid():
        m = 5
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()

    def init_bcc():
        """Body-centered cubic initialization."""
        pts = []
        spacing = 1.0 / 4.5
        for i in range(5):
            for j in range(5):
                for k in range(5):
                    pts.append([i * spacing - 0.5 + spacing/2, 
                                j * spacing - 0.5 + spacing/2, 
                                k * spacing - 0.5 + spacing/2])
                    if i < 4 and j < 4 and k < 4:
                        pts.append([i * spacing - 0.5 + spacing, 
                                    j * spacing - 0.5 + spacing, 
                                    k * spacing - 0.5 + spacing])
        pts = np.array(pts[:N])
        return pts

    def init_halton():
        """Halton sequence initialization for good distribution."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        pts = []
        for i in range(1, N + 1):
            x = halton(i, 2)
            y = halton(i, 3)
            z = halton(i, 5)
            pts.append([x - 0.5, y - 0.5, z - 0.5])
        return np.array(pts)

    def init_random():
        return np.random.uniform(-0.4, 0.4, size=(N, 3))

    def polish_with_lbfgs(pts, beta_start=50.0, beta_end=500.0, max_time=20.0):
        """Polish using L-BFGS-B with increasing beta."""
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        betas = [beta_start, beta_start * 2, beta_start * 4, beta_start * 8, beta_end]
        current_pts = pts.copy()
        
        for beta in betas:
            if time.time() > deadline - 5:
                break
            
            try:
                res = minimize(soft_min_objective, current_pts.flatten(),
                              method='L-BFGS-B', bounds=bounds,
                              args=(beta,),
                              jac=lambda x, b: soft_min_grad(x, b),
                              options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res.fun < 1e6:
                    new_pts = res.x.reshape(N, 3)
                    new_pts = np.clip(new_pts, -0.5, 0.5)
                    new_score = calculate_score(new_pts)
                    old_score = calculate_score(current_pts)
                    
                    if new_score >= old_score - 1e-8:
                        current_pts = new_pts
            except Exception:
                continue
        
        return current_pts

    # Generate diverse initializations
    starts = [
        init_grid(),
        init_grid() + np.random.normal(0, 0.01, size=(N, 3)),
        init_grid() + np.random.normal(0, 0.02, size=(N, 3)),
        init_bcc(),
        init_bcc() + np.random.normal(0, 0.01, size=(N, 3)),
        init_halton(),
        init_random(),
        init_random()
    ]
    
    # Clip starts to valid range
    for i in range(len(starts)):
        starts[i] = np.clip(starts[i], -0.49, 0.49)

    # Phase 1: Dynamics optimization for each start
    for x0 in starts:
        if time.time() > deadline - 20:
            break
        
        try:
            # Run dynamics
            dyn_pts = dynamics_optimize(x0, max_time=15.0)
            dyn_score = calculate_score(dyn_pts)
            
            if dyn_score > best_score:
                best_score = dyn_score
                best_points = dyn_pts.copy()
            
            # Phase 2: Polish with L-BFGS
            if time.time() < deadline - 15:
                polished_pts = polish_with_lbfgs(dyn_pts, beta_start=50.0, beta_end=500.0, max_time=15.0)
                pol_score = calculate_score(polished_pts)
                
                if pol_score > best_score:
                    best_score = pol_score
                    best_points = polished_pts.copy()
                    
        except Exception:
            continue
    
    # Final high-precision polish
    if best_points is not None and time.time() < deadline - 10:
        try:
            final_pts = polish_with_lbfgs(best_points, beta_start=200.0, beta_end=1000.0, max_time=10.0)
            final_score = calculate_score(final_pts)
            
            if final_score > best_score:
                best_score = final_score
                best_points = final_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        best_points = init_grid()
        
    return best_points
# EVOLVE-BLOCK-END
