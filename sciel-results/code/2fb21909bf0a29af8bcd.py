import numpy as np
import time

N = 100

def _score(points):
    """Largest common radius: min of half pairwise distances and wall clearance."""
    # points is (N, 3)
    d = points[:, None, :] - points[None, :, :]
    dists = np.sqrt(np.sum(d * d, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = dists[iu].min()
    clearance = (0.5 - np.abs(points)).min()
    return min(min_pair * 0.5, clearance)

def _forces_and_energy(x, target_r):
    """
    Calculate forces and energy for the soft-sphere potential.
    x: (N, 3) array of positions.
    target_r: desired radius.
    Returns: (energy, forces)
    """
    c = x
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0)
    
    # Only consider pairs closer than 2 * target_r
    active = dists < 2.0 * target_r
    active[np.diag_indices_from(active)] = False
    
    # Violations: (2r - d)
    violations = np.where(active, 2.0 * target_r - dists, 0.0)
    
    # Energy: 0.5 * sum(violations^2)
    energy = 0.5 * np.sum(violations**2)
    
    # Forces: -gradient of energy
    # d/dx_i (0.5 * (2r - d)^2) = (2r - d) * d/dx_i (2r - d)
    # d/dx_i (2r - d) = (x_i - x_j) / d
    # Force on i from j = - (2r - d) * (x_i - x_j) / d
    # Note: violations[i,j] = 2r - d.
    # So Force[i,j] = - violations[i,j] * (x_i - x_j) / d
    # We sum over j for each i.
    
    safe_dists = np.where(dists > 1e-12, dists, 1.0)
    unit_vecs = diff / safe_dists[:, :, np.newaxis]
    
    # Force matrix F[i,j] is force on i due to j.
    # F[i,j] = - violations[i,j] * unit_vecs[i,j]
    # But we only want active pairs. violations is 0 for inactive, so it's fine.
    # However, we must be careful with the direction.
    # diff[i,j] = c[i] - c[j].
    # Force on i should push i away from j if overlapping.
    # If c[i] > c[j], diff is positive. Force should be positive (push right).
    # Formula: - (2r - d) * (c[i] - c[j]) / d.
    # If c[i] > c[j], c[i]-c[j] is +. (2r-d) is +. Result is -. This pushes i left (towards j).
    # Wait. If they overlap, we want to push them apart.
    # Potential U = 0.5 (2r - d)^2.
    # Force F = - dU/dx.
    # dU/dx_i = (2r - d) * d(2r-d)/dx_i = (2r-d) * (x_i - x_j)/d.
    # So F_i = - (2r-d) * (x_i - x_j)/d.
    # If x_i > x_j, x_i - x_j > 0. F_i is negative. Force is towards decreasing x_i.
    # This pushes i towards j. That's attractive!
    # Let's re-evaluate.
    # We want repulsion.
    # Usually, hard sphere potential is 0 if d > 2r, infinity if d < 2r.
    # Soft sphere: U = k (2r - d)^2 for d < 2r.
    # If d decreases, U increases.
    # Force should push d to increase.
    # Vector from j to i is x_i - x_j.
    # Force on i should be in direction of (x_i - x_j).
    # My derivation: F_i = - (2r-d) * (x_i - x_j)/d.
    # This is opposite to the direction of separation.
    # Let's check the gradient again.
    # U(d) = 0.5 (2r - d)^2.
    # dU/dd = -(2r - d).
    # F = - dU/dx = - (dU/dd * dd/dx).
    # dd/dx_i = (x_i - x_j) / d.
    # F_i = - (-(2r - d)) * (x_i - x_j) / d = (2r - d) * (x_i - x_j) / d.
    # Ah, the chain rule sign.
    # dU/dd = 2 * 0.5 * (2r - d) * (-1) = -(2r - d).
    # So F_i = - (-(2r - d)) * (x_i - x_j) / d = (2r - d) * (x_i - x_j) / d.
    # This is in the direction of (x_i - x_j). Correct.
    
    # So Force[i,j] = violations[i,j] * unit_vecs[i,j]
    # Sum over j.
    forces_pair = np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
    
    # Wall forces
    # If c_k < -0.5 + r, push right. If c_k > 0.5 - r, push left.
    # Wall violation: r - (0.5 - |c|).
    # Let's use a soft wall potential too.
    # U_wall = 0.5 * max(0, r - (0.5 - |c|))^2
    # Force is derivative.
    # For x > 0: dist = 0.5 - x. viol = r - (0.5 - x) = x - (0.5 - r).
    # dU/dx = viol * 1. Force = -viol. (Pushes left).
    # For x < 0: dist = 0.5 - (-x) = 0.5 + x. viol = r - (0.5 + x) = (0.5 - r) - x.
    # dU/dx = viol * (-1). Force = -viol * (-1) = viol. (Pushes right).
    
    wall_dists = 0.5 - np.abs(c)
    wall_violations = np.maximum(0.0, target_r - wall_dists)
    
    # Force direction: towards center.
    # If c > 0, force is negative. If c < 0, force is positive.
    # Force = - sign(c) * wall_violation
    signs = np.sign(c)
    forces_wall = -signs * wall_violations
    
    # Add wall energy to total energy
    energy += np.sum(wall_violations**2)
    
    total_forces = forces_pair + forces_wall
    
    return energy, total_forces

def _md_anneal(init_pts, target_r, time_limit=10.0, t0=0.0):
    """
    Run MD annealing to optimize positions for a given target radius.
    """
    t_start = time.time()
    positions = init_pts.copy()
    velocities = np.zeros_like(positions)
    
    # Parameters
    dt = 0.001
    damping = 0.99
    mass = 1.0
    
    # Initial temperature (noise magnitude)
    temp = 0.01
    cooling_rate = 0.999
    
    best_energy = np.inf
    best_positions = positions.copy()
    
    steps = 0
    while time.time() - t_start < time_limit:
        # Calculate forces
        energy, forces = _forces_and_energy(positions, target_r)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
            
        # Add thermal noise (Joule heating)
        # Noise is added to velocities
        noise = np.random.normal(0, temp, size=positions.shape)
        velocities += noise
        
        # Update velocities (Euler-Cromer)
        # v += (F/m) * dt
        velocities += (forces / mass) * dt
        
        # Damping
        velocities *= damping
        
        # Update positions
        positions += velocities * dt
        
        # Clip to hard walls to prevent flying out (optional, soft walls handle it, but safety)
        # Actually, if we clip, we break momentum. Better to rely on strong soft walls.
        # But if temp is high, they might fly out.
        # Let's add a hard clip with velocity zeroing if they hit the hard boundary.
        out_of_bounds = (positions < -0.5) | (positions > 0.5)
        if np.any(out_of_bounds):
            # Push back and kill velocity component
            positions[out_of_bounds] = np.clip(positions[out_of_bounds], -0.5, 0.5)
            velocities[out_of_bounds] = 0.0
            
        # Cool down
        temp *= cooling_rate
        
        steps += 1
        
        # Batch processing for speed? 
        # The current loop is Python-level. 
        # We can vectorize the time stepping? 
        # No, forces depend on positions which change each step.
        # However, we can take larger time steps if we use a symplectic integrator and smaller dt.
        # Or we can run multiple steps in a tight loop without time checking every step.
        
        if steps % 100 == 0 and time.time() - t_start > time_limit:
            break
            
    return best_positions

def construct():
    t0 = time.time()
    DEADLINE = 145.0  # Safety margin under 150s
    
    # ---- Seed: Simple cubic 5x5x5 grid ----
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    pts = grid[:N].copy()

    best_pts = pts.copy()
    best_r = _score(best_pts)
    
    rng = np.random.RandomState(42)
    
    # Strategy: 
    # 1. Try to improve using MD annealing with increasing target radii.
    # 2. Use BFGS for final polish.
    
    # Current best is ~0.11098. Best known 0.11138.
    # Let's try targets: 0.111, 0.1112, 0.1114, 0.1115
    
    targets = [0.111, 0.1112, 0.1114, 0.1116]
    
    current_pts = best_pts.copy()
    
    for target_r in targets:
        if time.time() - t0 > DEADLINE - 20.0:
            break
            
        # Add small noise to break symmetry
        noise = rng.normal(0, 0.01, size=(N, 3))
        init_pts = np.clip(current_pts + noise, -0.49, 0.49)
        
        # Run MD Annealing
        # Time limit per target: 10 seconds
        time_left = DEADLINE - (time.time() - t0) - 10.0
        if time_left < 5.0:
            break
            
        opt_pts = _md_anneal(init_pts, target_r, time_limit=min(10.0, time_left), t0=t0)
        
        if np.all(np.isfinite(opt_pts)):
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
                current_pts = opt_pts.copy()
                
    # Final Polish with BFGS if time permits
    if time.time() - t0 < DEADLINE - 10.0 and best_r > 0.1:
        from scipy.optimize import minimize
        
        def _energy_and_grad(x, target_r):
            c = x.reshape(N, 3)
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, 1.0)
            active = dists < 2.0 * target_r
            active[np.diag_indices_from(active)] = False
            violations = np.where(active, 2.0 * target_r - dists, 0.0)
            energy_pair = 0.5 * np.sum(violations**2)
            safe_dists = np.where(dists > 1e-12, dists, 1.0)
            unit_vecs = diff / safe_dists[:, :, np.newaxis]
            grad_pair = -np.sum(violations[:, :, np.newaxis] * unit_vecs, axis=1)
            
            wall_dists = 0.5 - np.abs(c)
            wall_violations = np.maximum(0.0, target_r - wall_dists)
            energy_wall = np.sum(wall_violations**2)
            signs = np.sign(c)
            grad_wall = 2.0 * wall_violations * signs
            
            total_energy = energy_pair + energy_wall
            total_grad = (grad_pair + grad_wall).flatten()
            return total_energy, total_grad

        # Try to optimize for the best_r found, or slightly higher
        target_r = best_r * 1.001
        noise = rng.normal(0, 0.001, size=(N, 3))
        init_pts = np.clip(best_pts + noise, -0.49, 0.49)
        
        t_start = time.time()
        time_limit = DEADLINE - (time.time() - t0) - 2.0
        
        try:
            result = minimize(
                lambda x: _energy_and_grad(x, target_r),
                init_pts.flatten(),
                method='BFGS',
                jac=True,
                options={'maxiter': 1000, 'gtol': 1e-10, 'disp': False}
            )
            opt_pts = result.x.reshape(N, 3)
            opt_pts = np.clip(opt_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            r_opt = _score(opt_pts)
            
            if r_opt > best_r:
                best_r = r_opt
                best_pts = opt_pts.copy()
        except Exception:
            pass

    # Final safety check
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    if not np.all(np.isfinite(best_pts)):
        best_pts = grid[:N].copy()
        
    return best_pts
