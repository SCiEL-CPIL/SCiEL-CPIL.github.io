"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 26 circles in the unit square maximizing sum of radii.
    Uses multi-start physics relaxation + analytic gradient polishing.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    def compute_overlap_penalty(radii, centers, mu=100.0):
        """
        Compute soft overlap penalty and its gradient.
        For each pair i,j: if dist < r_i + r_j, penalty = mu * (r_i + r_j - dist)^2
        Gradient w.r.t. positions: 2*mu*(r_i+r_j-d) * (p_i - p_j)/d
        Gradient w.r.t. radii: 2*mu*(r_i+r_j-d)
        """
        n = len(radii)
        penalty = 0.0
        grad_pos = np.zeros((n, 2))
        grad_rad = np.zeros(n)
        
        for i in range(n):
            for j in range(i+1, n):
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                min_d = radii[i] + radii[j]
                if d < min_d and d > 1e-12:
                    overlap = min_d - d
                    penalty += mu * overlap * overlap
                    factor = 2.0 * mu * overlap / d
                    grad_pos[i] += factor * d_vec
                    grad_pos[j] -= factor * d_vec
                    grad_rad[i] += 2.0 * mu * overlap
                    grad_rad[j] += 2.0 * mu * overlap
                elif d <= 1e-12:
                    # Degenerate case: circles at same point
                    overlap = min_d
                    penalty += mu * overlap * overlap
                    grad_rad[i] += 2.0 * mu * overlap
                    grad_rad[j] += 2.0 * mu * overlap
                    # Push apart randomly
                    rand_dir = np.random.randn(2)
                    rand_dir /= np.linalg.norm(rand_dir)
                    grad_pos[i] += 2.0 * mu * overlap * rand_dir
                    grad_pos[j] -= 2.0 * mu * overlap * rand_dir
        
        return penalty, grad_pos, grad_rad
    
    def boundary_penalty(radii, centers, mu=1000.0):
        """
        Penalty for circles going outside [0,1]^2.
        For each circle i: 
          left: if x_i < r_i, penalty = mu*(r_i - x_i)^2
          right: if x_i > 1-r_i, penalty = mu*(x_i - (1-r_i))^2
          etc.
        """
        n = len(radii)
        penalty = 0.0
        grad_pos = np.zeros((n, 2))
        grad_rad = np.zeros(n)
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left boundary
            if x < r:
                viol = r - x
                penalty += mu * viol * viol
                grad_pos[i, 0] -= 2.0 * mu * viol
                grad_rad[i] += 2.0 * mu * viol
            
            # Right boundary
            if x > 1.0 - r:
                viol = x - (1.0 - r)
                penalty += mu * viol * viol
                grad_pos[i, 0] += 2.0 * mu * viol
                grad_rad[i] -= 2.0 * mu * viol
            
            # Bottom boundary
            if y < r:
                viol = r - y
                penalty += mu * viol * viol
                grad_pos[i, 1] -= 2.0 * mu * viol
                grad_rad[i] += 2.0 * mu * viol
            
            # Top boundary
            if y > 1.0 - r:
                viol = y - (1.0 - r)
                penalty += mu * viol * viol
                grad_pos[i, 1] += 2.0 * mu * viol
                grad_rad[i] -= 2.0 * mu * viol
        
        return penalty, grad_pos, grad_rad
    
    def objective_and_grad(vars_):
        """
        vars_ = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        Objective: -sum(r_i) + overlap_penalty + boundary_penalty
        """
        n = len(vars_) // 3
        centers = vars_[0:2*n:3].reshape(n, 2)  # This is wrong, let me fix
        # Actually: vars_ = [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        x = vars_[0::3]
        y = vars_[1::3]
        radii = vars_[2::3]
        centers = np.column_stack([x, y])
        
        # Objective: maximize sum(radii), so minimize -sum(radii)
        obj = -np.sum(radii)
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0  # d(-sum(r))/dr_i = -1
        
        # Overlap penalty
        op, op_grad_pos, op_grad_rad = compute_overlap_penalty(radii, centers, mu=100.0)
        obj += op
        grad[0::3] += op_grad_pos[:, 0]
        grad[1::3] += op_grad_pos[:, 1]
        grad[2::3] += op_grad_rad
        
        # Boundary penalty
        bp, bp_grad_pos, bp_grad_rad = boundary_penalty(radii, centers, mu=1000.0)
        obj += bp
        grad[0::3] += bp_grad_pos[:, 0]
        grad[1::3] += bp_grad_pos[:, 1]
        grad[2::3] += bp_grad_rad
        
        return obj, grad
    
    def is_valid(centers, radii, tol=1e-6):
        """Check if configuration is valid (all circles inside, no overlaps)."""
        n = len(radii)
        # Check boundaries
        for i in range(n):
            if centers[i, 0] - radii[i] < -tol or centers[i, 0] + radii[i] > 1.0 + tol:
                return False
            if centers[i, 1] - radii[i] < -tol or centers[i, 1] + radii[i] > 1.0 + tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def polish(centers, radii):
        """
        Use L-BFGS-B to polish the configuration.
        Variables: positions and radii.
        Constraints: radii >= 0, positions in [0,1] (loose), and we rely on penalties for the rest.
        """
        n = len(radii)
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        
        # Bounds: x in [0,1], y in [0,1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((1e-6, 0.5))  # r
        
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
        )
        
        new_vars = result.x
        new_centers = np.column_stack([new_vars[0::3], new_vars[1::3]])
        new_radii = new_vars[2::3]
        
        return new_centers, new_radii
    
    def physics_relax(centers, radii, num_steps=2000, dt=0.01, temp=0.01):
        """
        Simple physics-based relaxation: push overlapping circles apart.
        """
        n = len(radii)
        centers = centers.copy()
        radii = radii.copy()
        
        for step in range(num_steps):
            forces = np.zeros((n, 2))
            forces_r = np.zeros(n)
            
            # Pairwise repulsion
            for i in range(n):
                for j in range(i+1, n):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    min_d = radii[i] + radii[j]
                    if d < min_d and d > 1e-12:
                        overlap = min_d - d
                        force_mag = overlap
                        forces[i] += (d_vec / d) * force_mag
                        forces[j] -= (d_vec / d) * force_mag
                        # Slightly shrink radii if overlapping
                        forces_r[i] -= 0.1 * overlap
                        forces_r[j] -= 0.1 * overlap
            
            # Boundary forces
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x < r:
                    forces[i, 0] += (r - x) * 2.0
                    forces_r[i] -= (r - x) * 0.5
                if x > 1.0 - r:
                    forces[i, 0] -= (x - (1.0 - r)) * 2.0
                    forces_r[i] -= (x - (1.0 - r)) * 0.5
                if y < r:
                    forces[i, 1] += (r - y) * 2.0
                    forces_r[i] -= (r - y) * 0.5
                if y > 1.0 - r:
                    forces[i, 1] -= (y - (1.0 - r)) * 2.0
                    forces_r[i] -= (y - (1.0 - r)) * 0.5
            
            # Update
            centers += dt * forces
            radii += dt * forces_r
            radii = np.maximum(radii, 1e-6)
            centers = np.clip(centers, 0, 1)
            
            # Add small random perturbation to escape local minima
            if step % 100 == 0:
                centers += temp * np.random.randn(n, 2) * 0.01
                centers = np.clip(centers, 0, 1)
        
        return centers, radii
    
    def initial_config(seed):
        """Generate an initial random configuration."""
        rng = np.random.RandomState(seed)
        centers = rng.uniform(0.1, 0.9, size=(N, 2))
        radii = rng.uniform(0.02, 0.08, size=N)
        return centers, radii
    
    best_score = -np.inf
    best_result = None
    
    # Multi-start
    max_starts = 20
    for start in range(max_starts):
        if time.time() > deadline:
            break
        
        centers, radii = initial_config(start * 12345 + 67890)
        
        # Physics relaxation
        centers, radii = physics_relax(centers, radii, num_steps=1000, dt=0.02, temp=0.01)
        
        # Polish with L-BFGS-B
        centers, radii = polish(centers, radii)
        
        # Check validity
        if is_valid(centers, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([centers, radii])
        
        # Also try a second polish pass
        if time.time() > deadline:
            break
        centers, radii = polish(centers, radii)
        if is_valid(centers, radii):
            score = np.sum(radii)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([centers, radii])
    
    # If no valid result found, fall back to a simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    # Final validation and cleanup
    centers = best_result[:, :2]
    radii = best_result[:, 2]
    
    # Ensure all circles are strictly inside
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = max(radii[i], 1e-8)
    
    return np.column_stack([centers, radii])
# EVOLVE-BLOCK-END
