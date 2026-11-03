"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using a hexagonal lattice.
2. Use Simulated Annealing to explore the landscape and escape local minima.
3. Use a smooth soft-min objective function and L-BFGS-B for final high-precision polish.
4. Use a deadline guard to ensure termination within 120s.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin before 120s limit

    # --- Step 1: Generate good initial positions ---
    def hexagonal_positions(n_target):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        
        best_points = None
        best_count_diff = float('inf')
        
        for spacing_mult in np.linspace(0.85, 1.15, 20):
            spacing = base_spacing * spacing_mult
            
            a = spacing
            b = a * np.sqrt(3) / 2
            k = int(1.0 / spacing) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.98**2:
                        points.append((x, y))
            
            points = np.array(points)
            if len(points) == 0:
                continue
            
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx]
            
            if len(points) >= n_target:
                points = points[:n_target]
                diff = abs(len(points) - n_target)
                if diff < best_count_diff:
                    best_count_diff = diff
                    best_points = points
            
            if len(points) == n_target:
                best_points = points
                break
        
        if best_points is None or len(best_points) != n_target:
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points
    
    positions = hexagonal_positions(N)
    
    def compute_radius(pos):
        """Compute the current minimum radius."""
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist) / 2.0
        
        return min(min_boundary, min_pair_dist)
    
    current_r = compute_radius(positions)
    if current_r < 0.04:
        max_norm = np.max(np.linalg.norm(positions, axis=1))
        if max_norm > 0:
            positions = positions * (0.95 / max_norm)
        current_r = compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r

    # --- Step 2: Simulated Annealing for Global Optimization ---
    # We use a simple SA loop to maximize the minimum distance.
    # This helps escape the local minima of the hexagonal start.
    
    def sa_optimize(pos, current_r, time_limit):
        """Run Simulated Annealing to improve the packing."""
        pos = pos.copy()
        r = current_r
        best_pos = pos.copy()
        best_r = r
        
        # SA parameters
        T = 0.005  # Initial temperature
        T_min = 1e-6
        cooling = 0.9995
        steps_per_iter = 1000
        
        # Precompute pair indices for efficiency
        i_idx, j_idx = np.triu_indices(N, k=1)
        
        # We'll use a simple random perturbation strategy
        # Perturb a random subset of points
        
        while time.time() - start_time < time_limit:
            # Pick a random subset of points to perturb
            n_perturb = np.random.randint(1, 20)
            idx_perturb = np.random.choice(N, n_perturb, replace=False)
            
            # Generate random perturbations
            scale = T * 0.5
            perturbations = np.random.randn(n_perturb, 2) * scale
            
            # Apply perturbations
            old_pos = pos[idx_perturb].copy()
            pos[idx_perturb] += perturbations
            
            # Ensure points stay within the unit disk
            norms = np.linalg.norm(pos[idx_perturb], axis=1)
            mask = norms > 0.999
            if np.any(mask):
                pos[idx_perturb][mask] *= 0.999 / norms[mask]
            
            # Compute new radius
            new_r = compute_radius(pos)
            
            # Accept or reject
            if new_r > r:
                r = new_r
                if r > best_r:
                    best_r = r
                    best_pos = pos.copy()
            else:
                # Metropolis criterion
                delta = r - new_r
                if np.random.rand() < np.exp(-delta / T):
                    r = new_r
                else:
                    # Reject
                    pos[idx_perturb] = old_pos
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_pos, best_r

    # Run SA for a portion of the time budget
    sa_time_limit = start_time + (deadline - start_time) * 0.6
    if time.time() < sa_time_limit:
        best_positions, best_r = sa_optimize(positions, best_r, sa_time_limit)
        positions = best_positions.copy()

    # --- Step 3: Soft-min optimization with L-BFGS-B ---
    # We maximize the soft-min of the gaps.
    # Gaps: d_ij / 2 for pairs, 1 - |p_i| for boundaries.
    # Soft-min: S(x, k) = -1/k * ln( sum exp(-k * gap) )
    # We minimize f(x) = -S(x, k) = 1/k * ln( sum exp(-k * gap) )
    
    # Precompute pair indices for efficiency
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)
    
    def soft_min_objective_and_grad(x, k):
        pos = x.reshape(N, 2)
        
        # Pairwise distances
        diff = pos[i_idx] - pos[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        safe_dists = np.maximum(dists, 1e-9)
        
        # Boundary distances
        rad = np.linalg.norm(pos, axis=1)
        b_vals = 1.0 - rad
        
        # Gaps
        pair_gaps = dists / 2.0
        all_gaps = np.concatenate([pair_gaps, b_vals])
        
        # Soft-min calculation
        # To avoid overflow/underflow, subtract min
        min_gap = np.min(all_gaps)
        exponents = -k * (all_gaps - min_gap)
        weights = np.exp(exponents)
        sum_weights = np.sum(weights)
        
        # Objective: f = -1/k * ln(sum_weights) - min_gap
        # Actually, S = min_gap - 1/k * ln(sum_weights)
        # We want to maximize S, so minimize -S
        # -S = -min_gap + 1/k * ln(sum_weights)
        
        obj_val = -min_gap + (1.0 / k) * np.log(sum_weights)
        
        # Gradient
        # dS/dx = -1/k * (1/sum_weights) * sum(-k * exp(-k * gap) * d gap/dx)
        # dS/dx = (1/sum_weights) * sum(exp(-k * gap) * d gap/dx)
        # d(-S)/dx = - dS/dx
        
        # Weights normalized
        w = weights / sum_weights
        
        # Gradient of gaps
        # d(pair_gap)/d(p_i) = 0.5 * (p_i - p_j) / dist
        # d(b_val)/d(p_i) = -p_i / rad
        
        grad = np.zeros_like(pos)
        
        # Pairwise gradients
        # d(pair_gaps)/dx
        # For each pair (i, j), d_gap = 0.5 * (p_i - p_j) / dist
        # Contribution to grad[i] += w_pair * 0.5 * (p_i - p_j) / dist
        # Contribution to grad[j] += w_pair * 0.5 * (p_j - p_i) / dist
        
        unit_vecs = diff / safe_dists[:, np.newaxis]
        # d_gap_i = 0.5 * unit_vec
        # d_gap_j = -0.5 * unit_vec
        
        pair_grads = 0.5 * unit_vecs * w[:n_pairs, np.newaxis]
        
        np.add.at(grad, i_idx, pair_grads)
        np.add.at(grad, j_idx, -pair_grads)
        
        # Boundary gradients
        # d(b_val_i)/d(p_i) = -p_i / rad_i
        safe_rad = np.maximum(rad, 1e-9)
        bound_grads = -pos / safe_rad[:, np.newaxis] * w[n_pairs:, np.newaxis]
        grad += bound_grads
        
        # Gradient of objective f = -S is -grad_S
        # So return -grad
        return obj_val, -grad

    # Optimization loop with annealing
    k = 10.0
    k_max = 1000.0
    k_step = 1.2
    
    while time.time() - start_time < deadline:
        # Define objective and gradient for current k
        def objective(x):
            val, _ = soft_min_objective_and_grad(x, k)
            return val
        
        def grad(x):
            _, g = soft_min_objective_and_grad(x, k)
            return g
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective,
                positions.flatten(),
                method='L-BFGS-B',
                jac=grad,
                options={'maxiter': 50, 'gtol': 1e-5}
            )
            
            if res.success or res.fun < best_r: # Note: res.fun is -S, so lower is better for -S, higher S is better
                # Check if we improved
                new_pos = res.x.reshape(N, 2)
                new_r = compute_radius(new_pos)
                
                if new_r > best_r:
                    best_r = new_r
                    best_positions = new_pos.copy()
                    positions = new_pos.copy()
        except Exception:
            pass
        
        # Increase k to sharpen the minimum
        k *= k_step
        if k > k_max:
            k = k_max
            
        # Periodically re-evaluate or break if no improvement
        # We rely on the deadline and the fact that L-BFGS-B is fast
        
    final_r = compute_radius(best_positions)
    return best_positions

def _validate_and_fix(pos):
    """Ensure positions are valid: finite, inside unit disk, no coincident points."""
    pos = np.array(pos, dtype=np.float64)
    if pos.shape != (N, 2):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    if not np.all(np.isfinite(pos)):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.1, 0.9, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
    norms = np.linalg.norm(pos, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        pos[mask] = pos[mask] * (0.999 / norms[mask])
        
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-4:
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-4:
                    pos[i] += np.random.randn(2) * 1e-3
                    pos[j] -= np.random.randn(2) * 1e-3
                    n = np.linalg.norm(pos[i])
                    if n > 0.999:
                        pos[i] = pos[i] * (0.999 / n)
                    n = np.linalg.norm(pos[j])
                    if n > 0.999:
                        pos[j] = pos[j] * (0.999 / n)
                        
    return pos

_original_construct = construct

def construct():
    try:
        pos = _original_construct()
        pos = _validate_and_fix(pos)
        return pos
    except Exception:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = np.linspace(0.05, 0.95, N)
        pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        return pos
# EVOLVE-BLOCK-END
