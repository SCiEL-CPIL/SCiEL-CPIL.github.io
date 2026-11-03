"""Packing 300 equal circles in the unit disk.
Strategy:
1. Generate initial positions using a hexagonal lattice.
2. Use L-BFGS-B with soft-min objective for coarse optimization.
3. Active-set Newton refinement on binding constraints for precision.
4. Multiple restarts with perturbation to escape local minima.
5. Deadline guard to ensure termination within 120s.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Leave 5s margin before 150s limit

    def compute_radius(pos):
        """Compute the current minimum radius efficiently."""
        boundary = 1.0 - np.linalg.norm(pos, axis=1)
        min_boundary = np.min(boundary)
        
        # Use chunked computation to avoid memory issues
        min_pair_dist = np.inf
        chunk_size = 100
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            for s2 in range(start, end):
                diffs = pos[s2] - pos[s2+1:]
                dists = np.linalg.norm(diffs, axis=1)
                if len(dists) > 0:
                    min_pair_dist = min(min_pair_dist, np.min(dists))
        
        return min(min_boundary, min_pair_dist / 2.0)

    def hexagonal_positions(n_target):
        """Generate n_target points in a hexagonal pattern within the unit disk."""
        base_spacing = 1.0 / np.sqrt(n_target * 2.0 / np.pi)
        
        best_points = None
        
        for spacing_mult in np.linspace(0.88, 1.12, 15):
            spacing = base_spacing * spacing_mult
            
            a = spacing
            b = a * np.sqrt(3) / 2
            k = int(1.0 / spacing) + 5
            points = []
            for j in range(-k, k + 1):
                y = j * b
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * a
                    if x*x + y*y <= 0.97**2:
                        points.append((x, y))
            
            if len(points) == 0:
                continue
            
            points = np.array(points)
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            points = points[idx]
            
            if len(points) >= n_target:
                points = points[:n_target]
                if best_points is None:
                    best_points = points
            
            if len(points) == n_target:
                best_points = points
                break
        
        if best_points is None or len(best_points) != n_target:
            angles = np.linspace(0, 2*np.pi, n_target, endpoint=False)
            radii = np.linspace(0.05, 0.95, n_target)
            best_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        return best_points

    # Precompute pair indices
    i_idx, j_idx = np.triu_indices(N, k=1)
    n_pairs = len(i_idx)

    def soft_min_obj_grad(x, k):
        """Soft-min objective and gradient for L-BFGS-B."""
        pos = x.reshape(N, 2)
        
        diff = pos[i_idx] - pos[j_idx]
        dists = np.linalg.norm(diff, axis=1)
        safe_dists = np.maximum(dists, 1e-10)
        
        rad = np.linalg.norm(pos, axis=1)
        b_vals = 1.0 - rad
        
        pair_gaps = dists / 2.0
        all_gaps = np.concatenate([pair_gaps, b_vals])
        
        min_gap = np.min(all_gaps)
        exponents = -k * (all_gaps - min_gap)
        weights = np.exp(exponents)
        sum_weights = np.sum(weights)
        
        obj_val = -min_gap + (1.0 / k) * np.log(sum_weights)
        
        w = weights / sum_weights
        
        grad = np.zeros_like(pos)
        
        unit_vecs = diff / safe_dists[:, np.newaxis]
        pair_grads = 0.5 * unit_vecs * w[:n_pairs, np.newaxis]
        
        np.add.at(grad, i_idx, pair_grads)
        np.add.at(grad, j_idx, -pair_grads)
        
        safe_rad = np.maximum(rad, 1e-10)
        bound_grads = -pos / safe_rad[:, np.newaxis] * w[n_pairs:, np.newaxis]
        grad += bound_grads
        
        return obj_val, -grad

    def newton_refine(pos, r_target, max_iter=50):
        """Active-set Newton refinement on binding constraints."""
        pos = pos.copy()
        n_active = 0
        
        for it in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
                
            # Compute all gaps
            diff = pos[i_idx] - pos[j_idx]
            dists = np.linalg.norm(diff, axis=1)
            safe_dists = np.maximum(dists, 1e-10)
            pair_gaps = dists / 2.0
            
            rad = np.linalg.norm(pos, axis=1)
            b_vals = 1.0 - rad
            safe_rad = np.maximum(rad, 1e-10)
            
            all_gaps = np.concatenate([pair_gaps, b_vals])
            min_gap = np.min(all_gaps)
            
            # Find active constraints (within tolerance of min_gap)
            tol = 1e-8
            active_mask = all_gaps < min_gap + tol
            n_active = np.sum(active_mask)
            
            if n_active == 0:
                break
            
            # Build the active constraint system
            # For each active pair constraint: |p_i - p_j| = 2*r_target
            # For each active boundary constraint: |p_i| = 1 - r_target
            # We want to find delta such that these constraints are satisfied
            
            # Use a simple iterative approach: project onto active constraints
            # Each active constraint defines a direction to move
            
            delta = np.zeros_like(pos)
            
            # Pair constraints
            active_pairs = np.where(active_mask[:n_pairs])[0]
            for idx in active_pairs:
                i, j = i_idx[idx], j_idx[idx]
                d = dists[idx]
                if d < 1e-10:
                    continue
                # Current gap is d/2, target is r_target
                # Need to increase distance to 2*r_target
                # Move i and j apart along the connecting vector
                needed = (2.0 * r_target - d) / 2.0
                if abs(needed) > 1e-12:
                    unit = (pos[i] - pos[j]) / d
                    delta[i] += needed * unit
                    delta[j] -= needed * unit
            
            # Boundary constraints
            active_bounds = np.where(active_mask[n_pairs:])[0]
            for idx in active_bounds:
                i = idx
                r = rad[i]
                if r < 1e-10:
                    continue
                # Current boundary gap is 1 - r, target is r_target
                # Need |p_i| = 1 - r_target
                needed = (1.0 - r_target - r)
                if abs(needed) > 1e-12:
                    unit = pos[i] / r
                    delta[i] += needed * unit
            
            # Apply delta with damping
            step_size = 0.5
            new_pos = pos + step_size * delta
            
            # Validate
            new_r = compute_radius(new_pos)
            if new_r >= r_target - 1e-10:
                pos = new_pos
            else:
                # Try smaller step
                step_size *= 0.5
                new_pos = pos + step_size * delta
                new_r = compute_radius(new_pos)
                if new_r >= r_target - 1e-10:
                    pos = new_pos
                else:
                    break
            
            # Check convergence
            if np.max(np.abs(delta)) < 1e-10:
                break
        
        return pos

    def polish_with_newton(pos, r_current):
        """Iteratively increase radius using Newton refinement."""
        pos = pos.copy()
        r = r_current
        
        for _ in range(20):
            if time.time() - start_time > deadline - 5:
                break
            
            # Try to increase r slightly
            r_new = r * 1.0001 + 1e-10
            pos_new = newton_refine(pos, r_new, max_iter=30)
            r_new_actual = compute_radius(pos_new)
            
            if r_new_actual > r + 1e-12:
                pos = pos_new
                r = r_new_actual
            else:
                break
        
        return pos, r

    # --- Main optimization loop ---
    positions = hexagonal_positions(N)
    current_r = compute_radius(positions)
    
    # Scale to fit if needed
    max_norm = np.max(np.linalg.norm(positions, axis=1))
    if max_norm > 0.95:
        positions = positions * (0.95 / max_norm)
        current_r = compute_radius(positions)
    
    best_positions = positions.copy()
    best_r = current_r
    
    # Phase 1: L-BFGS-B with annealing
    k = 20.0
    k_max = 500.0
    
    while time.time() - start_time < deadline - 30:
        def objective(x):
            val, _ = soft_min_obj_grad(x, k)
            return val
        
        def grad(x):
            _, g = soft_min_obj_grad(x, k)
            return g
        
        try:
            res = minimize(
                objective,
                positions.flatten(),
                method='L-BFGS-B',
                jac=grad,
                options={'maxiter': 30, 'gtol': 1e-6}
            )
            
            new_pos = res.x.reshape(N, 2)
            new_r = compute_radius(new_pos)
            
            if new_r > best_r:
                best_r = new_r
                best_positions = new_pos.copy()
                positions = new_pos.copy()
        except Exception:
            pass
        
        k *= 1.15
        if k > k_max:
            k = k_max
    
    # Phase 2: Newton refinement
    positions = best_positions.copy()
    best_r = compute_radius(positions)
    
    while time.time() - start_time < deadline - 10:
        positions, best_r = polish_with_newton(positions, best_r)
        
        # Check if we can improve further
        if time.time() - start_time > deadline - 10:
            break
        
        # Small perturbation and re-optimize
        if best_r < 0.0531:  # Still below known best
            rng = np.random.default_rng()
            perturbed = positions + rng.normal(0, 0.001, size=(N, 2))
            perturbed_r = compute_radius(perturbed)
            
            if perturbed_r > best_r * 0.99:
                # Re-run L-BFGS-B from perturbed position
                def objective(x):
                    val, _ = soft_min_obj_grad(x, 50.0)
                    return val
                
                def grad(x):
                    _, g = soft_min_obj_grad(x, 50.0)
                    return g
                
                try:
                    res = minimize(
                        objective,
                        perturbed.flatten(),
                        method='L-BFGS-B',
                        jac=grad,
                        options={'maxiter': 20, 'gtol': 1e-6}
                    )
                    
                    new_pos = res.x.reshape(N, 2)
                    new_r = compute_radius(new_pos)
                    
                    if new_r > best_r:
                        best_r = new_r
                        best_positions = new_pos.copy()
                        positions = new_pos.copy()
                except Exception:
                    pass
            
            # Polish again
            positions, best_r = polish_with_newton(positions, best_r)
    
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
        
    # Check for coincident points
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-6:
        # Separate coincident points
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 1e-6:
                    rng = np.random.default_rng()
                    pos[i] += rng.normal(0, 1e-4, size=2)
                    pos[j] -= rng.normal(0, 1e-4, size=2)
                    n_i = np.linalg.norm(pos[i])
                    if n_i > 0.999:
                        pos[i] = pos[i] * (0.999 / n_i)
                    n_j = np.linalg.norm(pos[j])
                    if n_j > 0.999:
                        pos[j] = pos[j] * (0.999 / n_j)
                        
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
