"""
Pack 200 equal circles in a unit square using multi-start gradient-based optimization
with soft-min objective, force-directed repulsion, and diverse initializations.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200
_HARD_LIMIT = 145  # Safety margin before 150s kill

def _get_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    
    wall_dists = np.array([
        np.min(pts[:, 0]) + 0.5,
        0.5 - np.max(pts[:, 0]),
        np.min(pts[:, 1]) + 0.5,
        0.5 - np.max(pts[:, 1])
    ])
    min_wall = np.min(wall_dists)
    
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    min_pair = np.min(D) / 2.0
    
    return min(min_wall, min_pair)

def _objective_and_grad(x, beta):
    """
    Objective: minimize -soft_min(constraints)
    Constraints: wall distances and pairwise half-distances
    """
    pts = x.reshape((N, 2))
    
    # Build constraint values
    i_idx, j_idx = np.triu_indices(N, k=1)
    num_pairs = len(i_idx)
    
    C = np.empty(4*N + num_pairs)
    
    # Wall constraints
    C[0:N] = pts[:, 0] + 0.5
    C[N:2*N] = 0.5 - pts[:, 0]
    C[2*N:3*N] = pts[:, 1] + 0.5
    C[3*N:4*N] = 0.5 - pts[:, 1]
    
    # Pair constraints
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    C[4*N:] = dists / 2.0
    
    # Soft min via log-sum-exp
    y = -beta * C
    max_y = np.max(y)
    if not np.isfinite(max_y):
        return -np.min(C), np.zeros_like(x)
    
    shifted = y - max_y
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    if sum_exp < 1e-300:
        return -np.min(C), np.zeros_like(x)
    
    soft_min_val = -(max_y + np.log(sum_exp)) / beta
    obj_val = -soft_min_val
    
    # Weights
    weights = exp_shifted / sum_exp
    
    # Gradient accumulation
    grad_pts = np.zeros((N, 2))
    
    # Wall gradients
    grad_pts[:, 0] += weights[0:N] - weights[N:2*N]
    grad_pts[:, 1] += weights[2*N:3*N] - weights[3*N:4*N]
    
    # Pair gradients
    valid = dists > 1e-10
    dx_safe = np.where(valid, dx, 1.0)
    dy_safe = np.where(valid, dy, 1.0)
    dists_safe = np.where(valid, dists, 1.0)
    
    pair_weights = weights[4*N:]
    
    grad_pair_x_i = np.where(valid, dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_x_j = np.where(valid, -dx_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_i = np.where(valid, dy_safe / (2.0 * dists_safe), 0.0)
    grad_pair_y_j = np.where(valid, -dy_safe / (2.0 * dists_safe), 0.0)
    
    np.add.at(grad_pts[:, 0], i_idx, grad_pair_x_i * pair_weights)
    np.add.at(grad_pts[:, 0], j_idx, grad_pair_x_j * pair_weights)
    np.add.at(grad_pts[:, 1], i_idx, grad_pair_y_i * pair_weights)
    np.add.at(grad_pts[:, 1], j_idx, grad_pair_y_j * pair_weights)
    
    grad_pts = -grad_pts / beta
    
    return obj_val, grad_pts.flatten()

def _force_directed_step(pts, n_steps=500, lr=0.001, repulsion_range=0.15):
    """
    Force-directed relaxation: push overlapping/nearby circles apart.
    Uses inverse-square repulsion for close pairs.
    """
    t0 = time.time()
    for step in range(n_steps):
        if time.time() - t0 > 30:  # Time guard for this phase
            break
        
        D = cdist(pts, pts)
        np.fill_diagonal(D, np.inf)
        
        # Repulsion force: strong for close pairs
        # F = k / d^2 for d < repulsion_range, else 0
        mask = D < repulsion_range
        forces = np.zeros_like(pts)
        
        if np.any(mask):
            # For each pair (i,j) with d < range, add repulsive force
            for i in range(N):
                for j in range(N):
                    if i != j and D[i, j] < repulsion_range and D[i, j] > 1e-10:
                        d = D[i, j]
                        f = 1.0 / (d * d) * (D[i, j] / max(D[i, j], 1e-10))
                        # Force direction: from j to i (push i away from j)
                        dir_x = (pts[i, 0] - pts[j, 0]) / d
                        dir_y = (pts[i, 1] - pts[j, 1]) / d
                        forces[i, 0] += f * dir_x
                        forces[i, 1] += f * dir_y
        
        # Wall forces: push back if too close to walls
        wall_margin = 0.01
        for i in range(N):
            if pts[i, 0] < -0.5 + wall_margin:
                forces[i, 0] += 10.0
            elif pts[i, 0] > 0.5 - wall_margin:
                forces[i, 0] -= 10.0
            if pts[i, 1] < -0.5 + wall_margin:
                forces[i, 1] += 10.0
            elif pts[i, 1] > 0.5 - wall_margin:
                forces[i, 1] -= 10.0
        
        # Update positions
        pts = pts + lr * forces
        pts = np.clip(pts, -0.5 + 1e-6, 0.5 - 1e-6)
    
    return pts

def _generate_hex_initial(s, offset_x=0.0, offset_y=0.0):
    """Generate hexagonal lattice with given spacing and offset."""
    points = []
    row_h = s * np.sqrt(3) / 2
    
    y = -0.5 + s/2 + offset_y
    row_idx = 0
    while y < 0.5:
        shift = (s / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 + s/2 + shift + offset_x
        while x < 0.5:
            points.append([x, y])
            x += s
        y += row_h
        row_idx += 1
    
    pts = np.array(points) if points else np.empty((0, 2))
    
    # Filter to inside square
    if len(pts) > 0:
        mask = (pts[:, 0] > -0.49) & (pts[:, 0] < 0.49) & (pts[:, 1] > -0.49) & (pts[:, 1] < 0.49)
        pts = pts[mask]
    
    if len(pts) < N:
        # Pad with random points near edges if needed
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.45, 0.45, (1, 2))])
        pts = np.clip(pts, -0.49, 0.49)
    
    # Select N points closest to center
    if len(pts) > N:
        dists_to_center = np.sum(pts**2, axis=1)
        indices = np.argsort(dists_to_center)
        pts = pts[indices[:N]]
    
    return pts[:N]

def _generate_random_initial():
    """Random initial configuration."""
    return np.random.uniform(-0.45, 0.45, (N, 2))

def _generate_jittered_hex(s, jitter=0.01):
    """Hex lattice with random jitter to break symmetry."""
    pts = _generate_hex_initial(s)
    pts = pts + np.random.uniform(-jitter, jitter, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _run_optimization(pts, time_limit, betas=None, max_iter_per_beta=200):
    """Run L-BFGS-B optimization with annealing betas."""
    if betas is None:
        betas = [5, 10, 20, 50, 100, 200, 400]
    
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
        
        try:
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'ftol': 1e-14, 'gtol': 1e-14, 'maxls': 50}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5 + 1e-6, 0.5 - 1e-6)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
        
        # Check time
        if time.time() - t_start > time_limit:
            break
    
    return best_r, best_x.reshape((N, 2))

def _polish(pts, time_limit):
    """Final high-precision polish with high beta."""
    t_start = time.time()
    x0 = pts.flatten()
    bounds = [(-0.5 + 1e-6, 0.5 - 1e-6)] * (2 * N)
    
    best_r = _get_radius(pts)
    best_x = x0.copy()
    
    for beta in [500, 1000, 2000]:
        if time.time() - t_start > time_limit:
            break
        
        def fun_and_grad(x):
            val, g = _objective_and_grad(x, beta)
            return val, g
        
        try:
            res = minimize(
                fun_and_grad, 
                x0, 
                jac=True, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-15, 'maxls': 100}
            )
            x0 = res.x
            pts_curr = x0.reshape((N, 2))
            pts_curr = np.clip(pts_curr, -0.5 + 1e-6, 0.5 - 1e-6)
            
            r = _get_radius(pts_curr)
            if r > best_r:
                best_r = r
                best_x = x0.copy()
                
        except Exception:
            pass
    
    return best_r, best_x.reshape((N, 2))

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    best_r = 0.0
    best_pts = None
    
    # Define diverse initializations
    # Target radius ~0.0366, so spacing ~0.0732
    spacings = [0.073, 0.074, 0.075, 0.076, 0.077]
    offsets = [(0.0, 0.0), (0.01, 0.0), (0.0, 0.01), (0.01, 0.01), (-0.01, 0.0)]
    
    initializations = []
    
    # Hex lattices with various spacings and offsets
    for s in spacings:
        for ox, oy in offsets[:2]:  # Limit combinations
            initializations.append(_generate_hex_initial(s, ox, oy))
    
    # Jittered hex
    for s in [0.073, 0.075]:
        initializations.append(_generate_jittered_hex(s, jitter=0.008))
    
    # Random
    initializations.append(_generate_random_initial())
    
    # Allocate time: first pass ~60s, second pass ~40s, polish ~30s
    first_pass_time = 55
    second_pass_time = 40
    polish_time = 30
    
    # Phase 1: First pass on all initializations
    elapsed = time.time() - t0
    remaining = _HARD_LIMIT - elapsed - second_pass_time - polish_time - 5
    if remaining > 10:
        time_per_init = max(remaining / len(initializations), 2.0)
        
        for i, init_pts in enumerate(initializations):
            if time.time() - t0 > _HARD_LIMIT - second_pass_time - polish_time - 5:
                break
            
            # Force-directed pre-relaxation
            if time.time() - t0 < _HARD_LIMIT - 40:
                init_pts = _force_directed_step(init_pts, n_steps=200, lr=0.0005)
            
            r, pts_opt = _run_optimization(init_pts, time_limit=time_per_init, betas=[5, 10, 20, 50, 100])
            
            if r > best_r:
                best_r = r
                best_pts = pts_opt
    
    # Phase 2: Second pass on best candidates
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - elapsed - polish_time - 3
        if remaining > 10:
            # Try a few more betas on the best solution
            r, pts_opt = _run_optimization(best_pts, time_limit=remaining, betas=[50, 100, 200, 400])
            if r > best_r:
                best_r = r
                best_pts = pts_opt
    
    # Phase 3: Polish
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _HARD_LIMIT - elapsed - 2
        if remaining > 5:
            r, pts_polish = _polish(best_pts, time_limit=remaining)
            if r > best_r:
                best_r = r
                best_pts = pts_polish
    
    # Final validation and clipping
    if best_pts is None:
        best_pts = _generate_hex_initial(0.073)
    
    best_pts = np.clip(best_pts, -0.5 + 1e-6, 0.5 - 1e-6)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        best_pts = np.where(np.isfinite(best_pts), best_pts, 0.0)
    
    return best_pts

if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
