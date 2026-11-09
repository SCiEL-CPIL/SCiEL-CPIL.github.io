"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centres):
    """Compute the maximum common radius for given centres."""
    n = len(centres)
    if n == 0:
        return 0.0
    # Distance to boundary
    d_boundary = np.min(0.5 - np.abs(centres), axis=1)
    min_boundary = np.min(d_boundary)
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centres[i] - centres[j])
            if d < min_pair:
                min_pair = d
    return min(min_pair / 2.0, min_boundary)


def _soft_min_radius(centres, beta=50.0):
    """
    Smooth surrogate for the radius objective.
    Uses log-sum-exp to create a smooth approximation of the min over constraints.
    """
    n = len(centres)
    if n == 0:
        return 0.0
    
    # Boundary constraints: for each point i and each coordinate k, constraint is 0.5 - |c_ik|
    # We want min_i min_k (0.5 - |c_ik|)
    # Soft-min over all these values
    boundary_vals = []
    for i in range(n):
        for k in range(3):
            boundary_vals.append(0.5 - abs(centres[i, k]))
    boundary_vals = np.array(boundary_vals)
    
    # Pairwise constraints: for each pair (i,j), constraint is |c_i - c_j| / 2
    pair_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            pair_vals.append(np.linalg.norm(centres[i] - centres[j]) / 2.0)
    pair_vals = np.array(pair_vals)
    
    all_vals = np.concatenate([boundary_vals, pair_vals])
    
    # Soft-min using log-sum-exp: -log(sum(exp(-beta * x))) / beta approximates min(x)
    # For numerical stability, subtract max
    shifted = -beta * all_vals
    max_shift = np.max(shifted)
    log_sum = max_shift + np.log(np.sum(np.exp(shifted - max_shift)))
    soft_min = -log_sum / beta
    
    return soft_min


def _soft_min_radius_grad(centres, beta=50.0):
    """
    Compute soft-min radius and its gradient.
    """
    n = len(centres)
    if n == 0:
        return 0.0, np.zeros_like(centres)
    
    # Collect all constraint values and their gradients
    vals = []
    grads = []
    
    # Boundary constraints
    for i in range(n):
        for k in range(3):
            v = 0.5 - abs(centres[i, k])
            vals.append(v)
            g = np.zeros(3 * n)
            if centres[i, k] >= 0:
                g[i * 3 + k] = -1.0
            else:
                g[i * 3 + k] = 1.0
            grads.append(g)
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            diff = centres[i] - centres[j]
            d = np.linalg.norm(diff)
            if d < 1e-12:
                v = 0.0
                g = np.zeros(3 * n)
            else:
                v = d / 2.0
                g = np.zeros(3 * n)
                g[i * 3:i * 3 + 3] = diff / (2.0 * d)
                g[j * 3:j * 3 + 3] = -diff / (2.0 * d)
            vals.append(v)
            grads.append(g)
    
    vals = np.array(vals)
    grads = np.array(grads)  # shape (m, 3n)
    
    # Soft-min: f = -log(sum(exp(-beta * v))) / beta
    shifted = -beta * vals
    max_shift = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shift)
    log_sum = max_shift + np.log(np.sum(exp_shifted))
    soft_min_val = -log_sum / beta
    
    # Gradient: df/dx = sum_j (exp(-beta*v_j) / sum_k exp(-beta*v_k)) * (-beta) * dv_j/dx / beta
    #            = -sum_j (p_j * dv_j/dx) where p_j = exp(-beta*v_j) / sum_k exp(-beta*v_k)
    p = exp_shifted / np.sum(exp_shifted)
    grad = -np.sum(p[:, np.newaxis] * grads, axis=0)
    
    return soft_min_val, grad


def _fcc_lattice_init():
    """
    Generate FCC lattice points within the cube and select the best 100.
    FCC lattice: points at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) mod 1
    """
    points = []
    # Generate FCC points in a grid
    # FCC unit cell has 4 points. We'll generate a fine grid and pick FCC positions.
    # Actually, let's generate FCC lattice points properly.
    # FCC lattice vectors: a1=(0.5,0.5,0), a2=(0.5,0,0.5), a3=(0,0.5,0.5)
    # Generate points in a range and filter to those inside [-0.5, 0.5]^3
    
    a1 = np.array([0.5, 0.5, 0.0])
    a2 = np.array([0.5, 0.0, 0.5])
    a3 = np.array([0.0, 0.5, 0.5])
    
    # We need to cover the cube [-0.5, 0.5]^3
    # Generate integer combinations i*a1 + j*a2 + k*a3 + offset for offsets in FCC basis
    # FCC basis points: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.5, 0.5, 0.0]),
        np.array([0.5, 0.0, 0.5]),
        np.array([0.0, 0.5, 0.5])
    ]
    
    # Generate points in a range that covers the cube
    # The lattice spacing is about 0.5, so we need indices from -2 to 2 or so
    for offset in offsets:
        for i in range(-3, 4):
            for j in range(-3, 4):
                for k in range(-3, 4):
                    p = i * a1 + j * a2 + k * a3 + offset
                    # Check if inside [-0.5, 0.5]^3
                    if np.all(p >= -0.5 - 1e-9) and np.all(p <= 0.5 + 1e-9):
                        # Clamp to be safely inside
                        p_clamped = np.clip(p, -0.49, 0.49)
                        points.append(p_clamped)
    
    points = np.array(points)
    
    # Remove duplicates
    if len(points) > 0:
        # Use rounding to remove near-duplicates
        rounded = np.round(points, decimals=6)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        points = points[unique_idx]
    
    if len(points) < N:
        # Not enough FCC points, fall back to simple cubic
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Select the N points with the largest minimum distance to boundary
    # For each point, compute min distance to boundary
    d_boundary = np.min(0.5 - np.abs(points), axis=1)
    # Sort by d_boundary descending, take top N
    if len(points) > N:
        idx = np.argsort(-d_boundary)[:N]
        points = points[idx]
    
    return points[:N]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with FCC lattice
    centres = _fcc_lattice_init().astype(np.float64)
    
    # Track best radius
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Optimize using L-BFGS-B on the smooth surrogate
    # We maximize soft_min_radius, so minimize -soft_min_radius
    beta = 30.0
    
    def objective(x):
        centres_flat = x.reshape(-1, 3)
        val, _ = _soft_min_radius_grad(centres_flat, beta)
        return -val
    
    def grad(x):
        centres_flat = x.reshape(-1, 3)
        _, g = _soft_min_radius_grad(centres_flat, beta)
        return g
    
    # Bounds: each coordinate in [-0.5, 0.5]
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Run L-BFGS-B
    x0 = centres.flatten()
    
    # Use multiple restarts with different beta values and small perturbations
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # First pass: optimize with moderate beta
    result = minimize(
        objective, x0, jac=grad, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or result.fun < objective(x0):
        new_centres = result.x.reshape(-1, 3)
        # Clamp to be inside
        new_centres = np.clip(new_centres, -0.5, 0.5)
        r = _compute_radius(new_centres)
        if r > best_radius:
            best_radius = r
            best_centres = new_centres
    
    if time.time() > deadline - 30.0:
        return best_centres
    
    # Second pass: higher beta for sharper approximation
    beta2 = 80.0
    x0 = best_centres.flatten()
    
    def objective2(x):
        centres_flat = x.reshape(-1, 3)
        val, _ = _soft_min_radius_grad(centres_flat, beta2)
        return -val
    
    def grad2(x):
        centres_flat = x.reshape(-1, 3)
        _, g = _soft_min_radius_grad(centres_flat, beta2)
        return g
    
    result2 = minimize(
        objective2, x0, jac=grad2, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result2.success or result2.fun < objective2(x0):
        new_centres = result2.x.reshape(-1, 3)
        new_centres = np.clip(new_centres, -0.5, 0.5)
        r = _compute_radius(new_centres)
        if r > best_radius:
            best_radius = r
            best_centres = new_centres
    
    if time.time() > deadline - 15.0:
        return best_centres
    
    # Third pass: even higher beta
    beta3 = 150.0
    x0 = best_centres.flatten()
    
    def objective3(x):
        centres_flat = x.reshape(-1, 3)
        val, _ = _soft_min_radius_grad(centres_flat, beta3)
        return -val
    
    def grad3(x):
        centres_flat = x.reshape(-1, 3)
        _, g = _soft_min_radius_grad(centres_flat, beta3)
        return g
    
    result3 = minimize(
        objective3, x0, jac=grad3, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-13, 'gtol': 1e-11}
    )
    
    if result3.success or result3.fun < objective3(x0):
        new_centres = result3.x.reshape(-1, 3)
        new_centres = np.clip(new_centres, -0.5, 0.5)
        r = _compute_radius(new_centres)
        if r > best_radius:
            best_radius = r
            best_centres = new_centres
    
    # Final polish: a few more iterations with very high beta
    if time.time() < deadline - 5.0:
        beta4 = 300.0
        x0 = best_centres.flatten()
        
        def objective4(x):
            centres_flat = x.reshape(-1, 3)
            val, _ = _soft_min_radius_grad(centres_flat, beta4)
            return -val
        
        def grad4(x):
            centres_flat = x.reshape(-1, 3)
            _, g = _soft_min_radius_grad(centres_flat, beta4)
            return g
        
        result4 = minimize(
            objective4, x0, jac=grad4, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        
        if result4.success or result4.fun < objective4(x0):
            new_centres = result4.x.reshape(-1, 3)
            new_centres = np.clip(new_centres, -0.5, 0.5)
            r = _compute_radius(new_centres)
            if r > best_radius:
                best_radius = r
                best_centres = new_centres
    
    return best_centres
# EVOLVE-BLOCK-END
