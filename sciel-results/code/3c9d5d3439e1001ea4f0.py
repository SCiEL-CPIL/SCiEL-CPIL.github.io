"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _soft_min(values, temperature):
    """Compute soft minimum using log-sum-exp trick."""
    # soft_min = -log(sum(exp(-x_i / T))) * T
    # Use max for numerical stability
    v = -values / temperature
    v_max = np.max(v)
    log_sum = v_max + np.log(np.sum(np.exp(v - v_max)))
    return -log_sum * temperature


def _soft_min_grad(values, temperature):
    """Gradient of soft_min with respect to values."""
    v = -values / temperature
    v_max = np.max(v)
    exp_v = np.exp(v - v_max)
    weights = exp_v / np.sum(exp_v)
    return -weights / temperature * (-1)  # d/dx soft_min = weights (since soft_min = -T*log(sum(exp(-x/T))))
    # Actually: soft_min(x) = -T * log(sum(exp(-x_i/T)))
    # d/dx_j = -T * (1/sum) * (-1/T) * exp(-x_j/T) = exp(-x_j/T) / sum(exp(-x_i/T)) = weights_j
    return weights


def _compute_radius(centers):
    """Compute the actual minimum radius (exact, not soft)."""
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_pair:
                min_pair = d
    
    # Boundary distances
    min_boundary = np.inf
    for i in range(n):
        for k in range(3):
            d = 0.5 - abs(centers[i, k])
            if d < min_boundary:
                min_boundary = d
    
    return min(min_pair / 2.0, min_boundary)


def _objective_and_grad(centers, temperature):
    """
    Negative of soft-min radius (we minimize this).
    Returns (value, gradient).
    """
    n = centers.shape[0]
    
    # Compute all pairwise distances
    pairwise_dists = np.zeros((n*(n-1)//2,))
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pairwise_dists[idx] = np.linalg.norm(centers[i] - centers[j])
            idx += 1
    
    # Boundary distances
    boundary_dists = np.zeros((n*3,))
    idx = 0
    for i in range(n):
        for k in range(3):
            boundary_dists[idx] = 0.5 - abs(centers[i, k])
            idx += 1
    
    # Combine all distances (we want min of all these, then radius = min/2 for pairs, min for boundary)
    # Actually radius = min(min_pair/2, min_boundary)
    # Let's work with: all "radius contributions" = [pairwise_dists/2, boundary_dists]
    all_radii = np.concatenate([pairwise_dists / 2.0, boundary_dists])
    
    # Soft min of all_radii
    soft_min_val = _soft_min(all_radii, temperature)
    weights = _soft_min_grad(all_radii, temperature)
    
    # Gradient with respect to centers
    grad = np.zeros_like(centers)
    
    # Pairwise part
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            w = weights[idx]
            diff = centers[i] - centers[j]
            dist = pairwise_dists[idx]
            if dist > 1e-12:
                # d/d centers[i] of (dist/2) = diff/dist * (1/2) * 2 ... 
                # soft_min derivative: d soft_min / d centers[i] = sum_k w_k * d(all_radii_k)/d centers[i]
                # all_radii[idx] = dist/2, d(dist/2)/d centers[i] = diff / (2*dist) * 2 = diff/dist
                # Wait: dist = |c_i - c_j|, d(dist)/d(c_i) = (c_i - c_j)/dist = diff/dist
                # d(dist/2)/d(c_i) = diff/(2*dist)
                # But we have weight w on all_radii[idx] = dist/2
                # So contribution to grad from this pair: w * diff/(2*dist) for c_i, -w * diff/(2*dist) for c_j
                # But we're minimizing -soft_min, so gradient of objective = -gradient of soft_min
                # Let me be careful.
                
                # We want to maximize soft_min(all_radii), so we minimize -soft_min.
                # grad of -soft_min w.r.t. centers = - (grad of soft_min w.r.t. centers)
                
                # soft_min = sum_k w_k * all_radii_k (approximately, for the gradient direction)
                # Actually soft_min is a function, its gradient w.r.t. all_radii is w
                # d soft_min / d c_i = sum_k w_k * d(all_radii_k) / d c_i
                
                # For pairwise term idx: all_radii[idx] = |c_i - c_j| / 2
                # d(all_radii[idx])/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = diff / (2*dist)
                # d(all_radii[idx])/d(c_j) = (c_j - c_i) / (2 * |c_j - c_i|) = -diff / (2*dist)
                
                grad[i] += w * diff / (2.0 * dist)
                grad[j] -= w * diff / (2.0 * dist)
            idx += 1
    
    # Boundary part
    idx = 0
    for i in range(n):
        for k in range(3):
            w = weights[idx]
            # all_radii[idx] = 0.5 - |c_ik|
            # d/d c_ik = -sign(c_ik)
            sign_val = np.sign(centers[i, k])
            if sign_val == 0:
                sign_val = 1.0
            grad[i, k] += w * (-sign_val)
            idx += 1
    
    # We're minimizing -soft_min, so gradient of objective = -grad
    value = -soft_min_val
    return value, -grad


def _fcc_init(n_points, seed=42):
    """
    Generate FCC-like lattice points and select n_points of them.
    FCC in a cube: positions at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell.
    """
    rng = np.random.RandomState(seed)
    
    # Generate FCC lattice points in a larger region
    # FCC unit cell has 4 points. For 100 points, we need about 25 unit cells.
    # Let's generate a grid of FCC points and randomly select.
    
    # Alternative: use a more direct approach
    # Start with a BCC or FCC arrangement
    
    # Let's try: create a 5x5x5 grid of FCC unit cells
    # Each cell has 4 points, so 125 points total, pick 100
    
    points = []
    cell_size = 0.2  # adjust to fit in [-0.5, 0.5]
    
    # FCC basis within a cell of size s:
    # (0,0,0), (0,s/2,s/2), (s/2,0,s/2), (s/2,s/2,0)
    
    n_cells = 5
    s = 1.0 / n_cells
    
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                base = np.array([ix * s, iy * s, iz * s])
                # 4 FCC points
                pts = [
                    base,
                    base + np.array([0, s/2, s/2]),
                    base + np.array([s/2, 0, s/2]),
                    base + np.array([s/2, s/2, 0])
                ]
                for p in pts:
                    # Shift to center in [-0.5, 0.5]
                    p = p - 0.5
                    points.append(p)
    
    points = np.array(points)
    
    # Add small random jitter to break symmetry
    points += rng.uniform(-0.01, 0.01, size=points.shape)
    
    # Clip to valid range
    points = np.clip(points, -0.49, 0.49)
    
    # Randomly select n_points
    idx = rng.choice(len(points), size=n_points, replace=False)
    points = points[idx]
    
    return points


def _jittered_grid_init(n_points, seed=42):
    """Generate a jittered grid initial configuration."""
    rng = np.random.RandomState(seed)
    
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Add jitter
    grid += rng.uniform(-0.02, 0.02, size=grid.shape)
    
    # Clip
    grid = np.clip(grid, -0.49, 0.49)
    
    return grid[:n_points]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations and optimizations
    inits = [
        ("fcc", _fcc_init(N, seed=42)),
        ("fcc2", _fcc_init(N, seed=123)),
        ("jitter", _jittered_grid_init(N, seed=42)),
        ("jitter2", _jittered_grid_init(N, seed=99)),
    ]
    
    for name, init_centers in inits:
        if time.time() > deadline - 30:
            break
        
        # Optimize with decreasing temperature
        centers = init_centers.copy()
        
        # Start with higher temperature for exploration, decrease for exploitation
        temperatures = [0.05, 0.02, 0.01, 0.005, 0.002]
        
        for temp in temperatures:
            if time.time() > deadline - 10:
                break
            
            # Use scipy minimize with L-BFGS-B
            # We minimize -soft_min_radius, which is smooth
            result = minimize(
                _objective_and_grad,
                centers,
                args=(temp,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.499, 0.499)] * (N * 3),
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            centers = result.x
            centers = np.clip(centers, -0.499, 0.499)
            
            # Reshape
            centers = centers.reshape(N, 3)
        
        # Evaluate actual radius
        r = _compute_radius(centers)
        
        if r > best_radius:
            best_radius = r
            best_centers = centers.copy()
    
    # Final polish: use very low temperature for a few more iterations
    if best_centers is not None and time.time() < deadline - 5:
        centers = best_centers.copy()
        
        for temp in [0.001, 0.0005]:
            if time.time() > deadline - 2:
                break
            result = minimize(
                _objective_and_grad,
                centers,
                args=(temp,),
                jac=True,
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            centers = np.clip(result.x.reshape(N, 3), -0.4999, 0.4999)
        
        r = _compute_radius(centers)
        if r > best_radius:
            best_radius = r
            best_centers = centers
    
    if best_centers is None:
        # Fallback to basic grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
