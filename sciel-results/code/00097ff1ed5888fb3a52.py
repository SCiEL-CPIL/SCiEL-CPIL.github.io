"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the score: min(min pairwise dist/2, min wall clearance)."""
    n = len(centers)
    # Wall clearance
    wall_clearance = np.min(0.5 - np.abs(centers))
    # Pairwise distances
    min_dist = np.inf
    for i in range(n):
        diff = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        if len(dists) > 0:
            min_dist = min(min_dist, np.min(dists))
    return min(min_dist / 2.0, wall_clearance)

def _energy(centers, power=6, wall_power=6, alpha=1.0):
    """Repulsive energy to minimize. Lower is better."""
    n = len(centers)
    energy = 0.0
    # Pairwise repulsion
    for i in range(n):
        diff = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        # Avoid division by zero
        dists = np.maximum(dists, 1e-10)
        energy += alpha * np.sum(dists ** (-power))
    # Wall repulsion
    for i in range(n):
        for k in range(2):
            d = 0.5 - abs(centers[i, k])
            d = max(d, 1e-10)
            energy += d ** (-wall_power)
    return energy

def _energy_grad(centers, power=6, wall_power=6, alpha=1.0):
    """Gradient of the energy function."""
    n = len(centers)
    grad = np.zeros_like(centers)
    # Pairwise repulsion gradient
    for i in range(n):
        diff = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        dists = np.maximum(dists, 1e-10)
        # d/dx_i sum_j |x_j - x_i|^{-power} = -power * sum_j (x_i - x_j) / |x_j - x_i|^{power+2}
        inv_dist = dists ** (-(power + 2))
        grad[i] -= alpha * power * np.sum(diff * inv_dist[:, None], axis=0)
        grad[i+1:] += alpha * power * np.sum(diff * inv_dist[:, None], axis=0)
    # Wall repulsion gradient
    for i in range(n):
        for k in range(2):
            d = 0.5 - abs(centers[i, k])
            d = max(d, 1e-10)
            # d/dx_k (0.5 - |x_k|)^{-wall_power}
            sign = 1.0 if centers[i, k] >= 0 else -1.0
            grad[i, k] += wall_power * d ** (-(wall_power + 1)) * sign
    return grad

def _hexagonal_init(n):
    """Initialize with a hexagonal packing pattern, clipped to square."""
    # Approximate radius for hex packing of n circles in unit square
    # Area per circle ~ (2*r)^2 * sqrt(3)/2, total area n * (2r)^2 * sqrt(3)/2 <= 1
    # r ~ 1 / sqrt(2 * n * sqrt(3))
    r_est = 1.0 / np.sqrt(2.0 * n * np.sqrt(3.0))
    # Hex lattice spacing
    a = 2.0 * r_est
    dx = a
    dy = a * np.sqrt(3.0) / 2.0
    
    centers = []
    # Generate hex points
    row = 0
    while len(centers) < n:
        offset = (row % 2) * dx / 2.0
        col = 0
        while len(centers) < n:
            x = offset + col * dx
            y = row * dy
            # Center the grid
            centers.append([x, y])
            col += 1
        row += 1
    centers = np.array(centers[:n])
    
    # Center and scale to fit in [-0.5, 0.5]^2
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= np.array([cx, cy])
    
    # Scale to fit
    max_ext = np.max(np.abs(centers))
    if max_ext > 0:
        centers *= 0.5 / max_ext
    return centers

def _random_init(n, seed=42):
    """Random uniform initialization."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.45, 0.45, size=(n, 2))
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Try multiple initializations
    inits = []
    inits.append(('hex', _hexagonal_init(N)))
    inits.append(('rand1', _random_init(N, seed=42)))
    inits.append(('rand2', _random_init(N, seed=123)))
    inits.append(('rand3', _random_init(N, seed=999)))
    
    for init_name, centers in inits:
        if time.time() > deadline - 20.0:
            break
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _energy,
                centers,
                jac=_energy_grad,
                method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 2),
                options={'maxiter': 2000, 'gtol': 1e-8, 'ftol': 1e-10}
            )
            opt_centers = result.x
        except Exception:
            opt_centers = centers
        
        # Ensure centers are within bounds
        opt_centers = np.clip(opt_centers, -0.5, 0.5)
        
        radius = _compute_radius(opt_centers)
        if radius > best_radius:
            best_radius = radius
            best_centers = opt_centers.copy()
        
        # Polish step: local optimization with tighter tolerance
        if time.time() < deadline - 5.0:
            try:
                result2 = minimize(
                    _energy,
                    opt_centers,
                    jac=_energy_grad,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 2),
                    options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12}
                )
                polish_centers = np.clip(result2.x, -0.5, 0.5)
                polish_radius = _compute_radius(polish_centers)
                if polish_radius > best_radius:
                    best_radius = polish_radius
                    best_centers = polish_centers.copy()
            except Exception:
                pass
    
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers
# EVOLVE-BLOCK-END
