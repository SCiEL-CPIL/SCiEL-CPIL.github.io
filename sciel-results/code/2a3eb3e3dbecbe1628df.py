"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_radius(centres):
    """Compute the largest common radius for the given centres."""
    # Distance to boundary
    boundary = 0.5 - np.abs(centres)
    min_boundary = boundary.min()
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = dists.min() / 2.0
    
    return min(min_boundary, min_pair)

def _fcc_lattice_init(n):
    """Generate initial positions based on FCC-like lattice structure."""
    # Start with a grid and then perturb
    m = int(np.ceil(n ** (1.0/3.0)))
    # Create a slightly denser grid than needed
    ticks = (np.arange(m) + 0.5) / m - 0.5
    
    # Generate all grid points
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(grid) < n:
        # If not enough, just use what we have and fill rest randomly
        centres = grid.copy()
        if len(centres) < n:
            remaining = n - len(centres)
            random_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 3))
            centres = np.vstack([centres, random_pts])
    else:
        # Select n points from grid, prioritizing well-spread ones
        # Use a simple selection: take first n
        centres = grid[:n].copy()
    
    return centres

def _energy(centres, n):
    """Energy function to minimize: negative of the radius (or a penalty for overlap)."""
    # We want to maximize the minimum distance
    # Use a soft-min approach: -min(dists) is non-smooth, so use a penalty
    
    # Distance to boundary
    boundary = 0.5 - np.abs(centres)
    
    # Pairwise distances
    diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # We want to maximize the minimum of (min_boundary, min_pair/2)
    # Equivalent to minimizing the negative
    # Use a smooth approximation: -soft_min
    
    # Soft minimum using log-sum-exp trick (for negative values)
    # Actually, let's just use the raw minimum for simplicity in the objective,
    # but we need gradients. Let's use a different approach.
    
    # Objective: maximize r = min(min_boundary, min_pair/2)
    # This is equivalent to minimizing -r
    # But -min is non-smooth. Let's use a penalty approach.
    
    # For optimization, let's try to maximize the sum of squared distances
    # to encourage spreading, while keeping them in bounds.
    
    # Actually, let's use a direct approach: minimize the maximum violation
    # Or better: use scipy's minimize with a proper objective
    
    # Let's define the objective as the negative of the minimum clearance
    # But that's non-smooth. Let's use a smooth approximation.
    
    # Smooth minimum approximation: -log(sum(exp(-a*x))) / a for large a
    # where x are the clearances
    
    clearances = np.concatenate([boundary.flatten(), (dists[dists < np.inf]) / 2.0])
    
    # Filter out very large distances to avoid numerical issues
    clearances = clearances[clearances < 1.0]
    
    if len(clearances) == 0:
        return 0.0
    
    # Use soft min
    a = 50.0  # Sharpness parameter
    soft_min = -np.log(np.sum(np.exp(-a * clearances))) / a
    
    return soft_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin for 150s limit
    
    # Initial guess: FCC-like lattice
    centres = _fcc_lattice_init(N)
    best_radius = _compute_radius(centres)
    best_centres = centres.copy()
    
    # Multiple restarts with different initializations
    np.random.seed(42)
    
    n_restarts = 0
    max_restarts = 20
    
    while time.time() < deadline and n_restarts < max_restarts:
        # Try different initializations
        if n_restarts == 0:
            # Use the lattice init
            init_centres = _fcc_lattice_init(N)
        else:
            # Random initialization with some structure
            # Use a perturbed grid
            m = int(np.ceil(N ** (1.0/3.0)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
            
            if len(grid) >= N:
                # Shuffle and take N
                idx = np.random.choice(len(grid), N, replace=False)
                init_centres = grid[idx].copy()
                # Add small random perturbation
                init_centres += np.random.normal(0, 0.01, size=(N, 3))
            else:
                init_centres = np.random.uniform(-0.45, 0.45, size=(N, 3))
            
            # Clip to stay in bounds
            init_centres = np.clip(init_centres, -0.49, 0.49)
        
        # Optimize
        try:
            def objective(x):
                centres_x = x.reshape(N, 3)
                return _energy(centres_x, N)
            
            def jac(x):
                centres_x = x.reshape(N, 3)
                # Numerical gradient for simplicity (could be slow)
                # For speed, let's use finite differences with small step
                eps = 1e-8
                grad = np.zeros_like(x)
                for i in range(len(x)):
                    x_plus = x.copy()
                    x_plus[i] += eps
                    x_minus = x.copy()
                    x_minus[i] -= eps
                    grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
                return grad
            
            # Use L-BFGS-B for bounded optimization
            bounds = [(-0.5, 0.5)] * (N * 3)
            
            result = minimize(
                objective,
                init_centres.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'gtol': 1e-6}
            )
            
            optimised_centres = result.x.reshape(N, 3)
            
            # Clip to ensure validity
            optimised_centres = np.clip(optimised_centres, -0.5, 0.5)
            
            radius = _compute_radius(optimised_centres)
            
            if radius > best_radius:
                best_radius = radius
                best_centres = optimised_centres.copy()
                
        except Exception:
            pass
        
        n_restarts += 1
        
        # Check time
        if time.time() > deadline - 20:
            break
    
    return best_centres

# EVOLVE-BLOCK-END
