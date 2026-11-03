"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(centers):
    """Compute the maximum common radius for given centers."""
    n = len(centers)
    # Wall constraint: min distance to cube boundary
    wall_r = np.min(0.5 - np.abs(centers))
    # Pairwise constraint: min pairwise distance / 2
    if n > 1:
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        # Set diagonal to infinity
        np.fill_diagonal(dists, np.inf)
        pair_r = np.min(dists) / 2.0
    else:
        pair_r = np.inf
    return min(wall_r, pair_r)


def _score(centers):
    """Score function: the maximum radius."""
    return _compute_radius(centers)


def _make_seed(m=5):
    """Create a simple cubic grid seed."""
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N]


def _repulsion_optimize(centers, max_iter=500, step_size=0.05):
    """Local optimization using repulsion forces."""
    n = len(centers)
    centers = centers.copy()
    
    for iteration in range(max_iter):
        # Compute forces from pairwise repulsion and wall attraction
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    forces[i] += np.random.randn(3) * 0.001
                    forces[j] -= np.random.randn(3) * 0.001
                else:
                    # Repulsive force, stronger when closer
                    force_mag = 1.0 / (dist ** 2 + 1e-6)
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
        
        # Wall constraint: push back if too close to boundary
        for i in range(3):
            for idx in range(n):
                # Left wall
                if centers[idx, i] < -0.5 + 0.05:
                    forces[idx, i] += (0.5 - 0.05 - (-0.5)) ** 2 * 10
                # Right wall
                if centers[idx, i] > 0.5 - 0.05:
                    forces[idx, i] -= (0.5 - 0.05 - 0.5) ** 2 * 10
        
        # Update positions
        centers += step_size * forces
        # Clip to cube
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Decrease step size
        step_size *= 0.995
    
    return centers


def _slsqp_optimize(centers, time_budget=60):
    """Optimize using SLSQP with objective to maximize minimum radius."""
    start_time = time.time()
    
    def objective(x):
        """Objective: minimize negative minimum radius."""
        c = x.reshape(-1, 3)
        r = _compute_radius(c)
        return -r
    
    # Constraints: keep centers inside cube
    constraints = []
    for i in range(N):
        for j in range(3):
            # x >= -0.5
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: x[i * 3 + j] + 0.5})
            # x <= 0.5
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: 0.5 - x[i * 3 + j]})
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    x0 = centers.flatten()
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 100, 'ftol': 1e-8})
        if result.success:
            return result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return centers


def construct():
    """Main construct function with time budget management."""
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Start with grid seed
    best_centers = _make_seed(5)
    best_radius = _score(best_centers)
    
    # Try multiple random seeds
    np.random.seed(42)
    
    while time.time() < deadline - 30:  # Leave 30s margin
        # Generate a random initial configuration
        centers = np.random.uniform(-0.4, 0.4, (N, 3))
        
        # Optimize with repulsion
        optimized = _repulsion_optimize(centers, max_iter=200, step_size=0.03)
        
        # Evaluate
        radius = _score(optimized)
        if radius > best_radius:
            best_radius = radius
            best_centers = optimized.copy()
        
        # If time allows, try SLSQP on the best
        if time.time() < deadline - 60:
            slsqp_result = _slsqp_optimize(best_centers, time_budget=20)
            slsqp_radius = _score(slsqp_result)
            if slsqp_radius > best_radius:
                best_radius = slsqp_radius
                best_centers = slsqp_result.copy()
    
    # Final check
    if time.time() < deadline - 5:
        # One more repulsion pass on best
        final = _repulsion_optimize(best_centers, max_iter=100, step_size=0.02)
        final_radius = _score(final)
        if final_radius > best_radius:
            best_centers = final
            best_radius = final_radius
    
    return best_centers
# EVOLVE-BLOCK-END
