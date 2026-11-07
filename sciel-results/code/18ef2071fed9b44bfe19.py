"""
Strategy: Deterministic Radial Density Matching (DRDM).
Instead of stochastic basin hopping, we define a target radial density profile
based on known global minimum structures for Sutton-Chen clusters (e.g., icosahedral
for n=13, 55, 100; decahedral for n=38, 59, 75, 79, 80). We then use a
gradient-free optimizer (Powell or Nelder-Mead via SciPy) to minimize the
discrepancy between the cluster's actual radial density and this target profile,
while simultaneously relaxing the structure with L-BFGS-B. This hybrid approach
guides the search toward the correct structural motif by explicitly penalizing
deviations from the expected radial distribution of atoms.
"""

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE_SEC = 290.0  # Safety margin before 300s hard kill
_START_TIME = None

def _time_left():
    if _START_TIME is None:
        return _DEADLINE_SEC
    return _DEADLINE_SEC - (time.time() - _START_TIME)

def _generate_target_density(n, kind):
    """
    Generate a target radial density profile for the given cluster size.
    For Sutton-Chen potentials, the global minima are often icosahedral or decahedral.
    We approximate the radial density as a sum of Gaussians at the expected shell radii.
    """
    # Typical shell radii for icosahedral/decahedral clusters (in reduced units, a=1)
    # These are approximate and depend on the specific packing.
    # For n=13: 1 + 12 (icosahedron)
    # For n=55: 1 + 12 + 42 (Mackay icosahedron)
    # For n=38: 1 + 12 + 25 (decahedral-like)
    # For n=59: 1 + 12 + 42 + 4 (partial next shell)
    # For n=75: 1 + 12 + 42 + 20 (partial)
    # For n=79: 1 + 12 + 42 + 24
    # For n=80: 1 + 12 + 42 + 25
    # For n=100: 1 + 12 + 42 + 45 (Mackay)
    
    # We use a generic approach: assume the cluster is roughly spherical with
    # a density profile that peaks at certain radii.
    # For simplicity, we define target shells based on common cluster geometries.
    
    if n <= 13:
        shells = [(1, 0.0), (12, 1.0)]  # 1 center, 12 at r=1
    elif n <= 38:
        shells = [(1, 0.0), (12, 1.0), (n-13, 1.8)]  # Approximate decahedral
    elif n <= 55:
        shells = [(1, 0.0), (12, 1.0), (n-13, 1.8)]  # Partial icosahedral
    elif n <= 80:
        shells = [(1, 0.0), (12, 1.0), (42, 1.8), (n-55, 2.6)]  # Partial Mackay
    else:
        shells = [(1, 0.0), (12, 1.0), (42, 1.8), (n-55, 2.6)]  # Approximate
    
    # Build target density profile
    r_max = 4.0
    dr = 0.05
    r = np.arange(0, r_max, dr)
    target_density = np.zeros_like(r)
    for count, radius in shells:
        # Gaussian shell
        sigma = 0.15
        target_density += count * np.exp(-0.5 * ((r - radius) / sigma) ** 2)
    
    return r, target_density

def _compute_radial_density(positions, r_grid):
    """
    Compute the radial density of the cluster on the given r_grid.
    """
    n = len(positions)
    # Center of mass
    com = np.mean(positions, axis=0)
    positions_centered = positions - com
    r = np.linalg.norm(positions_centered, axis=1)
    
    # Histogram
    density, _ = np.histogram(r, bins=r_grid)
    # Normalize by bin width
    dr = r_grid[1] - r_grid[0]
    density /= dr
    return density

def _density_mismatch(positions, r_grid, target_density):
    """
    Compute the mismatch between the actual radial density and the target density.
    """
    actual_density = _compute_radial_density(positions, r_grid)
    # Normalize both densities
    if np.sum(actual_density) > 0:
        actual_density = actual_density / np.sum(actual_density)
    if np.sum(target_density) > 0:
        target_density = target_density / np.sum(target_density)
    
    # L2 norm of the difference
    return np.sum((actual_density - target_density) ** 2)

def _hybrid_objective(params, n, kind, r_grid, target_density):
    """
    Hybrid objective: Sutton-Chen energy + density mismatch penalty.
    """
    positions = params.reshape(n, 3)
    
    # Check for invalid configurations
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    if min_d < 0.35:
        return 1e10  # Penalize invalid configurations
    
    # Compute Sutton-Chen energy
    try:
        energy, _ = sutton_chen_energy_gradient(positions, kind)
        if not np.isfinite(energy):
            return 1e10
    except Exception:
        return 1e10
    
    # Compute density mismatch
    mismatch = _density_mismatch(positions, r_grid, target_density)
    
    # Hybrid objective: energy + lambda * mismatch
    # We use a large lambda to strongly guide the search toward the target density
    lambda_weight = 100.0
    return energy + lambda_weight * mismatch

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Generate target density profile
    r_grid, target_density = _generate_target_density(n, kind)
    
    # Initial configuration: random compact cluster
    rng = np.random.default_rng(42)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.85
    positions = rng.uniform(-radius, radius, (n, 3))
    
    # Relax the initial configuration
    try:
        energy, positions = relax(positions, potential, kind)
    except Exception:
        pass
    
    # Hybrid optimization: minimize energy + density mismatch
    # Use Powell method for gradient-free optimization
    best_energy = energy
    best_positions = positions.copy()
    
    # Run hybrid optimization for a few iterations
    max_iter = 50
    for i in range(max_iter):
        if _time_left() < 10:
            break
        
        # Optimize the hybrid objective
        try:
            result = minimize(
                _hybrid_objective,
                positions.flatten(),
                args=(n, kind, r_grid, target_density),
                method='Powell',
                options={'maxiter': 10, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            if result.success or result.fun < best_energy:
                new_positions = result.x.reshape(n, 3)
                # Check validity
                min_d = float('inf')
                for j in range(n):
                    for k in range(j+1, n):
                        d = np.linalg.norm(new_positions[j] - new_positions[k])
                        if d < min_d:
                            min_d = d
                if min_d >= 0.35:
                    # Relax the new configuration
                    try:
                        new_energy, new_positions = relax(new_positions, potential, kind)
                        if np.isfinite(new_energy) and new_energy < best_energy:
                            best_energy = new_energy
                            best_positions = new_positions.copy()
                    except Exception:
                        pass
            positions = result.x.reshape(n, 3)
        except Exception:
            break
    
    # Final relaxation
    try:
        final_energy, final_positions = relax(best_positions, potential, kind)
        if np.isfinite(final_energy):
            min_d = float('inf')
            for j in range(n):
                for k in range(j+1, n):
                    d = np.linalg.norm(final_positions[j] - final_positions[k])
                    if d < min_d:
                        min_d = d
            if min_d >= 0.35:
                best_energy = final_energy
                best_positions = final_positions
    except Exception:
        pass
    
    return best_positions
# EVOLVE-BLOCK-END
