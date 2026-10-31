"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

N = 16
D = 2

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute (max_pairwise / min_pairwise)^2."""
    n = len(points)
    # Compute all pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Exclude self-distances (diagonal)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d <= 0:
        return np.inf
    return (max_d / min_d) ** 2


def _hexagonal_cluster_16():
    """
    Generate a hexagonal lattice cluster of 16 points.
    
    Hexagonal numbers: 1, 7, 19... 
    For 16 points, we use a subset of the hexagonal lattice.
    
    A good arrangement: center + 6 (first ring) + 9 from second ring = 16
    But let's think more carefully.
    
    Actually, let's use a more systematic approach. The hexagonal lattice
    has points at positions (i + j/2, j*sqrt(3)/2) for integers i, j.
    
    For 16 points, a natural cluster would be:
    - Center: (0, 0)
    - Ring 1 (6 points): at distance 1
    - Ring 2 partial (9 points): at distance 2
    
    Total: 1 + 6 + 9 = 16
    
    Let me define the hexagonal lattice coordinates properly.
    """
    # Hexagonal lattice basis vectors
    # Point (i, j) in axial coordinates maps to:
    # x = i + j/2
    # y = j * sqrt(3) / 2
    
    sqrt3 = np.sqrt(3.0)
    
    # Generate candidate points from hexagonal lattice
    # Center + ring 1 + ring 2 (partial)
    
    points = []
    
    # Center
    points.append((0.0, 0.0))
    
    # Ring 1: 6 points at distance 1
    for k in range(6):
        angle = np.pi / 3 * k
        points.append((np.cos(angle), np.sin(angle)))
    
    # Ring 2: We need 9 more points
    # In hexagonal lattice, ring 2 has 12 points at distance 2
    # We'll take 9 of them, choosing those that maintain good symmetry
    
    # The 12 points of ring 2 in hexagonal lattice:
    # They are at positions corresponding to (i,j) where max(|i|,|j|,|i+j|) = 2
    # Let's generate all ring 2 points and pick 9
    
    ring2_candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            # Hexagonal distance
            d = max(abs(i), abs(j), abs(i + j))
            if d == 2:
                x = i + j / 2.0
                y = j * sqrt3 / 2.0
                ring2_candidates.append((x, y))
    
    # We have 12 candidates. Pick 9 that give good coverage.
    # For symmetry, let's pick every other one or use a pattern.
    # Actually, let's just take the first 9, or better, select for symmetry.
    
    # A symmetric selection: take all 12 but we need only 9.
    # Let's pick those with specific angles to maintain 3-fold or 6-fold symmetry.
    
    # Alternative: use a different cluster shape.
    # Let's use a 4-4-4-4 rectangular-ish arrangement but hexagonal.
    
    # Actually, let me reconsider. A better approach for 16 points:
    # Use a 4x4 grid but with hexagonal distortion.
    
    # Let me generate a proper hexagonal cluster.
    # For 16 points, one good arrangement is based on the hexagonal lattice
    # taking the 16 points closest to the origin.
    
    # Generate many hexagonal lattice points and sort by distance
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i + j / 2.0
            y = j * sqrt3 / 2.0
            d = np.sqrt(x**2 + y**2)
            lattice_points.append((d, x, y))
    
    lattice_points.sort()
    selected = [(x, y) for _, x, y in lattice_points[:16]]
    
    return np.array(selected, dtype=float)


def _optimize_hexagonal(points_0, time_limit=140.0):
    """
    Optimize the hexagonal cluster by adjusting parameters.
    We'll use a simple iterative refinement approach.
    """
    start_time = time.time()
    points = points_0.copy()
    n = len(points)
    
    # First, center the points
    points -= np.mean(points, axis=0)
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    # Use scipy if available for optimization
    try:
        from scipy.optimize import minimize, differential_evolution
        from scipy.spatial.distance import pdist, squareform
    except ImportError:
        # Fallback to simple gradient-free method
        return best_points, best_score
    
    def objective_flat(flat_params):
        """Objective function for optimization."""
        p = flat_params.reshape(n, 2)
        # Center
        p -= np.mean(p, axis=0)
        d = pdist(p)
        if np.min(d) <= 1e-10:
            return 1e10
        return (np.max(d) / np.min(d)) ** 2
    
    # Try differential evolution first for global search
    x0 = points.ravel()
    
    # Set bounds
    bounds = [(-5, 5)] * (n * 2)
    
    try:
        result = differential_evolution(
            objective_flat, bounds, seed=42, maxiter=50, tol=1e-6,
            polish=True, workers=1, init='sobol'
        )
        if result.success or result.fun < best_score:
            best_points = result.x.reshape(n, 2)
            best_points -= np.mean(best_points, axis=0)
            best_score = _compute_score(best_points)
    except Exception:
        pass
    
    # Refine with L-BFGS-B
    try:
        def grad_objective(flat_params):
            p = flat_params.reshape(n, 2)
            p -= np.mean(p, axis=0)
            d = pdist(p)
            min_d = np.min(d)
            max_d = np.max(d)
            score = (max_d / min_d) ** 2
            return score
        
        result2 = minimize(
            grad_objective, best_points.ravel(), method='L-BFGS-B',
            bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result2.fun < best_score:
            best_points = result2.x.reshape(n, 2)
            best_points -= np.mean(best_points, axis=0)
            best_score = _compute_score(best_points)
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > time_limit:
        return best_points, best_score
    
    # Additional refinement: try to improve by moving individual points
    for _ in range(100):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n):
            # Try small perturbations
            for dx in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                for dy in [-0.1, -0.05, 0.0, 0.05, 0.1]:
                    new_points = best_points.copy()
                    new_points[i, 0] += dx
                    new_points[i, 1] += dy
                    new_points -= np.mean(new_points, axis=0)
                    s = _compute_score(new_points)
                    if s < best_score - 1e-10:
                        best_points = new_points
                        best_score = s
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_points, best_score


def construct():
    start_time = time.time()
    
    # Generate initial hexagonal cluster
    points = _hexagonal_cluster_16()
    
    # Optimize
    best_points, best_score = _optimize_hexagonal(points, time_limit=140.0)
    
    # Ensure we return within time limit
    if time.time() - start_time > 145.0:
        # Return whatever we have
        pass
    
    # Final check: ensure all points are distinct
    d = np.sqrt(np.sum((best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 1e-10:
        # Fallback to grid
        side = 4
        grid = list(itertools.product(range(side), repeat=2))[:N]
        return np.array(grid, dtype=float)
    
    return best_points

# EVOLVE-BLOCK-END
