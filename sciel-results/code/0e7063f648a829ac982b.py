"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from nearest-neighbor search. We optimize positions and radii
    simultaneously, using smooth penalty functions for constraint violations.
    """
    best_result = None
    best_score = -np.inf
    
    def grid_init(n_rows, n_cols, n_circles):
        """Generate a grid initialization."""
        positions = []
        radii = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / n_cols
                y = (i + 0.5) / n_rows
                positions.append([x, y])
                radii.append(0.5 / max(n_rows, n_cols) * 0.85)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def random_init(n_circles, seed=42):
        """Generate a random initialization."""
        rng = np.random.RandomState(seed)
        positions = rng.uniform(0.1, 0.9, size=(n_circles, 2))
        radii = np.full(n_circles, 0.04)
        return positions, radii
    
    def hex_init(n_circles):
        """Generate a hexagonal packing initialization."""
        positions = []
        radii = []
        r = 0.07
        dx = 2 * r
        dy = r * np.sqrt(3)
        for row in range(8):
            for col in range(8):
                if len(positions) >= n_circles:
                    break
                x = dx * (col + 0.5 * (row % 2))
                y = dy * row
                x = 0.5 + (x - 3.5 * dx) * 0.12
                y = 0.5 + (y - 3.5 * dy) * 0.12
                if 0.08 < x < 0.92 and 0.08 < y < 0.92:
                    positions.append([x, y])
                    radii.append(r * 0.85)
        while len(positions) < n_circles:
            positions.append([0.5 + np.random.uniform(-0.25, 0.25), 
                              0.5 + np.random.uniform(-0.25, 0.25)])
            radii.append(0.04)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def objective_and_constraints(x, n_circles, contact_threshold=0.15):
        """
        Compute objective (negative sum of radii) and constraint violations.
        
        x: array of shape (3*n,) containing [x1, y1, r1, x2, y2, r2, ...]
        """
        positions = x[0::3]
        radii = x[2::3]
        
        obj = -np.sum(radii)
        
        # Compute all pairwise distances
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Soft contact graph: weight pairs by proximity
        sigma = contact_threshold
        contact_weights = np.exp(-dists**2 / (2 * sigma**2))
        np.fill_diagonal(contact_weights, 0)
        
        # Overlap violations with soft contact weighting
        overlap_violations = np.maximum(0, r_sums - dists)
        weighted_violation = np.sum(contact_weights * overlap_violations**2)
        
        # Boundary violations
        boundary_viol = 0.0
        for i in range(n_circles):
            boundary_viol += max(0, radii[i] - positions[i, 0])**2
            boundary_viol += max(0, radii[i] - (1 - positions[i, 0]))**2
            boundary_viol += max(0, radii[i] - positions[i, 1])**2
            boundary_viol += max(0, radii[i] - (1 - positions[i, 1]))**2
        
        penalty = 100 * weighted_violation + 50 * boundary_viol
        
        return obj + penalty
    
    def solve_from_init(positions, radii, maxiter=500):
        """Solve the optimization problem from a given initialization."""
        n_circles = len(radii)
        
        x0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.001, 0.5))
        
        result = minimize(
            objective_and_constraints,
            x0,
            args=(n_circles,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        return result
    
    def refine_with_local_search(positions, radii, iterations=100):
        """Iteratively refine using local neighbor-based adjustments."""
        positions = positions.copy()
        radii = radii.copy()
        n = len(radii)
        
        for it in range(iterations):
            # Build KD-tree for fast neighbor queries
            tree = cKDTree(positions)
            
            for i in range(n):
                # Find nearby circles
                idx, dist = tree.query(positions[i], k=min(8, n))
                
                # Adjust radius based on local density
                local_pressure = 0
                for j in idx:
                    if j == i:
                        continue
                    gap = dist[j] - radii[i] - radii[j]
                    if gap < 0.05:
                        local_pressure += max(0, 0.05 - gap)
                
                # Try to increase radius if there's room
                if local_pressure < 0.01:
                    radii[i] = min(radii[i] * 1.01, 0.45)
                
                # Reposition if overlapping
                for j in idx:
                    if j == i:
                        continue
                    d = dist[j]
                    r_sum = radii[i] + radii[j]
                    if d < r_sum and d > 1e-10:
                        # Push apart
                        direction = (positions[i] - positions[j]) / d
                        push = (r_sum - d) * 0.5
                        positions[i] += direction * push
                        positions[j] -= direction * push
            
            # Clamp to unit square
            for i in range(n):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], 1 - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], 1 - radii[i])
        
        return positions, radii
    
    def is_valid(pos, rad, tol=1e-5):
        """Check if the configuration is valid."""
        for i in range(len(rad)):
            if pos[i, 0] < rad[i] - tol or pos[i, 0] > 1 - rad[i] + tol:
                return False
            if pos[i, 1] < rad[i] - tol or pos[i, 1] > 1 - rad[i] + tol:
                return False
        
        for i in range(len(rad)):
            for j in range(i + 1, len(rad)):
                dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
                if dist < rad[i] + rad[j] - tol:
                    return False
        return True
    
    # Try multiple initializations
    inits = []
    
    # Grid inits
    inits.append(grid_init(6, 6, N))
    inits.append(grid_init(5, 7, N))
    inits.append(grid_init(7, 5, N))
    
    # Random inits with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        inits.append(random_init(N, seed))
    
    # Hex init
    inits.append(hex_init(N))
    
    # Solve from each initialization
    for positions, radii in inits:
        try:
            # First phase: L-BFGS-B optimization
            result = solve_from_init(positions, radii, maxiter=400)
            
            x_opt = result.x
            pos_opt = x_opt[0::3]
            rad_opt = x_opt[2::3]
            
            # Second phase: local refinement
            pos_ref, rad_ref = refine_with_local_search(pos_opt, rad_opt, iterations=50)
            
            # Check validity and score
            if is_valid(pos_ref, rad_ref):
                score = np.sum(rad_ref)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pos_ref, rad_ref])
            
            # Also check the pre-refinement solution
            if is_valid(pos_opt, rad_opt):
                score = np.sum(rad_opt)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack([pos_opt, rad_opt])
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
