"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Pack 32 circles in the unit square maximizing sum of radii.
    
    Uses a continuous differentiable optimization with a soft contact graph
    derived from differentiable nearest-neighbor search. Constraints are
    only active for pairs predicted to be in contact, reducing effective
    constraint dimensionality.
    """
    best_result = None
    best_score = -np.inf
    
    def hexagonal_init(n_circles):
        """
        Generate a proper hexagonal packing initialization scaled to fit
        within the unit square with margin.
        """
        # For n=32, use a hexagonal arrangement
        # Rows: 6,6,6,6,6,2 or similar
        rows = [6, 6, 6, 6, 6, 2]
        positions = []
        radii = []
        
        # Calculate spacing for hexagonal packing
        # In hexagonal packing, row offset is sqrt(3)/2 * d where d is center-to-center
        d = 0.16  # initial center-to-center distance
        dx = d
        dy = d * np.sqrt(3) / 2
        
        for row_idx, n_in_row in enumerate(rows):
            if len(positions) >= n_circles:
                break
            offset = (row_idx % 2) * dx / 2
            for col in range(n_in_row):
                if len(positions) >= n_circles:
                    break
                x = 0.5 + (col - (n_in_row - 1) / 2.0) * dx + offset
                y = 0.5 + (row_idx - (len(rows) - 1) / 2.0) * dy
                # Clip to ensure we stay in bounds with margin
                x = np.clip(x, 0.1, 0.9)
                y = np.clip(y, 0.1, 0.9)
                positions.append([x, y])
                radii.append(d / 2 * 0.95)
        
        # Fill remaining if needed
        rng = np.random.RandomState(0)
        while len(positions) < n_circles:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            # Check if it's far enough from existing
            valid = True
            for px, py in positions:
                if (x - px)**2 + (y - py)**2 < (d*0.9)**2:
                    valid = False
                    break
            if valid:
                positions.append([x, y])
                radii.append(d / 2 * 0.95)
            else:
                positions.append([x, y])
                radii.append(d / 2 * 0.8)
        
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def grid_init(n_rows, n_cols, n_circles):
        """Generate a grid initialization with appropriate radii."""
        positions = []
        radii = []
        for i in range(n_rows):
            for j in range(n_cols):
                if len(positions) >= n_circles:
                    break
                x = (j + 0.5) / n_cols
                y = (i + 0.5) / n_rows
                positions.append([x, y])
                # Radius is half the minimum spacing
                min_spacing = min(1.0 / n_rows, 1.0 / n_cols)
                radii.append(min_spacing / 2 * 0.95)
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def random_init(n_circles, seed=42):
        """Generate a random initialization with non-overlapping constraint."""
        rng = np.random.RandomState(seed)
        positions = []
        radii = []
        base_r = 0.06
        
        for _ in range(n_circles * 5):
            if len(positions) >= n_circles:
                break
            x = rng.uniform(base_r + 0.02, 1 - base_r - 0.02)
            y = rng.uniform(base_r + 0.02, 1 - base_r - 0.02)
            # Check minimum distance
            min_dist = np.inf
            for px, py in positions:
                dist = np.sqrt((x - px)**2 + (y - py)**2)
                min_dist = min(min_dist, dist)
            if min_dist > 2 * base_r * 1.1:
                positions.append([x, y])
                radii.append(base_r)
        
        # If we couldn't place all, use random placement
        while len(positions) < n_circles:
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            positions.append([x, y])
            radii.append(base_r * 0.7)
        
        return np.array(positions[:n_circles]), np.array(radii[:n_circles])
    
    def soft_contact_weight(dists, sigma=0.08):
        """
        Compute soft contact weights using Gaussian kernel.
        This creates a differentiable approximation of a contact graph
        where pairs with small distances get high weight.
        """
        weights = np.exp(-dists**2 / (2 * sigma**2))
        np.fill_diagonal(weights, 0)
        return weights
    
    def objective(x, n_circles, sigma=0.08, penalty_scale=1000.0):
        """
        Differentiable objective function with soft contact graph.
        
        x: array of shape (3*n,) containing [x1, y1, r1, x2, y2, r2, ...]
        
        Returns: negative sum of radii + penalty for constraint violations
        """
        positions = x[0::3].reshape(-1, 2)
        radii = x[2::3]
        
        # Objective: maximize sum of radii (so minimize negative)
        obj = -np.sum(radii)
        
        # Compute pairwise distances
        diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Sum of radii pairs
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Soft contact weights - differentiable approximation of contact graph
        contact_weights = soft_contact_weight(dists, sigma)
        
        # Overlap violations: r_i + r_j - dist_ij (positive means overlap)
        overlap_viol = np.maximum(0, r_sums - dists)
        
        # Weighted overlap penalty - only active for near-contact pairs
        weighted_overlap = np.sum(contact_weights * overlap_viol**2)
        
        # Boundary violations (smooth)
        # For each circle, check if it fits in [0,1]^2
        # x_i >= r_i, x_i <= 1 - r_i, y_i >= r_i, y_i <= 1 - r_i
        boundary_viol = np.zeros(n_circles)
        for i in range(n_circles):
            # Left boundary
            boundary_viol[i] += np.maximum(0, radii[i] - positions[i, 0])**2
            # Right boundary
            boundary_viol[i] += np.maximum(0, radii[i] - (1 - positions[i, 0]))**2
            # Bottom boundary
            boundary_viol[i] += np.maximum(0, radii[i] - positions[i, 1])**2
            # Top boundary
            boundary_viol[i] += np.maximum(0, radii[i] - (1 - positions[i, 1]))**2
        
        boundary_penalty = np.sum(boundary_viol)
        
        # Total penalty
        penalty = penalty_scale * weighted_overlap + 500.0 * boundary_penalty
        
        return obj + penalty
    
    def solve_from_init(positions, radii, maxiter=500, sigma=0.08):
        """Solve the optimization problem from a given initialization."""
        n_circles = len(radii)
        
        # Flatten: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        # Bounds: positions in [0,1], radii in [0.001, 0.3]
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0))  # x
            bounds.append((0.0, 1.0))  # y
            bounds.append((0.001, 0.3))  # radius
        
        # Use L-BFGS-B
        result = minimize(
            objective,
            x0,
            args=(n_circles, sigma),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
        )
        
        return result
    
    def verify_solution(pos, rad):
        """Verify that the solution satisfies all constraints."""
        n = len(rad)
        for i in range(n):
            if pos[i, 0] < rad[i] - 1e-5 or pos[i, 0] > 1 - rad[i] + 1e-5:
                return False
            if pos[i, 1] < rad[i] - 1e-5 or pos[i, 1] > 1 - rad[i] + 1e-5:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                if dist < rad[i] + rad[j] - 1e-5:
                    return False
        return True
    
    def fix_solution(pos, rad):
        """Fix minor constraint violations by shrinking radii or adjusting positions."""
        pos = pos.copy()
        rad = rad.copy()
        n = len(rad)
        
        # First, ensure boundary constraints
        for i in range(n):
            rad[i] = min(rad[i], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1])
            rad[i] = max(rad[i], 0.001)
        
        # Then, fix overlaps by shrinking radii
        for _ in range(100):
            fixed = True
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                    required = rad[i] + rad[j]
                    if dist < required - 1e-8:
                        # Shrink both radii
                        excess = required - dist
                        shrink_i = excess * rad[i] / required
                        shrink_j = excess * rad[j] / required
                        rad[i] = max(0.001, rad[i] - shrink_i)
                        rad[j] = max(0.001, rad[j] - shrink_j)
                        fixed = False
            if fixed:
                break
        
        return pos, rad
    
    # Generate diverse initializations
    inits = []
    
    # Hexagonal packing - best initial structure
    inits.append(hexagonal_init(N))
    
    # Grid initializations
    inits.append(grid_init(6, 6, N))
    inits.append(grid_init(5, 7, N))
    inits.append(grid_init(7, 5, N))
    inits.append(grid_init(4, 8, N))
    inits.append(grid_init(8, 4, N))
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718]:
        inits.append(random_init(N, seed))
    
    # Solve from each initialization
    for idx, (positions, radii) in enumerate(inits):
        try:
            # Try with different sigma values for contact detection
            for sigma in [0.06, 0.08, 0.10]:
                result = solve_from_init(positions, radii, maxiter=400, sigma=sigma)
                
                if result is not None:
                    x_opt = result.x
                    pos_opt = x_opt[0::3].reshape(-1, 2)
                    rad_opt = x_opt[2::3]
                    
                    # Verify and fix if needed
                    if verify_solution(pos_opt, rad_opt):
                        score = np.sum(rad_opt)
                    else:
                        pos_opt, rad_opt = fix_solution(pos_opt, rad_opt)
                        if verify_solution(pos_opt, rad_opt):
                            score = np.sum(rad_opt)
                        else:
                            continue
                    
                    if score > best_score:
                        best_score = score
                        best_result = np.column_stack([pos_opt, rad_opt])
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side * 0.95
                cells.append([x, y, r])
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
