"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def _compute_max_radius(positions, w, h):
    """Compute the maximum uniform radius for given positions in rectangle w x h."""
    # Minimum pairwise distance
    dists = cdist(positions, positions)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Minimum distance to boundary
    boundary_dists = np.min([
        positions[:, 0],
        w - positions[:, 0],
        positions[:, 1],
        h - positions[:, 1]
    ], axis=1)
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair, min_boundary) / 2 - 1e-7


def _local_optimize(positions, w, h, max_iters=200, step_size=0.01):
    """Local optimization to maximize the minimum radius by adjusting positions."""
    positions = positions.copy()
    
    for iteration in range(max_iters):
        r = _compute_max_radius(positions, w, h)
        if r <= 0:
            break
        
        # Compute gradient-like direction: push circles apart
        # and away from boundaries
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = positions[i] - positions[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random small perturbation
                    diff = np.random.randn(2) * 1e-6
                    dist = np.linalg.norm(diff)
                
                # Force is stronger when closer than 2*r
                if dist < 2 * r:
                    force_mag = (2 * r - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
        
        # Boundary repulsion
        for i in range(N):
            # Left boundary
            if positions[i, 0] < r:
                forces[i, 0] += (r - positions[i, 0])
            # Right boundary
            if positions[i, 0] > w - r:
                forces[i, 0] -= (positions[i, 0] - (w - r))
            # Bottom boundary
            if positions[i, 1] < r:
                forces[i, 1] += (r - positions[i, 1])
            # Top boundary
            if positions[i, 1] > h - r:
                forces[i, 1] -= (positions[i, 1] - (h - r))
        
        # Apply forces with step size
        positions += step_size * forces
        
        # Clamp to valid region
        positions[:, 0] = np.clip(positions[:, 0], 0, w)
        positions[:, 1] = np.clip(positions[:, 1], 0, h)
    
    return positions


def _grid_to_positions(nx, ny, w, h):
    """Create grid positions for nx x ny grid in rectangle w x h."""
    xs = np.linspace(w / (2 * nx), w - w / (2 * nx), nx)
    ys = np.linspace(h / (2 * ny), h - h / (2 * ny), ny)
    
    positions = []
    for i in range(nx):
        for j in range(ny):
            positions.append([xs[i], ys[j]])
    
    return np.array(positions)


def _hungarian_match_circles_to_cells(N_circles, grid_x, grid_y, w, h):
    """
    Use Hungarian algorithm to match circles to grid cells.
    Returns matched positions.
    """
    nx = len(grid_x)
    ny = len(grid_y)
    total_cells = nx * ny
    
    if total_cells < N_circles:
        return None
    
    # Create cost matrix: cost of assigning circle i to cell j
    # We want to maximize minimum distance, so minimize negative of min distance
    # Simplified: assign circles to cells to maximize spread
    
    # For simplicity, use a greedy approach with Hungarian on a simplified cost
    # Cost = negative of distance to nearest other assigned cell (approximation)
    
    # Create cell positions
    cell_positions = []
    for i in range(nx):
        for j in range(ny):
            cell_positions.append([grid_x[i], grid_y[j]])
    cell_positions = np.array(cell_positions)
    
    # Use Hungarian algorithm to assign circles to cells
    # Cost: we want circles to be well-spread, so use a heuristic cost
    # Simple approach: assign first N cells in a space-filling order
    
    # For now, use a simple assignment: sort cells by distance from center
    center = np.array([w / 2, h / 2])
    cell_dists = np.linalg.norm(cell_positions - center, axis=1)
    
    # Assign circles to cells starting from outer cells (more boundary space)
    # Actually, let's just use the first N cells in row-major order
    # This is a simplification; a full Hungarian implementation would be more complex
    
    # For now, return the first N cell positions
    return cell_positions[:N_circles]


def construct():
    best_result = None
    best_sum = -1
    
    # Try different grid configurations and aspect ratios
    configs = [
        (5, 5),
        (4, 6),
        (6, 4),
        (3, 7),
        (7, 3),
        (2, 11),
        (11, 2),
        (4, 5),
        (5, 4),
    ]
    
    for nx, ny in configs:
        if nx * ny < N:
            continue
        
        # Try different aspect ratios
        for w_frac in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
            w = 2 * w_frac
            h = 2 - w
            
            # Create grid positions
            positions = _grid_to_positions(nx, ny, w, h)
            
            if len(positions) < N:
                continue
            
            positions = positions[:N]
            
            # Local optimization
            optimized_positions = _local_optimize(positions, w, h, max_iters=100, step_size=0.005)
            
            # Compute radius
            r = _compute_max_radius(optimized_positions, w, h)
            
            if r <= 0:
                continue
            
            total_sum = N * r
            if total_sum > best_sum:
                best_sum = total_sum
                result = np.zeros((N, 3))
                result[:, 0] = optimized_positions[:, 0]
                result[:, 1] = optimized_positions[:, 1]
                result[:, 2] = r
                best_result = result
    
    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
