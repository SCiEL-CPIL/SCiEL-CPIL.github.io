"""
Hybrid approach: Discrete greedy lattice search + continuous L-BFGS-B polish.

Strategy:
1. Generate a fine hexagonal lattice of candidate sites.
2. Use a greedy best-fit algorithm with multiple random offsets to select 200 sites
   maximizing the minimum distance to neighbors and boundaries.
3. Take the best discrete configuration found and refine it using scipy's L-BFGS-B
   to maximize the minimum distance (radius).
4. Deadline guard to return the best result before the 150s limit.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 200

# EVOLVE-BLOCK-START
def construct():
    """
    Pack 200 equal circles in the unit square [-0.5, 0.5]^2 to maximize the common radius.
    
    Strategy:
    1. Discrete greedy search on a fine hexagonal grid with multiple random offsets.
    2. Continuous optimization (L-BFGS-B) to polish the best discrete configuration.
    3. Deadline guard to return best result before 150s.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin: 10s before 150s limit

    def compute_radius(positions):
        """Compute the maximum possible radius given positions."""
        positions = np.asarray(positions)
        if positions.shape != (N, 2) or not np.all(np.isfinite(positions)):
            return 0.0
        
        # Check boundary constraints
        min_coord = -0.5 - positions.min(axis=0)
        max_coord = 0.5 - positions.max(axis=0)
        boundary_margin = min(min_coord.min(), max_coord.min())
        
        if boundary_margin < 0:
            return 0.0
        
        # Check pairwise distances
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = dist_matrix.min()
        
        # The radius is limited by both boundary and pairwise constraints
        return min(boundary_margin, min_pair_dist / 2.0)

    def negative_radius(positions):
        """Objective function to minimize (negative of radius)."""
        r = compute_radius(positions)
        return -r

    def hexagonal_lattice(spacing, offset_x=0.0, offset_y=0.0):
        """
        Generate a hexagonal lattice of points within the square.
        spacing: distance between adjacent points in the lattice.
        offset_x, offset_y: shift the entire lattice.
        """
        points = []
        # Hexagonal lattice:
        # Row i: y = i * spacing * sqrt(3)/2
        # Col j: x = j * spacing + (i % 2) * spacing / 2
        
        # Determine range to cover [-0.5, 0.5]^2
        # y range: -0.5 to 0.5
        # x range: -0.5 to 0.5
        
        # Number of rows
        y_spacing = spacing * np.sqrt(3) / 2
        n_rows = int(np.ceil(1.0 / y_spacing)) + 2
        n_cols = int(np.ceil(1.0 / spacing)) + 2
        
        for i in range(-n_rows, n_rows + 1):
            for j in range(-n_cols, n_cols + 1):
                x = j * spacing + (i % 2) * spacing / 2 + offset_x
                y = i * y_spacing + offset_y
                # Only keep points inside the square (with small margin for safety)
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append((x, y))
        
        if not points:
            return np.array([])
        
        return np.array(points)

    def greedy_select(lattice_points, n_select, rng):
        """
        Greedy best-fit selection of n_select points from lattice_points.
        Maximizes the minimum distance to already selected points and boundaries.
        """
        if len(lattice_points) < n_select:
            # Not enough points, pad with random
            result = lattice_points.copy()
            needed = n_select - len(result)
            extra = rng.uniform(-0.45, 0.45, size=(needed, 2))
            result = np.vstack([result, extra])
            return result[:n_select]
        
        selected = []
        remaining = list(range(len(lattice_points)))
        
        # Precompute boundary distances for all lattice points
        xs = lattice_points[:, 0]
        ys = lattice_points[:, 1]
        dist_x = np.minimum(0.5 - xs, xs + 0.5)
        dist_y = np.minimum(0.5 - ys, ys + 0.5)
        boundary_dists = np.minimum(dist_x, dist_y)
        
        # For efficiency, we'll use a simple approach:
        # At each step, find the point that maximizes the minimum of:
        # - distance to boundary
        # - distance to all already selected points
        
        for step in range(n_select):
            if not remaining:
                break
            
            best_idx = -1
            best_score = -np.inf
            
            # Vectorized computation for speed
            # For each candidate, compute min distance to selected points
            if selected:
                selected_arr = np.array(selected)
                # Distance from each remaining point to each selected point
                # remaining: (n_rem, 2), selected_arr: (n_sel, 2)
                # Use cdist for efficiency
                dists_to_selected = cdist(lattice_points[remaining], selected_arr)
                min_dist_to_selected = dists_to_selected.min(axis=1)
            else:
                min_dist_to_selected = np.full(len(remaining), np.inf)
            
            # Score = min(boundary_dist, min_dist_to_selected)
            candidate_boundary = boundary_dists[remaining]
            scores = np.minimum(candidate_boundary, min_dist_to_selected)
            
            best_local_idx = np.argmax(scores)
            best_score = scores[best_local_idx]
            best_idx = remaining[best_local_idx]
            
            selected.append(best_idx)
            remaining.pop(best_local_idx)
            
            # Early termination if we can't do better
            if best_score <= 0:
                break
        
        return lattice_points[selected]

    def square_grid_initialization(n_circles):
        """Create initial positions using square grid."""
        m = int(np.ceil(np.sqrt(n_circles)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:n_circles]

    best_positions = None
    best_radius = -np.inf
    
    # Phase 1: Discrete greedy search
    # Try multiple lattice spacings and offsets
    spacings = [0.07, 0.072, 0.074, 0.076, 0.078, 0.08]
    n_offsets = 10
    
    for spacing in spacings:
        if time.time() > deadline - 30:  # Leave time for optimization
            break
        
        for offset_idx in range(n_offsets):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.RandomState(hash((spacing, offset_idx)) % (2**32))
            offset_x = rng.uniform(-spacing/4, spacing/4)
            offset_y = rng.uniform(-spacing/4, spacing/4)
            
            lattice_points = hexagonal_lattice(spacing, offset_x, offset_y)
            
            if len(lattice_points) < N:
                continue
            
            positions = greedy_select(lattice_points, N, rng)
            
            if positions.shape != (N, 2):
                continue
            
            # Validate
            if not np.all(np.isfinite(positions)):
                continue
            
            radius = compute_radius(positions)
            
            if radius > best_radius:
                best_radius = radius
                best_positions = positions.copy()
    
    # Phase 2: Continuous optimization to polish the best discrete configuration
    if best_positions is not None:
        if time.time() < deadline - 10:
            try:
                result = minimize(
                    negative_radius,
                    best_positions,
                    method='L-BFGS-B',
                    bounds=[(-0.4999, -0.4999), (0.4999, 0.4999)] * N,
                    options={'maxiter': 200, 'maxfun': 20000, 'ftol': 1e-10}
                )
                
                if result.fun < -best_radius:
                    current_radius = -result.fun
                    if current_radius > best_radius:
                        best_radius = current_radius
                        best_positions = result.x.copy()
            except Exception:
                pass
        
        # Additional polish with tighter tolerance if time allows
        if time.time() < deadline - 5:
            try:
                result = minimize(
                    negative_radius,
                    best_positions,
                    method='L-BFGS-B',
                    bounds=[(-0.4999, -0.4999), (0.4999, 0.4999)] * N,
                    options={'maxiter': 100, 'maxfun': 10000, 'ftol': 1e-12}
                )
                
                if result.fun < -best_radius:
                    current_radius = -result.fun
                    if current_radius > best_radius:
                        best_radius = current_radius
                        best_positions = result.x.copy()
            except Exception:
                pass
    
    # Fallback to square grid if nothing was found
    if best_positions is None:
        best_positions = square_grid_initialization(N)
    
    # Final validation: ensure all positions are within bounds
    best_positions = np.clip(best_positions, -0.4999, 0.4999)
    
    return best_positions
# EVOLVE-BLOCK-END
