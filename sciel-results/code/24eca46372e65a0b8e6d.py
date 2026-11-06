"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def construct():
    """
    Pack N equal circles in unit disk using hexagonal lattice seeding 
    with iterative force-directed relaxation.
    """
    # Step 1: Generate initial positions using hexagonal lattice
    positions = _hexagonal_lattice_seed(N)
    
    # Step 2: Iterative force-directed relaxation
    # We'll use a simulated annealing-like approach with decreasing step size
    positions = _force_directed_relaxation(positions, max_iter=5000)
    
    # Step 3: Final local optimization to fine-tune
    positions = _final_local_optimize(positions)
    
    # Ensure validity
    radii = np.hypot(positions[:, 0], positions[:, 1])
    if np.any(radii > 1.0) or np.any(~np.isfinite(positions)):
        # Fallback: use a simpler packing
        positions = _simple_fallback(N)
    
    return positions[:N]


def _hexagonal_lattice_seed(N):
    """
    Generate N initial positions using a hexagonal lattice pattern
    scaled to fit within the unit disk.
    """
    # Estimate the spacing needed to fit N circles in unit disk
    # Area per circle ~ pi/(N) * some factor
    # For hexagonal packing, density is pi/(2*sqrt(3))
    # So effective radius ~ sqrt(area_per_circle / (pi/(2*sqrt(3))))
    
    # Start with a reasonable estimate and adjust
    # The optimal radius is ~0.053, so spacing ~ 2*0.053 = 0.106
    # But we need to account for the disk shape
    
    # Generate a large hexagonal lattice and select points within unit disk
    # with a margin
    
    # Estimate spacing: if we have N points in area pi, average area per point is pi/N
    # For hexagonal lattice, area per point = spacing^2 * sqrt(3)/2
    # So spacing ~ sqrt(2*pi/(N*sqrt(3)))
    spacing = math.sqrt(2 * math.pi / (N * math.sqrt(3)))
    
    # Add a margin factor since we're cutting to a disk
    spacing *= 0.95
    
    # Generate lattice points
    # Hexagonal lattice: (i*spacing + j*spacing/2, j*spacing*sqrt(3)/2)
    max_index = int(1.5 / spacing) + 3
    
    points = []
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            x = (i + j / 2) * spacing
            y = j * spacing * math.sqrt(3) / 2
            if math.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < N:
        # If not enough points, reduce spacing
        return _hexagonal_lattice_seed_with_adjusted_spacing(N, spacing * 0.9)
    
    # If more than N points, select the best N (closest to center for better packing)
    dists = np.hypot(points[:, 0], points[:, 1])
    order = np.argsort(dists)
    selected = points[order[:N]]
    
    return selected


def _hexagonal_lattice_seed_with_adjusted_spacing(N, spacing):
    """Generate hexagonal lattice with adjusted spacing."""
    max_index = int(1.5 / spacing) + 3
    
    points = []
    for i in range(-max_index, max_index + 1):
        for j in range(-max_index, max_index + 1):
            x = (i + j / 2) * spacing
            y = j * spacing * math.sqrt(3) / 2
            if math.hypot(x, y) < 1.0 - spacing / 2:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < N:
        # Reduce spacing further
        return _hexagonal_lattice_seed_with_adjusted_spacing(N, spacing * 0.95)
    
    dists = np.hypot(points[:, 0], points[:, 1])
    order = np.argsort(dists)
    selected = points[order[:N]]
    
    return selected


def _compute_min_radius(positions):
    """Compute the maximum common radius for the given positions."""
    n = len(positions)
    if n == 0:
        return 0.0
    
    # Minimum distance to container boundary
    dist_to_boundary = 1.0 - np.hypot(positions[:, 0], positions[:, 1])
    min_boundary = np.min(dist_to_boundary)
    
    if min_boundary < 0:
        return 0.0
    
    # Minimum pairwise distance / 2
    min_dist = _min_pairwise_distance(positions)
    min_pair = min_dist / 2.0
    
    return min(min_boundary, min_pair)


def _min_pairwise_distance(positions):
    """Compute minimum pairwise distance efficiently."""
    n = len(positions)
    if n < 2:
        return np.inf
    
    # Use a grid-based approach for efficiency
    # Estimate cell size based on expected minimum distance
    r_est = 0.05
    cell_size = 2 * r_est * 1.5
    
    # Create grid
    grid = {}
    for i, pos in enumerate(positions):
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        key = (cx, cy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    
    min_dist = np.inf
    for i in range(n):
        pos = positions[i]
        cx = int(pos[0] / cell_size)
        cy = int(pos[1] / cell_size)
        
        # Check neighboring cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (cx + dx, cy + dy)
                if key in grid:
                    for j in grid[key]:
                        if j > i:
                            d = np.linalg.norm(positions[i] - positions[j])
                            if d < min_dist:
                                min_dist = d
    
    return min_dist


def _force_directed_relaxation(positions, max_iter=5000):
    """
    Iterative force-directed relaxation to optimize circle packing.
    Uses repulsive forces between overlapping circles and attractive 
    forces toward the center for boundary violations.
    """
    positions = positions.copy()
    n = len(positions)
    
    # Temperature schedule for simulated annealing
    initial_temp = 0.01
    final_temp = 0.0001
    
    best_positions = positions.copy()
    best_radius = _compute_min_radius(positions)
    
    for iteration in range(max_iter):
        # Current temperature (exponential decay)
        temp = initial_temp * (final_temp / initial_temp) ** (iteration / max_iter)
        
        # Compute current radius
        current_r = _compute_min_radius(positions)
        
        if current_r > best_radius:
            best_radius = current_r
            best_positions = positions.copy()
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion: push apart circles that are too close
        # Use a threshold based on current best radius estimate
        threshold = 2 * max(best_radius, 0.01) * 1.2
        
        for i in range(n):
            for j in range(i + 1, n):
                d_vec = positions[i] - positions[j]
                d = np.linalg.norm(d_vec)
                if d < threshold and d > 1e-10:
                    # Strong repulsion when very close
                    overlap = threshold - d
                    force_mag = overlap / d
                    forces[i] += force_mag * d_vec
                    forces[j] -= force_mag * d_vec
        
        # Container constraint: push inward if too close to boundary
        radii = np.hypot(positions[:, 0], positions[:, 1])
        boundary_limit = 1.0 - best_radius
        for i in range(n):
            if radii[i] > boundary_limit:
                pos = positions[i]
                if radii[i] > 1e-10:
                    force_mag = (radii[i] - boundary_limit) / radii[i]
                    forces[i] -= force_mag * pos
        
        # Add small random perturbation (simulated annealing)
        forces += temp * np.random.randn(n, 2)
        
        # Update positions
        positions += forces * 0.05
        
        # Project back to valid region
        radii = np.hypot(positions[:, 0], positions[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            positions[mask] = positions[mask] * (1.0 / radii[mask])[:, None]
        
        # Check for convergence
        if iteration > 100 and abs(current_r - best_radius) < 1e-8:
            break
    
    return best_positions


def _final_local_optimize(positions):
    """
    Final local optimization using gradient-based methods to fine-tune positions.
    """
    n = len(positions)
    
    def objective(params):
        """Minimize negative of the minimum circle radius."""
        pos = params.reshape(n, 2)
        
        # Check validity
        radii = np.hypot(pos[:, 0], pos[:, 1])
        if np.any(radii > 1.0):
            return 1e10
        
        r = _compute_min_radius(pos)
        if r <= 0 or not np.isfinite(r):
            return 1e10
        
        return -r
    
    # Use L-BFGS-B optimizer
    result = minimize(
        objective,
        positions.flatten(),
        method='L-BFGS-B',
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.fun < 1e9:  # Valid result
        return result.x.reshape(n, 2)
    
    return positions


def _simple_fallback(N):
    """Simple fallback: evenly spaced on concentric rings."""
    # Use a simple ring structure
    num_rings = 15
    ring_counts = []
    remaining = N
    
    for i in range(num_rings - 1):
        # Outer rings get more circles
        count = int(remaining * (i + 1) / (num_rings * (num_rings + 1) / 2))
        count = max(1, count)
        ring_counts.append(count)
        remaining -= count
    
    ring_counts.append(remaining)
    
    positions = []
    for i, count in enumerate(ring_counts):
        r = (i + 1) / (num_rings + 0.5) * 0.9
        angles = np.arange(count) * 2 * np.pi / count + i * 0.1
        for angle in angles:
            positions.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(positions)
# EVOLVE-BLOCK-END
