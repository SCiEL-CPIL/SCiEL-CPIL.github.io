"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 300


# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s kill

    # Target radius from best known packing
    target_r = 0.0531542991469
    target_spacing = 2.0 * target_r

    # Initialize with hexagonal lattice
    points = _hex_init(target_spacing)

    if points is None or len(points) < N:
        # Fallback: simpler grid
        points = _grid_init(target_spacing)
        if points is None or len(points) < N:
            # Even simpler: random
            points = _random_init()
            points = points[:N]
            return points

    # Ensure we have exactly N points
    if len(points) > N:
        # Keep the N points that are closest to origin (better packing near center)
        radii = np.hypot(points[:, 0], points[:, 1])
        idx = np.argsort(radii, kind='stable')[:N]
        points = points[idx]
    elif len(points) < N:
        # Add random points in the disk
        while len(points) < N:
            angle = np.random.uniform(0, 2 * np.pi)
            r = np.sqrt(np.random.uniform(0, 1))
            points = np.vstack([points, [[r * np.cos(angle), r * np.sin(angle)]]])

    # Optimize using gradient-free methods
    best_points = points.copy()
    best_score = _score(points)

    # Phase 1: Coarse local search with Nelder-Mead style perturbations
    # Use a simple iterative improvement: move points to increase minimum distance
    step_size = 0.01
    for iteration in range(5000):
        if time.time() - start_time > time_limit:
            break

        # Try to improve by moving the most constrained point
        # Find the pair with minimum distance
        dists = _pairwise_distances(points)
        min_idx = np.argmin(dists)
        i, j = _linear_to_2d(min_idx, N)

        # Move points i and j apart slightly
        diff = points[i] - points[j]
        norm = np.linalg.norm(diff)
        if norm < 1e-10:
            continue
        diff = diff / norm

        # Try moving i in direction of diff, j in direction of -diff
        new_i = points[i] + step_size * 0.5 * diff
        new_j = points[j] - step_size * 0.5 * diff

        # Check if this improves the score
        old_score = _score(points)
        points[i] = new_i
        points[j] = new_j
        new_score = _score(points)

        if new_score > old_score:
            best_points = points.copy()
            best_score = new_score
            # Gradually decrease step size
            step_size *= 0.999
        else:
            points[i] = new_i - step_size * 0.5 * diff  # revert
            points[j] = new_j + step_size * 0.5 * diff  # revert
            step_size *= 0.995

    # Phase 2: Fine-tune with smaller steps
    step_size = 0.001
    for iteration in range(10000):
        if time.time() - start_time > time_limit:
            break

        # Randomly select a point and try to move it
        i = np.random.randint(N)
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        new_point = points[i] + step_size * direction

        # Check if new position is valid (inside unit circle)
        if np.hypot(new_point[0], new_point[1]) >= 1.0 - target_r * 1.01:
            continue

        old_score = _score(points)
        points[i] = new_point
        new_score = _score(points)

        if new_score > old_score:
            best_points = points.copy()
            best_score = new_score
        else:
            points[i] = new_point - step_size * direction  # revert

    # Final validation: ensure all points are inside the unit circle
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    mask = radii <= 1.0
    if np.any(~mask):
        # Push points back inside
        for i in range(len(best_points)):
            if radii[i] > 1.0:
                scale = (1.0 - target_r * 0.01) / radii[i]
                best_points[i] *= scale

    return best_points


def _hex_init(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    k = int(1.2 / spacing) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    if not points:
        return None
    return np.array(points)


def _grid_init(spacing):
    """Generate square grid points inside unit disk."""
    k = int(1.2 / spacing) + 2
    points = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = i * spacing
            y = j * spacing
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                points.append([x, y])
    if not points:
        return None
    return np.array(points)


def _random_init():
    """Generate random points inside unit disk."""
    points = []
    for _ in range(N):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.sqrt(np.random.uniform(0, 1))
        points.append([r * np.cos(angle), r * np.sin(angle)])
    return np.array(points)


def _pairwise_distances(points):
    """Compute all pairwise distances."""
    n = len(points)
    dists = np.zeros((n * (n - 1) // 2,))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dists[idx] = np.linalg.norm(points[i] - points[j])
            idx += 1
    return dists


def _linear_to_2d(idx, n):
    """Convert linear index to 2D indices for upper triangle."""
    # Solve: idx = i*n - i*(i+1)//2 + (j - i - 1) for j > i
    # Approximate solution
    i = int((2 * n - 1 - np.sqrt((2 * n - 1) ** 2 - 8 * idx)) / 2)
    j = idx - i * n + i * (i + 1) // 2 + i + 1
    return i, j


def _score(points):
    """Compute the packing score: minimum of half-min-pair-distance and min-distance-to-boundary."""
    n = len(points)
    if n < 2:
        return 0.0

    # Minimum pairwise distance
    min_pair_dist = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < min_pair_dist:
                min_pair_dist = d

    # Minimum distance to boundary
    min_boundary = float('inf')
    for i in range(n):
        r = np.hypot(points[i, 0], points[i, 1])
        dist_to_boundary = 1.0 - r
        if dist_to_boundary < min_boundary:
            min_boundary = dist_to_boundary

    return min(min_pair_dist / 2.0, min_boundary)
# EVOLVE-BLOCK-END
