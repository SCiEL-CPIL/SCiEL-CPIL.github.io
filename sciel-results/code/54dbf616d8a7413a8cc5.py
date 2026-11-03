"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(centers):
    """Compute the packing score (radius) for given centers."""
    # Distance to boundary
    boundary_dist = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    # Use vectorized computation for efficiency
    # For 200 points, we can do this efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    # Set diagonal to inf to avoid self-distance
    np.fill_diagonal(dist, np.inf)
    min_pair = np.min(dist)
    
    return min(min_boundary, min_pair / 2.0)

def _hex_grid_init(n):
    """Initialize centers on a hexagonal grid pattern."""
    # Hexagonal packing: rows offset by half spacing
    # For N points in a unit square, estimate spacing
    # Area per point ~ 1/N, hexagonal packing efficiency ~ 0.9069
    # spacing ~ sqrt(1/(N * 0.9069))
    spacing = np.sqrt(1.0 / (n * 0.9069))
    
    # Generate hexagonal grid points
    points = []
    row_height = spacing * np.sqrt(3) / 2
    
    # Determine how many rows and columns we need
    n_rows = int(np.ceil(1.0 / row_height)) + 2
    n_cols = int(np.ceil(1.0 / spacing)) + 2
    
    for i in range(-1, n_rows):
        y = -0.5 + (i + 0.5) * row_height
        offset = (spacing / 2) if (i % 2 != 0) else 0.0
        for j in range(-1, n_cols):
            x = -0.5 + (j + 0.5) * spacing + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
    
    points = np.array(points)
    
    # If we have more points than needed, select the best N
    if len(points) > n:
        # Select points that maximize minimum distance
        # Simple approach: greedy selection
        selected = []
        remaining = points.copy()
        
        while len(selected) < n and len(remaining) > 0:
            if len(selected) == 0:
                # Pick the point closest to center
                dists_to_center = np.sum(remaining**2, axis=1)
                idx = np.argmin(dists_to_center)
            else:
                # Pick the point that maximizes min distance to selected
                min_dists = []
                for p in remaining:
                    d = np.min(np.sqrt(np.sum((selected - p)**2, axis=1)))
                    min_dists.append(d)
                idx = np.argmax(min_dists)
            
            selected.append(remaining[idx])
            remaining = np.delete(remaining, idx, axis=0)
        
        points = np.array(selected)
    elif len(points) < n:
        # Need to add more points - this shouldn't happen with our grid
        # Pad with random points near existing ones
        while len(points) < n:
            base = points[np.random.randint(len(points))]
            new_pt = base + np.random.uniform(-0.01, 0.01, size=2)
            new_pt[0] = np.clip(new_pt[0], -0.49, 0.49)
            new_pt[1] = np.clip(new_pt[1], -0.49, 0.49)
            points = np.vstack([points, new_pt])
    
    return points[:n]

def _push_apart(centers, iterations=100, learning_rate=0.001):
    """Simple gradient ascent to push apart close pairs."""
    centers = centers.copy()
    
    for _ in range(iterations):
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Find the minimum distance pair
        min_idx = np.unravel_index(np.argmin(dist), dist.shape)
        i, j = min_idx
        min_d = dist[i, j]
        
        if min_d < 1e-10:
            continue
        
        # Push apart the closest pair
        direction = centers[i] - centers[j]
        direction /= (np.linalg.norm(direction) + 1e-10)
        
        # Check boundary constraints
        # Only move if it doesn't violate boundary
        new_i = centers[i] + direction * learning_rate
        new_j = centers[j] - direction * learning_rate
        
        # Check boundaries
        if (np.all(np.abs(new_i) <= 0.5) and np.all(np.abs(new_j) <= 0.5)):
            centers[i] = new_i
            centers[j] = new_j
    
    return centers

def _simulated_annealing(centers, t_start=0.01, t_end=0.0001, n_iter=5000):
    """Simulated annealing to improve the packing."""
    centers = centers.copy()
    current_score = _score(centers)
    best_centers = centers.copy()
    best_score = current_score
    
    for i in range(n_iter):
        # Linear cooling schedule
        t = t_start * (t_end / t_start) ** (i / n_iter)
        
        # Propose a random move
        idx = np.random.randint(len(centers))
        delta = np.random.uniform(-t, t, size=2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        
        # Check boundaries
        if not np.all(np.abs(new_centers) <= 0.5):
            continue
        
        new_score = _score(new_centers)
        
        # Accept if better, or with probability exp((new-old)/t)
        if new_score > current_score or np.random.random() < np.exp((new_score - current_score) / t):
            centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = centers.copy()
    
    return best_centers

def construct():
    start_time = time.time()
    deadline = 135  # Safety margin before 150s limit
    
    # Start with hexagonal grid initialization
    centers = _hex_grid_init(N)
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Multiple random restarts with hexagonal offsets
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Add small random perturbation to break symmetry
        perturbed = centers + np.random.uniform(-0.005, 0.005, size=(N, 2))
        perturbed = np.clip(perturbed, -0.49, 0.49)
        
        # Run simulated annealing
        if time.time() > deadline - 30:
            break
        
        result = _simulated_annealing(perturbed, t_start=0.01, t_end=0.0001, n_iter=3000)
        score = _score(result)
        
        if score > best_score:
            best_score = score
            best_centers = result.copy()
    
    # Final polish: coordinate ascent
    if time.time() < deadline - 10:
        best_centers = _push_apart(best_centers, iterations=200, learning_rate=0.0005)
        final_score = _score(best_centers)
        
        if final_score > best_score:
            best_score = final_score
    
    return best_centers
# EVOLVE-BLOCK-END
