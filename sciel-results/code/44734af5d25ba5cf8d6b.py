"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers, dtype=np.float64)
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Check bounds
    if np.any(np.abs(centers) > 0.5 + 1e-10):
        return 0.0
    # Min distance to boundary (distance from center to nearest wall)
    boundary_dist = np.min(0.5 - np.abs(centers))
    if boundary_dist < -1e-10:
        return 0.0
    boundary_dist = max(boundary_dist, 0.0)
    # Min pairwise distance / 2
    min_pair_dist = np.inf
    for i in range(N):
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        if len(dists) > 0:
            min_pair_dist = min(min_pair_dist, np.min(dists))
    if min_pair_dist == np.inf:
        min_pair_dist = 1.0
    return min(boundary_dist, min_pair_dist / 2.0)

def _hexagonal_init():
    """Create a hexagonal-like initial configuration."""
    # Hexagonal packing is denser than square
    # For N circles, find rows and columns that fit
    # Hexagonal: rows offset by half spacing
    # Try to find good row/col split
    
    best_centers = None
    best_r = 0
    
    for rows in range(10, 25):
        cols_per_row = []
        remaining = N
        for r in range(rows):
            if r < rows - 1:
                c = remaining // (rows - r)
            else:
                c = remaining
            cols_per_row.append(c)
            remaining -= c
        
        # Estimate spacing
        # In hexagonal, vertical spacing = s * sqrt(3)/2 where s is horizontal spacing
        # Total width ~ (max_cols - 0.5) * s (for offset rows)
        # Total height ~ (rows - 1) * s * sqrt(3)/2 + s
        
        max_cols = max(cols_per_row) if cols_per_row else 1
        # Approximate: width = max_cols * s, height = (rows-1)*s*sqrt(3)/2 + s
        # We want width <= 1 and height <= 1
        # s <= 1/max_cols and s <= 1/((rows-1)*sqrt(3)/2 + 1)
        s1 = 1.0 / max_cols
        s2 = 1.0 / ((rows - 1) * np.sqrt(3) / 2 + 1)
        s = min(s1, s2)
        
        if s < 0.01:
            continue
        
        # Generate hexagonal positions
        centers_list = []
        y_pos = -0.5 + s / 2
        for r in range(rows):
            n_cols = cols_per_row[r]
            if r % 2 == 0:
                x_start = -0.5 + s / 2
            else:
                x_start = -0.5 + s * 0.75  # offset by 3/4 s for hexagonal
            
            for c in range(n_cols):
                x = x_start + c * s
                y = y_pos + r * s * np.sqrt(3) / 2
                centers_list.append([x, y])
            if len(centers_list) >= N:
                break
        
        centers = np.array(centers_list[:N])
        if len(centers) < N:
            continue
        
        # Check if all within bounds
        if np.any(np.abs(centers) > 0.5 + 1e-9):
            continue
        
        r = _compute_radius(centers)
        if r > best_r:
            best_r = r
            best_centers = centers
    
    if best_centers is not None:
        return best_centers
    else:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with hexagonal initialization
    centers = _hexagonal_init()
    current_radius = _compute_radius(centers)
    best_centers = centers.copy()
    best_radius = current_radius
    
    # Local optimization: iteratively move circles to increase min distance
    rng = np.random.RandomState(123)
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline and no_improve < max_no_improve:
        improved = False
        
        # Find the current bottleneck: which constraint is limiting
        # Compute all pairwise distances and boundary distances
        boundary_dists = 0.5 - np.abs(centers)
        min_boundary = np.min(boundary_dists)
        
        # Find closest pair
        min_pair_dist = np.inf
        min_pair_idx = (0, 1)
        for i in range(N):
            diffs = centers[i+1:] - centers[i]
            dists = np.sqrt(np.sum(diffs**2, axis=1))
            if len(dists) > 0:
                min_d = np.min(dists)
                if min_d < min_pair_dist:
                    min_pair_dist = min_d
                    min_j = i + 1 + np.argmin(dists)
                    min_pair_idx = (i, min_j)
        
        current_r = min(min_boundary, min_pair_dist / 2)
        
        # Try to improve by moving circles
        # Strategy: move circles that are in the bottleneck
        # If boundary is limiting, move those circles inward
        # If pair is limiting, move those two circles apart
        
        # Try multiple random moves
        for _ in range(300):
            if time.time() > deadline:
                break
            
            # Random choice: move a random circle
            idx = rng.randint(N)
            
            # Determine a good direction to move
            # Move away from nearest neighbor or toward center if near boundary
            nearest_dist = np.inf
            nearest_idx = -1
            for j in range(N):
                if j == idx:
                    continue
                d = np.linalg.norm(centers[idx] - centers[j])
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_idx = j
            
            # Direction: away from nearest neighbor
            if nearest_idx >= 0 and nearest_dist < 0.1:
                direction = centers[idx] - centers[nearest_idx]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
            else:
                direction = rng.randn(2)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction /= norm
            
            # Also consider boundary: if close to boundary, add inward component
            for coord in range(2):
                if centers[idx, coord] > 0.45:
                    direction[coord] -= 0.5
                elif centers[idx, coord] < -0.45:
                    direction[coord] += 0.5
            
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction /= norm
            
            # Try different step sizes
            for step in [0.005, 0.01, 0.02, 0.003]:
                new_centers = centers.copy()
                new_centers[idx] += direction * step
                new_centers = np.clip(new_centers, -0.5, 0.5)
                
                r = _compute_radius(new_centers)
                if r > current_radius + 1e-11:
                    centers = new_centers
                    current_radius = r
                    if r > best_radius:
                        best_radius = r
                        best_centers = centers.copy()
                    improved = True
                    break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    # Final validation
    final_r = _compute_radius(best_centers)
    if final_r > best_radius:
        best_radius = final_r
    
    return best_centers

# EVOLVE-BLOCK-END
