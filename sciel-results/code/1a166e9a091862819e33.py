"""Rewritten program: local search for packing 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree
import time

def _score(centers):
    """Compute the score: min of half min-pair-distance and min distance to border."""
    # Distance to borders
    d_border = np.min(np.minimum(0.5 - np.abs(centers[:, 0]), 0.5 - np.abs(centers[:, 1])), axis=1)
    min_border = np.min(d_border)
    
    # Min pair distance
    tree = cKDTree(centers)
    # Query nearest neighbors, skip self
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    
    return min(min_border, min_pair / 2.0)

def _make_hex_initial():
    """Create a hexagonal packing initial configuration."""
    # For N=200, we can use a hexagonal grid
    # Hexagonal packing: rows with alternating offsets
    # Let's use a rectangular grid with hexagonal offset
    # For a square, we can try different aspect ratios
    
    best_centers = None
    best_score = -1
    
    # Try different grid configurations
    for rows in range(10, 21):
        cols = int(np.ceil(N / rows))
        if rows * cols < N:
            continue
        
        # Hexagonal spacing
        # In hexagonal packing, horizontal spacing = 2r, vertical spacing = r*sqrt(3)
        # For a square of side 1, we want to fit rows*cols circles
        # Let's parameterize by the horizontal spacing
        
        # Try to fit in the square
        # Width needed: (cols - 1) * dx + 2*r
        # Height needed: (rows - 1) * dy + 2*r
        # where dy = dx * sqrt(3)/2 for hexagonal
        
        # For hexagonal: dx = 2r, dy = r*sqrt(3)
        # Width = (cols-1)*2r + 2r = cols*2r
        # Height = (rows-1)*r*sqrt(3) + 2r
        
        # We want to maximize r such that width <= 1 and height <= 1
        # r <= 1/(2*cols) and r <= 1/((rows-1)*sqrt(3) + 2)
        
        r1 = 0.5 / cols
        r2 = 0.5 / ((rows - 1) * np.sqrt(3) + 2)
        r = min(r1, r2)
        
        # Create hexagonal grid
        centers = []
        dx = 2 * r
        dy = r * np.sqrt(3)
        
        for row in range(rows):
            for col in range(cols):
                x = -0.5 + r + col * dx + (row % 2) * dx / 2
                y = -0.5 + r + row * dy
                if x < 0.5 and y < 0.5 and x > -0.5 and y > -0.5:
                    centers.append([x, y])
        
        if len(centers) >= N:
            centers = np.array(centers[:N])
            # Center the configuration
            cx = np.mean(centers[:, 0])
            cy = np.mean(centers[:, 1])
            centers = centers - np.array([cx, cy])
            # Scale to fit
            # Actually, let's just use it as is if it fits
            if np.all(centers[:, 0] >= -0.5) and np.all(centers[:, 0] <= 0.5) and \
               np.all(centers[:, 1] >= -0.5) and np.all(centers[:, 1] <= 0.5):
                s = _score(centers)
                if s > best_score:
                    best_score = s
                    best_centers = centers.copy()
    
    if best_centers is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
    
    return best_centers

def _local_search(centers, max_iters=5000, step_scale=0.5):
    """Local search to improve the packing."""
    centers = centers.copy()
    n = len(centers)
    
    # Initial step size based on current packing density
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_dist = np.min(dists[:, 1])
    step = min_dist * step_scale
    
    best_score = _score(centers)
    current_score = best_score
    
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Randomly select a center to move
        idx = np.random.randint(n)
        
        # Try random perturbations
        for _ in range(10):
            new_center = centers[idx].copy()
            new_center += np.random.randn(2) * step
            
            # Check bounds
            if new_center[0] < -0.5 or new_center[0] > 0.5 or \
               new_center[1] < -0.5 or new_center[1] > 0.5:
                continue
            
            old = centers[idx]
            centers[idx] = new_center
            
            new_score = _score(centers)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                # Annealing: occasionally accept worse solutions
                # But for now, just accept improvements
            else:
                centers[idx] = old
        
        # Adapt step size
        if it % 100 == 0 and it > 0:
            # Decay step size
            step *= 0.98
            # Or occasionally increase
            if np.random.rand() < 0.1:
                step *= 1.2
    
    return centers, best_score

def construct():
    """Construct the centers of N equal circles inside the square [-1/2, 1/2]^2."""
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Generate initial configuration
    centers = _make_hex_initial()
    
    # Local search
    best_centers = centers.copy()
    best_score = _score(centers)
    
    # Multiple restarts
    num_restarts = 5
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # For restarts, perturb the best solution
        if r == 0:
            current = centers.copy()
        else:
            current = best_centers.copy()
            # Add noise
            current += np.random.randn(N, 2) * 0.01
            # Clip to bounds
            current[:, 0] = np.clip(current[:, 0], -0.49, 0.49)
            current[:, 1] = np.clip(current[:, 1], -0.49, 0.49)
        
        # Local search
        improved, score = _local_search(current, max_iters=2000)
        
        if score > best_score:
            best_score = score
            best_centers = improved.copy()
    
    # Final local search on the best solution
    if time.time() < deadline - 10:
        final, final_score = _local_search(best_centers, max_iters=3000, step_scale=0.3)
        if final_score > best_score:
            best_centers = final
            best_score = final_score
    
    # Ensure valid output
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
