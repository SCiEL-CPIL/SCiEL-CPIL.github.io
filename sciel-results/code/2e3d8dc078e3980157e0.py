"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the valid radius for a set of points."""
    n = len(pts)
    # Boundary constraint
    r_boundary = 0.5 - np.max(np.abs(pts), axis=1)
    r_min_boundary = np.min(r_boundary)
    # Pairwise constraint
    if n > 1:
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        r_min_pair = np.min(dist) / 2.0
    else:
        r_min_pair = r_min_boundary
    return min(r_min_boundary, r_min_pair)


def _hexagonal_init(n):
    """Create a hexagonal close-packed initial configuration."""
    # Use a hexagonal lattice pattern
    # For 50 circles, we can use rows of varying length
    # Hexagonal packing: row spacing = sqrt(3)/2 * d, col spacing = d
    # Let's figure out a good arrangement
    # Try different numbers of columns
    best_pts = None
    best_r = -1
    
    # Hexagonal arrangement: rows alternate offset
    # For n=50, try 7x7 grid with some removed, or 6-8-6-8-6-8-... patterns
    # Let's use a systematic approach: fill a hexagonal lattice
    
    for n_cols in range(5, 12):
        # Calculate how many rows we need
        # Hexagonal: each row has n_cols or n_cols-1 points
        # Row height = sqrt(3)/2 * spacing
        # Try to fit n points
        rows_needed = int(np.ceil(n / n_cols))
        
        # Create hexagonal lattice
        spacing = 1.0 / (n_cols - 0.5)  # approximate
        row_height = spacing * np.sqrt(3) / 2
        
        # Total height
        total_h = (rows_needed - 1) * row_height
        # Scale to fit in [-0.5, 0.5]
        scale = 1.0 / max(n_cols - 0.5, total_h / spacing)
        
        pts = []
        for row in range(rows_needed):
            y = -0.5 + row * row_height * scale
            offset = (row % 2) * 0.5 * spacing * scale
            for col in range(n_cols):
                x = -0.5 + (col + 0.5) * spacing * scale + offset
                if len(pts) < n:
                    # Keep within bounds
                    x = np.clip(x, -0.5, 0.5)
                    y = np.clip(y, -0.5, 0.5)
                    pts.append([x, y])
        
        if len(pts) == n:
            pts = np.array(pts)
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best_pts = pts.copy()
    
    if best_pts is None:
        # Fallback: square grid
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_pts = grid[:n]
    
    return best_pts


def _optimize(pts, max_time=100.0):
    """Optimize the packing using iterative relaxation."""
    start_time = time.time()
    pts = pts.copy().astype(float)
    n = len(pts)
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Simulated annealing style optimization
    lr = 0.01
    for iteration in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Boundary repulsion
        for i in range(n):
            for k in range(2):
                if pts[i, k] > 0.5:
                    forces[i, k] -= (pts[i, k] - 0.5) * 10
                elif pts[i, k] < -0.5:
                    forces[i, k] += (-0.5 - pts[i, k]) * 10
        
        # Pairwise repulsion for overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-10:
                    # Random small push
                    forces[i] += np.random.randn(2) * 0.001
                    forces[j] -= np.random.randn(2) * 0.001
                elif dist < 2 * best_r * 1.05:
                    # Repel to maintain separation
                    force_mag = (2 * best_r - dist) / dist
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Update positions
        pts += forces * lr
        
        # Keep within bounds (soft constraint)
        pts = np.clip(pts, -0.49, 0.49)
        
        # Compute new radius
        new_r = _compute_radius(pts)
        if new_r > best_r:
            best_r = new_r
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= 0.9999
    
    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Start with hexagonal initial configuration
    pts = _hexagonal_init(N)
    r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = r
    
    # Run optimization
    if time.time() < deadline:
        pts, r = _optimize(pts, max_time=deadline - time.time() - 5.0)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
    
    # Ensure validity: clip to bounds
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
