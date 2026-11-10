"""Improved packing: hexagonal init + physics-based repulsion + local polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist


def _compute_radius(pts):
    """Compute the max radius for given centers in [-0.5, 0.5]^2."""
    if len(pts) < 2:
        return 0.0
    d = pdist(pts, 'euclidean')
    min_pair = d.min() if len(d) > 0 else 0.0
    bounds = 0.5 - np.abs(pts).max(axis=1)
    min_bound = bounds.min()
    r = min(min_pair / 2.0, min_bound)
    return r


def _hexagonal_init(n):
    """Generate a hexagonal close-packed initial configuration."""
    # Hexagonal lattice spacing
    # For n circles, we need to figure out the right grid size
    # Hex packing: rows offset by half spacing
    # Try different numbers of columns
    best_pts = None
    best_score = -1
    
    for cols in range(10, 25):
        rows_needed = int(np.ceil(n / cols))
        if rows_needed < 5:
            continue
        
        # Hexagonal spacing
        # Horizontal spacing = 1, vertical spacing = sqrt(3)/2 * horizontal
        # We want to fit in [-0.5, 0.5]^2
        # Let's parameterize by the horizontal spacing
        h_spacing = 1.0  # will scale
        v_spacing = h_spacing * np.sqrt(3) / 2
        
        # Generate points
        pts = []
        for row in range(rows_needed):
            for col in range(cols):
                if len(pts) >= n:
                    break
                x = (col - (cols - 1) / 2.0) * h_spacing
                y = (row - (rows_needed - 1) / 2.0) * v_spacing
                if row % 2 == 1:
                    x += h_spacing / 2.0
                pts.append([x, y])
        
        pts = np.array(pts[:n])
        
        # Scale to fit in the square
        max_coord = np.abs(pts).max()
        if max_coord > 0:
            pts /= max_coord * 1.02  # slight margin
            pts *= 0.5
        
        # Compute a rough score (min distance)
        if len(pts) >= 2:
            d = pdist(pts, 'euclidean')
            min_d = d.min()
            bounds = 0.5 - np.abs(pts).max(axis=1)
            min_bound = bounds.min()
            score = min(min_d / 2.0, min_bound)
        else:
            score = 0
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Fallback: simple grid
    if best_pts is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        best_pts = np.array([(x, y) for y in ticks for x in ticks])[:n]
    
    return best_pts


def _repulsion_step(pts, n_iter=5000, lr=0.001):
    """Physics-based repulsion: push points apart and keep them in bounds."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(n_iter):
        if it % 100 == 0 and it > 0:
            # Decay learning rate
            lr *= 0.95
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion (use a cutoff for efficiency)
        # For 200 points, full pairwise is 19900 pairs - manageable
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    # Random push
                    diff = np.random.randn(2) * 1e-5
                    dist = np.linalg.norm(diff)
                
                # Repulsive force: stronger when close
                # Use a soft potential: F = 1/dist^2 in direction of separation
                if dist < 0.1:  # only interact when close
                    force_mag = 1.0 / (dist * dist + 1e-6)
                    force = force_mag * diff / dist
                    forces[i] += force
                    forces[j] -= force
        
        # Boundary forces: push back if too close to edge
        margin = 0.05
        for i in range(n):
            for k in range(2):
                if pts[i][k] > 0.5 - margin:
                    forces[i][k] -= (pts[i][k] - (0.5 - margin)) * 10
                elif pts[i][k] < -0.5 + margin:
                    forces[i][k] += ((-0.5 + margin) - pts[i][k]) * 10
        
        # Update positions
        pts += lr * forces
        
        # Clamp to valid region
        pts = np.clip(pts, -0.499, 0.499)
    
    return pts


def _local_polish(pts, n_iter=20000, deadline=None):
    """Fine local search to polish the solution."""
    pts = pts.copy()
    n = len(pts)
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Adaptive step size
    base_step = 0.005
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Occasionally reduce step size
        if it % 5000 == 0 and it > 0:
            base_step *= 0.8
        
        idx = np.random.randint(n)
        orig = pts[idx].copy()
        
        best_move = None
        best_move_r = current_r
        
        # Try several moves
        n_trials = 20
        for _ in range(n_trials):
            direction = np.random.randn(2)
            direction /= np.linalg.norm(direction)
            step = base_step * (0.3 + np.random.rand() * 1.4)
            new_pos = orig + direction * step
            new_pos = np.clip(new_pos, -0.499, 0.499)
            
            temp_pts = pts.copy()
            temp_pts[idx] = new_pos
            temp_r = _compute_radius(temp_pts)
            
            if temp_r > best_move_r:
                best_move_r = temp_r
                best_move = new_pos
        
        if best_move is not None:
            pts[idx] = best_move
            current_r = best_move_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
    
    return best_pts, best_r


def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin before 150s kill
    
    np.random.seed(42)
    
    # Step 1: Hexagonal initialization
    pts = _hexagonal_init(N)
    r0 = _compute_radius(pts)
    
    # Step 2: Physics-based repulsion to spread points
    # Run in chunks to check deadline
    chunk_size = 2000
    total_iter = 8000
    for start in range(0, total_iter, chunk_size):
        if time.time() > deadline:
            break
        end = min(start + chunk_size, total_iter)
        # Do a mini repulsion step
        n_iter_chunk = end - start
        lr = 0.001 * (0.9 ** (start // chunk_size))
        for it in range(n_iter_chunk):
            forces = np.zeros_like(pts)
            for i in range(N):
                for j in range(i + 1, N):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-10:
                        diff = np.random.randn(2) * 1e-5
                        dist = np.linalg.norm(diff)
                    if dist < 0.15:
                        force_mag = 1.0 / (dist * dist + 1e-6)
                        force = force_mag * diff / dist
                        forces[i] += force
                        forces[j] -= force
            
            # Boundary forces
            margin = 0.05
            for i in range(N):
                for k in range(2):
                    if pts[i][k] > 0.5 - margin:
                        forces[i][k] -= (pts[i][k] - (0.5 - margin)) * 10
                    elif pts[i][k] < -0.5 + margin:
                        forces[i][k] += ((-0.5 + margin) - pts[i][k]) * 10
            
            pts += lr * forces
            pts = np.clip(pts, -0.499, 0.499)
    
    r1 = _compute_radius(pts)
    
    # Step 3: Local polish
    if time.time() < deadline:
        pts, r2 = _local_polish(pts, n_iter=30000, deadline=deadline)
    else:
        r2 = r1
    
    # Ensure all points are strictly inside
    pts = np.clip(pts, -0.499, 0.499)
    
    return pts
# EVOLVE-BLOCK-END
