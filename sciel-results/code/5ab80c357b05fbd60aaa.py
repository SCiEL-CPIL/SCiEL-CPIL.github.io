"""Pack 300 equal circles in a unit disk using hexagonal init + force-based relaxation."""
import numpy as np
import time

N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(points):
    """Compute the maximum common radius for a set of points."""
    n = len(points)
    if n == 0:
        return 0.0
    # Boundary constraint
    dists = np.sqrt(np.sum(points**2, axis=1))
    boundary_r = np.min(1.0 - dists)
    if boundary_r < 0:
        return 0.0
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist_matrix, np.inf)
    min_pair_dist = np.min(dist_matrix)
    pair_r = min_pair_dist / 2.0
    return min(boundary_r, pair_r)

def _hex_lattice(spacing):
    """Generate hexagonal lattice points within unit disk."""
    sqrt3 = np.sqrt(3.0)
    points = []
    k = int(2.0 / spacing) + 3
    for j in range(-k, k + 1):
        y = j * spacing * sqrt3 / 2.0
        if abs(y) > 1.0:
            continue
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            if x * x + y * y <= 1.0:
                points.append([x, y])
    return np.array(points)

def _hexagonal_init(n):
    """Initialize with hexagonal lattice pattern, trying multiple spacings."""
    best_points = None
    best_score = 0.0
    
    for spacing in [0.10, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14]:
        points = _hex_lattice(spacing)
        if len(points) >= n:
            # Keep the n closest to center
            dists = np.sqrt(np.sum(points**2, axis=1))
            idx = np.argsort(dists)[:n]
            points = points[idx]
            score = _compute_score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
    
    if best_points is not None:
        return best_points
    
    # Fallback: golden angle spiral
    angle = np.pi * (3.0 - np.sqrt(5.0))
    k = np.arange(n)
    r = np.sqrt((k + 0.5) / n)
    theta = k * angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def _force_relaxation(points, max_iter=20000, deadline=None):
    """
    Force-based relaxation with 1/d^3 repulsion and boundary forces.
    Returns (best_points, best_score).
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    
    step_size = 0.02
    min_step = 1e-10
    cooling = 0.9998
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Find the current minimum distance to set threshold
        current_min = np.min(dist_matrix)
        threshold = max(3.0 * current_min, 0.15)
        
        mask = dist_matrix < threshold
        np.fill_diagonal(mask, False)
        
        # Compute forces
        forces = np.zeros_like(points)
        
        # Pairwise repulsion: 1/d^3 weighting
        for i in range(n):
            neighbors = np.where(mask[i])[0]
            for j in neighbors:
                if i < j:
                    d = dist_matrix[i, j]
                    if d > 1e-10:
                        force_mag = 1.0 / (d**3 + 1e-8)
                        direction = (points[i] - points[j]) / d
                        forces[i] += force_mag * direction
                        forces[j] -= force_mag * direction
        
        # Boundary forces: push inward if close to boundary
        dists = np.sqrt(np.sum(points**2, axis=1))
        boundary_margin = 0.08
        for i in range(n):
            if dists[i] > 1.0 - boundary_margin:
                if dists[i] > 1e-10:
                    direction = -points[i] / dists[i]
                    excess = dists[i] - (1.0 - boundary_margin)
                    forces[i] += 3.0 * excess * direction
        
        # Normalize forces
        force_mags = np.sqrt(np.sum(forces**2, axis=1))
        max_force = np.max(force_mags) if len(force_mags) > 0 else 0
        if max_force > 1.0:
            forces /= max_force
        
        # Update positions
        points += step_size * forces
        
        # Keep points within the disk
        dists = np.sqrt(np.sum(points**2, axis=1))
        outside_mask = dists > 1.0
        if np.any(outside_mask):
            points[outside_mask] *= (1.0 / dists[outside_mask, np.newaxis]) * 0.999
        
        # Compute score
        score = _compute_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
        
        # Cool down
        step_size *= cooling
        if step_size < min_step:
            step_size = min_step
    
    return best_points, best_score

def _local_refinement(points, max_iter=5000, deadline=None):
    """
    Fine-grained local refinement: identify the limiting constraint and nudge
    the involved points. Returns (best_points, best_score).
    """
    points = points.copy().astype(np.float64)
    n = len(points)
    
    best_score = _compute_score(points)
    best_points = points.copy()
    
    lr = 0.002
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 3.0:
            break
        
        dists = np.sqrt(np.sum(points**2, axis=1))
        boundary_r = np.min(1.0 - dists)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair = np.min(dist_matrix)
        pair_r = min_pair / 2.0
        
        score = min(boundary_r, pair_r)
        
        if score >= best_score - 1e-12:
            # No improvement possible with this step; try smaller step
            if lr < 1e-8:
                break
            lr *= 0.5
            continue
        
        # Identify limiting constraint
        if boundary_r <= pair_r:
            # Move points near boundary inward
            for i in range(n):
                if dists[i] > 1.0 - (boundary_r + 0.01) and dists[i] > 1e-6:
                    direction = -points[i] / dists[i]
                    points[i] += lr * direction
        else:
            # Push limiting pairs apart
            # Find all pairs within 1.05x of min distance
            close_thresh = min_pair * 1.05
            close_mask = dist_matrix < close_thresh
            np.fill_diagonal(close_mask, False)
            
            close_pairs = np.argwhere(close_mask)
            for p in close_pairs:
                pi, pj = int(p[0]), int(p[1])
                if pi >= pj:
                    continue
                d_vec = points[pi] - points[pj]
                d_norm = np.linalg.norm(d_vec)
                if d_norm > 1e-6:
                    d_unit = d_vec / d_norm
                    weight = 1.0 / (1.0 + (dist_matrix[pi, pj] - min_pair) * 50)
                    points[pi] += lr * weight * d_unit
                    points[pj] -= lr * weight * d_unit
        
        # Keep within disk
        new_dists = np.sqrt(np.sum(points**2, axis=1))
        outside_mask = new_dists > 1.0
        if np.any(outside_mask):
            points[outside_mask] *= (1.0 / new_dists[outside_mask, np.newaxis]) * 0.999
        
        # Check improvement
        new_score = _compute_score(points)
        if new_score > best_score:
            best_score = new_score
            best_points = points.copy()
    
    return best_points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # 1. Initialize with hexagonal lattice
    points = _hexagonal_init(N)
    current_r = _compute_score(points)
    
    best_points = points.copy()
    best_r = current_r
    
    # 2. Coarse force-based relaxation
    new_points, new_r = _force_relaxation(best_points, max_iter=10000, deadline=deadline)
    if new_r > best_r:
        best_points = new_points
        best_r = new_r
    
    # 3. Fine local refinement
    new_points, new_r = _local_refinement(best_points, max_iter=5000, deadline=deadline)
    if new_r > best_r:
        best_points = new_points
        best_r = new_r
    
    # 4. If time remains, do another round of relaxation with smaller steps
    if time.time() < deadline - 10.0:
        new_points, new_r = _force_relaxation(best_points, max_iter=5000, deadline=deadline)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r
    
    # 5. Final refinement
    if time.time() < deadline - 5.0:
        new_points, new_r = _local_refinement(best_points, max_iter=2000, deadline=deadline)
        if new_r > best_r:
            best_points = new_points
            best_r = new_r
    
    return best_points
# EVOLVE-BLOCK-END
