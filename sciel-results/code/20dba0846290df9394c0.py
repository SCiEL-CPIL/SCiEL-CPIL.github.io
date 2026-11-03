"""Circle packing optimizer for 300 equal circles in a unit disk."""
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

N = 300

def _hex_init(n_points, seed=0):
    """Generate a hexagonal lattice and select n_points best points."""
    rng = np.random.default_rng(seed)
    best_pts = None
    best_r = 0.0

    # Try a few different spacings and rotations
    for attempt in range(10):
        # Spacing estimation
        # Area per circle ~ pi * r^2. Total area ~ pi * 1^2.
        # N * pi * r^2 <= pi => r <= 1/sqrt(N)
        # Hex packing density is pi / (2*sqrt(3)) ~ 0.9069
        # So r ~ 0.95 / sqrt(N)
        base_spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        spacing = base_spacing * (0.95 + 0.015 * attempt)

        k = int(1.0 / spacing) + 5
        rows = []
        # Use a fixed rotation per attempt to break symmetry
        rot_angle = rng.uniform(0, 2 * np.pi)
        cos_a = np.cos(rot_angle)
        sin_a = np.sin(rot_angle)
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Apply rotation
                x_rot = x * cos_a - y * sin_a
                y_rot = x * sin_a + y * cos_a
                r = np.hypot(x_rot, y_rot)
                if r <= 1.0 - spacing / 2.0:
                    rows.append((x_rot, y_rot, r))

        if len(rows) < n_points:
            continue

        pts = np.array([(r[0], r[1]) for r in rows])
        radii = np.array([r[2] for r in rows])
        order = np.argsort(radii, kind="stable")
        selected = pts[order[:n_points]]

        # Add noise for attempts > 0
        if attempt > 0:
            selected = selected + rng.normal(0, spacing * 0.04, selected.shape)
            norms = np.linalg.norm(selected, axis=1)
            mask = norms < 1.0
            selected = selected[mask]
            if len(selected) < n_points:
                continue
            selected = selected[:n_points]

        d = cdist(selected, selected)
        np.fill_diagonal(d, np.inf)
        min_pair = d.min() / 2.0
        min_bound = 1.0 - np.linalg.norm(selected, axis=1).max()
        r = min(min_pair, min_bound)

        if r > best_r:
            best_r = r
            best_pts = selected.copy()

    if best_pts is None:
        # Fallback
        spacing = 2.0 / (np.sqrt(n_points) + 1.0) * 0.95
        k = int(1.0 / spacing) + 5
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    rows.append((x, y))
        pts = np.array(rows)
        order = np.argsort(np.linalg.norm(pts, axis=1), kind="stable")
        best_pts = pts[order[:n_points]]

    return best_pts

def _compute_radius(pts):
    """Compute the packing radius for given points."""
    norms = np.linalg.norm(pts, axis=1)
    min_bound = 1.0 - norms.max()
    if min_bound <= 0:
        return 0.0
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    min_pair = d.min() / 2.0
    return min(min_pair, min_bound)

def _delaunay_optimize(pts, max_iters=500, deadline=None):
    """
    Optimize using Delaunay triangulation to identify active constraints.
    This focuses the optimization on the nearest neighbors defined by the
    Delaunay graph, which is more efficient than all-pair optimization.
    """
    pos = pts.copy().astype(np.float64)
    n = len(pos)
    best_pos = pos.copy()
    best_r = _compute_radius(best_pos)
    
    lr = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Compute Delaunay triangulation
        try:
            tri = Delaunay(pos)
        except Exception:
            # Fallback if Delaunay fails (e.g., points too close)
            tri = None
            
        if tri is not None:
            # Get edges from Delaunay triangulation
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    edge = tuple(sorted((simplex[i], simplex[j])))
                    edges.add(edge)
            
            # Convert to numpy array for vectorization
            edge_list = np.array(list(edges))
            if len(edge_list) == 0:
                break
                
            i_idx = edge_list[:, 0]
            j_idx = edge_list[:, 1]
            
            # Compute distances for these edges
            diffs = pos[i_idx] - pos[j_idx]
            dists = np.linalg.norm(diffs, axis=1)
            
            # Gradient of distance with respect to points
            # d/dx_i |x_i - x_j| = (x_i - x_j) / |x_i - x_j|
            # d/dx_j |x_i - x_j| = -(x_i - x_j) / |x_i - x_j|
            
            safe_dists = np.where(dists > 1e-10, dists, 1.0)
            unit_vecs = diffs / safe_dists[:, np.newaxis]
            
            # Accumulate gradients
            grad = np.zeros_like(pos)
            np.add.at(grad, i_idx, unit_vecs)
            np.add.at(grad, j_idx, -unit_vecs)
            
            # Boundary gradient
            norms = np.linalg.norm(pos, axis=1)
            safe_norms = np.where(norms > 1e-10, norms, 1.0)
            boundary_grad = -pos / safe_norms[:, np.newaxis]
            
            # Combine gradients
            # We want to maximize the minimum of (dist/2, 1-norm)
            # This is equivalent to minimizing the negative of the soft-min
            # For simplicity, we use a weighted sum approach here
            # Weight boundary gradient more if points are close to boundary
            boundary_weight = 1.0 + 5.0 * np.clip((norms - 0.9) / 0.1, 0, 1)
            
            total_grad = grad * 0.5 + boundary_grad * boundary_weight[:, np.newaxis] * 0.5
            
            # Update
            pos_new = pos + lr * total_grad
            
            # Project to disk
            dists_new = np.linalg.norm(pos_new, axis=1)
            mask_out = dists_new > 0.999
            if np.any(mask_out):
                pos_new[mask_out] *= 0.999 / dists_new[mask_out, np.newaxis]
                
            r_new = _compute_radius(pos_new)
            if r_new > best_r:
                best_r = r_new
                best_pos = pos_new.copy()
                pos = pos_new
            else:
                # Reduce learning rate if no improvement
                lr *= 0.95
        else:
            break
            
    return best_pos, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s deadline, 10s safety margin
    
    # Initial hexagonal lattice
    pts = _hex_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(pts)
    
    # Delaunay-based optimization
    pts, r = _delaunay_optimize(pts, max_iters=200, deadline=deadline - 30)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
    
    # Check time
    if time.time() > deadline - 30:
        return best_pts
    
    # Second pass with more iterations
    pts, r = _delaunay_optimize(pts, max_iters=300, deadline=deadline - 10)
    if r > best_r:
        best_r = r
        best_pts = pts.copy()
        
    return best_pts
