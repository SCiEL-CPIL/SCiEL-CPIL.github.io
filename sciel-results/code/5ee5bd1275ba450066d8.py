"""Circle packing in unit square: 26 circles, maximize sum of radii."""
import numpy as np
from scipy.spatial import Voronoi, distance
import time

N = 26
DEADLINE_SECONDS = 115  # safety margin before 120s kill


def _hexagonal_init(n, rng):
    """Hexagonal close-packing initialization, then random subset of n."""
    # Generate a hexagonal lattice covering slightly more than unit square
    # Hex packing: rows offset, vertical spacing = sqrt(3)/2 * spacing
    # We want to generate enough candidates and pick n
    pts = []
    # Try various spacings to fill the square
    for spacing in [0.1, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.3]:
        h = spacing * np.sqrt(3) / 2
        for row in range(int(1 / h) + 2):
            y = h * row + 0.05
            if y > 0.95:
                break
            offset = 0 if row % 2 == 0 else spacing / 2
            for col in range(int(1 / spacing) + 2):
                x = spacing * col + offset + 0.05
                if x > 0.95:
                    break
                pts.append([x, y])
    pts = np.array(pts)
    # Add some random points too
    rand_pts = rng.uniform(0.05, 0.95, size=(n * 3, 2))
    all_pts = np.vstack([pts, rand_pts])
    
    # Select n points: prefer well-separated ones
    # Greedy: pick random point, then always pick the one farthest from all selected
    selected = []
    remaining = all_pts.copy()
    # Start with a random point
    idx = rng.integers(len(remaining))
    selected.append(remaining[idx])
    remaining = np.delete(remaining, idx, axis=0)
    
    for _ in range(n - 1):
        if len(remaining) == 0:
            break
        # Compute min distance from each remaining to all selected
        if len(selected) > 0:
            sel = np.array(selected)
            d = np.array([np.min(np.linalg.norm(remaining - sel, axis=1)) for _ in range(1)])
            # Actually compute properly
            d = np.min(np.linalg.norm(remaining[:, None, :] - sel[None, :, :], axis=2), axis=1)
            # Add small random perturbation to break ties
            d += rng.uniform(0, 0.01, size=len(d))
        else:
            d = np.zeros(len(remaining))
        idx = np.argmax(d)
        selected.append(remaining[idx])
        remaining = np.delete(remaining, idx, axis=0)
    
    return np.array(selected[:n])


def _compute_radii(positions):
    """Compute max radius for each circle given positions using Voronoi."""
    n = len(positions)
    # Add boundary points to make Voronoi cells bounded
    # Extend the domain
    margin = 2.0
    boundary_pts = []
    for i in range(50):
        t = i / 49.0
        # Bottom
        boundary_pts.append([-margin + t * (1 + 2*margin), -margin])
        # Top
        boundary_pts.append([-margin + t * (1 + 2*margin), 1 + margin])
        # Left
        boundary_pts.append([-margin, -margin + t * (1 + 2*margin)])
        # Right
        boundary_pts.append([1 + margin, -margin + t * (1 + 2*margin)])
    boundary_pts = np.array(boundary_pts)
    
    all_pts = np.vstack([positions, boundary_pts])
    
    try:
        vor = Voronoi(all_pts)
    except Exception:
        # Fallback: use pairwise distances
        radii = np.zeros(n)
        for i in range(n):
            # Distance to boundaries
            r_boundary = min(positions[i, 0], 1 - positions[i, 0],
                             positions[i, 1], 1 - positions[i, 1])
            # Distance to other circles
            d = np.linalg.norm(positions - positions[i], axis=1)
            d[i] = np.inf
            r_circle = np.min(d) / 2
            radii[i] = min(r_boundary, r_circle)
        return radii
    
    radii = np.zeros(n)
    for i in range(n):
        # Get the Voronoi cell for point i
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            # Unbounded cell (shouldn't happen with boundary points)
            r_boundary = min(positions[i, 0], 1 - positions[i, 0],
                             positions[i, 1], 1 - positions[i, 1])
            radii[i] = r_boundary
            continue
        
        # Compute inradius of the Voronoi cell
        # The inradius is the max distance from point i to the cell boundary
        # which equals the min distance to each edge of the cell
        cell_verts = vor.vertices[region]
        
        # Also consider distance to square boundaries
        r_boundary = min(positions[i, 0], 1 - positions[i, 0],
                         positions[i, 1], 1 - positions[i, 1])
        
        # Distance to each edge of the Voronoi cell
        r_cell = np.inf
        for k in range(len(cell_verts)):
            v1 = cell_verts[k]
            v2 = cell_verts[(k + 1) % len(cell_verts)]
            # Distance from positions[i] to line segment v1-v2
            d = distance.point_to_line(positions[i], v1, v2)
            r_cell = min(r_cell, d)
        
        radii[i] = min(r_boundary, r_cell)
    
    return radii


def _total_radius(positions):
    """Sum of radii for given positions."""
    radii = _compute_radii(positions)
    return np.sum(radii)


def _local_search(positions, rng, time_budget):
    """Perturbation local search to improve total radius."""
    n = len(positions)
    best_pos = positions.copy()
    best_val = _total_radius(best_pos)
    
    step_size = 0.05
    start_time = time.time()
    
    while time.time() - start_time < time_budget:
        # Generate perturbation
        perturb = rng.normal(0, step_size, size=n)
        new_pos = best_pos + perturb
        
        # Clamp to valid range (with small margin)
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.001, 0.999)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.001, 0.999)
        
        new_val = _total_radius(new_pos)
        
        if new_val > best_val:
            best_pos = new_pos.copy()
            best_val = new_val
            # Reduce step size slightly on success
            step_size *= 0.95
        else:
            # Increase step size on failure (restart-like)
            step_size *= 1.05
            step_size = min(step_size, 0.1)
    
    return best_pos, best_val


def _refine_with_grad(positions, time_budget):
    """Gradient-based refinement using numerical gradients."""
    n = len(positions)
    best_pos = positions.copy()
    best_val = _total_radius(best_pos)
    start_time = time.time()
    
    lr = 0.01
    while time.time() - start_time < time_budget:
        # Compute numerical gradient of total radius
        grad = np.zeros_like(best_pos)
        eps = 0.001
        for i in range(n):
            for d in range(2):
                orig = best_pos[i, d]
                best_pos[i, d] = orig + eps
                f_plus = _total_radius(best_pos)
                best_pos[i, d] = orig - eps
                f_minus = _total_radius(best_pos)
                best_pos[i, d] = orig
                grad[i, d] = (f_plus - f_minus) / (2 * eps)
        
        # Update
        new_pos = best_pos + lr * grad
        new_pos[:, 0] = np.clip(new_pos[:, 0], 0.001, 0.999)
        new_pos[:, 1] = np.clip(new_pos[:, 1], 0.001, 0.999)
        
        new_val = _total_radius(new_pos)
        if new_val > best_val:
            best_pos = new_pos
            best_val = new_val
            lr *= 0.9
        else:
            lr *= 0.7
            if lr < 0.0001:
                break
    
    return best_pos, best_val


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_val = -np.inf
    
    # Multiple restarts with different seeds
    for seed in range(20):
        if time.time() - start_time > DEADLINE_SECONDS - 20:
            break
        
        rng = np.random.default_rng(seed * 12345 + 67890)
        positions = _hexagonal_init(N, rng)
        
        # Quick local search
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining < 5:
            break
        ls_budget = min(remaining * 0.4, 8)
        positions, val = _local_search(positions, rng, ls_budget)
        
        if val > best_val:
            best_val = val
            best_pos = positions.copy()
    
    # Final refinement on best solution
    remaining = DEADLINE_SECONDS - (time.time() - start_time)
    if remaining > 5 and best_pos is not None:
        # Gradient refinement
        grad_budget = min(remaining * 0.5, 15)
        best_pos, best_val = _refine_with_grad(best_pos, grad_budget)
        
        # One more local search polish
        remaining = DEADLINE_SECONDS - (time.time() - start_time)
        if remaining > 3:
            rng = np.random.default_rng(99999)
            best_pos, best_val = _local_search(best_pos, rng, min(remaining * 0.5, 5))
    
    # Build result
    radii = _compute_radii(best_pos)
    result = np.column_stack([best_pos, radii])
    
    # Safety: ensure all circles are inside [0,1]^2
    for i in range(N):
        x, y, r = result[i]
        result[i, 0] = min(max(x, r), 1 - r)
        result[i, 1] = min(max(y, r), 1 - r)
        result[i, 2] = min(r, result[i, 0], 1 - result[i, 0],
                           result[i, 1], 1 - result[i, 1])
        result[i, 2] = max(result[i, 2], 0.001)
    
    return result
