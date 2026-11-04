"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import time


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    # Check pairwise distances
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return np.sum(r), True


def _largest_empty_circle(xy, r, n_samples=20000):
    """
    Find the center and radius of the largest empty circle in the unit square
    that does not overlap existing circles.
    """
    # Sample points in the unit square
    # Use a mix of random and grid points for better coverage
    n_grid = 100
    grid_x = np.linspace(0.01, 0.99, n_grid)
    grid_y = np.linspace(0.01, 0.99, n_grid)
    gx, gy = np.meshgrid(grid_x, grid_y, indexing='ij')
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    
    n_rand = n_samples - len(grid_pts)
    if n_rand > 0:
        rand_pts = np.random.rand(n_rand, 2) * 0.98 + 0.01
        samples = np.vstack([grid_pts, rand_pts])
    else:
        samples = grid_pts
    
    # Compute clearance for each sample point
    # Wall clearance
    wall_clear = np.minimum(np.minimum(samples[:, 0], 1.0 - samples[:, 0]),
                            np.minimum(samples[:, 1], 1.0 - samples[:, 1]))
    
    # Circle clearance
    n_circles = len(xy)
    if n_circles > 0:
        # To save memory, process in chunks
        chunk_size = 5000
        min_circle_clear = np.full(len(samples), np.inf)
        for start in range(0, len(samples), chunk_size):
            end = min(start + chunk_size, len(samples))
            pts = samples[start:end]
            dx = pts[:, 0:1] - xy[:, 0][np.newaxis, :]
            dy = pts[:, 1:2] - xy[:, 1][np.newaxis, :]
            dists = np.sqrt(dx * dx + dy * dy)
            clear = dists - r[np.newaxis, :]
            min_c = np.min(clear, axis=1)
            min_circle_clear[start:end] = min_c
        
        total_clear = np.minimum(wall_clear, min_circle_clear)
    else:
        total_clear = wall_clear.copy()
    
    # Find the best point
    idx = np.argmax(total_clear)
    best_x, best_y = samples[idx, 0], samples[idx, 1]
    best_r = total_clear[idx]
    
    # Refine around the best point
    if best_r > 0.01:
        # Local search around the best point
        best_r_local = best_r
        best_pos = np.array([best_x, best_y])
        for _ in range(50):
            # Random perturbations
            perturbations = np.random.randn(50, 2) * 0.01
            new_pts = best_pos + perturbations
            new_pts[:, 0] = np.clip(new_pts[:, 0], 0.0, 1.0)
            new_pts[:, 1] = np.clip(new_pts[:, 1], 0.0, 1.0)
            
            new_wall_clear = np.minimum(np.minimum(new_pts[:, 0], 1.0 - new_pts[:, 0]),
                                        np.minimum(new_pts[:, 1], 1.0 - new_pts[:, 1]))
            
            if n_circles > 0:
                dx = new_pts[:, 0:1] - xy[:, 0][np.newaxis, :]
                dy = new_pts[:, 1:2] - xy[:, 1][np.newaxis, :]
                dists = np.sqrt(dx * dx + dy * dy)
                clear = dists - r[np.newaxis, :]
                min_c = np.min(clear, axis=1)
                new_total_clear = np.minimum(new_wall_clear, min_c)
            else:
                new_total_clear = new_wall_clear
            
            local_idx = np.argmax(new_total_clear)
            if new_total_clear[local_idx] > best_r_local:
                best_r_local = new_total_clear[local_idx]
                best_pos = new_pts[local_idx]
            else:
                break
        
        best_x, best_y = best_pos
        best_r = best_r_local
    
    return best_x, best_y, best_r


def _force_directed_packing(xy, r, deadline, max_iter=5000, initial_temp=0.01):
    """
    Force-directed packing using repulsive forces for overlaps and
    attractive forces for a contact graph to maintain structure.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    
    # Initial contact graph (Delaunay)
    if n >= 3:
        try:
            tri = Delaunay(xy)
            edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        edges.add((simplex[i], simplex[j]))
        except Exception:
            edges = set()
    else:
        edges = set()
    
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    
    temp = initial_temp
    cooling = 0.999
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute forces
        F = np.zeros_like(xy)
        
        # Repulsive forces (all pairs)
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy[i, 0] - xy[j, 0]
                dy = xy[i, 1] - xy[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < 1e-10:
                    dx = np.random.randn() * 1e-10
                    dy = np.random.randn() * 1e-10
                    dist = np.sqrt(dx * dx + dy * dy)
                
                # Overlap force
                min_dist = r[i] + r[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    force = 10.0 * overlap / dist  # Strong repulsion
                    F[i, 0] += force * dx
                    F[i, 1] += force * dy
                    F[j, 0] -= force * dx
                    F[j, 1] -= force * dy
                else:
                    # Weak repulsion to maintain spacing
                    force = 0.1 / (dist * dist)
                    F[i, 0] += force * dx
                    F[i, 1] += force * dy
                    F[j, 0] -= force * dx
                    F[j, 1] -= force * dy
        
        # Wall forces
        for i in range(n):
            # Left wall
            if xy[i, 0] < r[i]:
                F[i, 0] += 5.0 * (r[i] - xy[i, 0])
            # Right wall
            if xy[i, 0] > 1.0 - r[i]:
                F[i, 0] -= 5.0 * (xy[i, 0] - (1.0 - r[i]))
            # Bottom wall
            if xy[i, 1] < r[i]:
                F[i, 1] += 5.0 * (r[i] - xy[i, 1])
            # Top wall
            if xy[i, 1] > 1.0 - r[i]:
                F[i, 1] -= 5.0 * (xy[i, 1] - (1.0 - r[i]))
        
        # Attractive forces (contact graph)
        for i, j in edges:
            dx = xy[j, 0] - xy[i, 0]
            dy = xy[j, 1] - xy[i, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 1e-10:
                # Spring force to maintain contact
                target_dist = r[i] + r[j]
                force = 0.5 * (dist - target_dist) / dist
                F[i, 0] += force * dx
                F[i, 1] += force * dy
                F[j, 0] -= force * dx
                F[j, 1] -= force * dy
        
        # Update positions with simulated annealing
        step_size = temp
        for i in range(n):
            # Random perturbation
            noise = np.random.randn(2) * step_size
            new_pos = xy[i] + F[i] * 0.1 + noise
            
            # Keep within bounds
            new_pos[0] = np.clip(new_pos[0], 0.0, 1.0)
            new_pos[1] = np.clip(new_pos[1], 0.0, 1.0)
            
            # Check if this move is valid or accept with probability
            # Simple: always accept if it doesn't increase overlap
            old_overlap = 0.0
            new_overlap = 0.0
            
            for j in range(n):
                if j == i:
                    continue
                old_dist = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                new_dist = np.sqrt((new_pos[0] - xy[j, 0])**2 + (new_pos[1] - xy[j, 1])**2)
                min_dist = r[i] + r[j]
                if old_dist < min_dist:
                    old_overlap += (min_dist - old_dist) ** 2
                if new_dist < min_dist:
                    new_overlap += (min_dist - new_dist) ** 2
            
            # Accept if overlap decreases or with probability exp(-delta/temp)
            delta = new_overlap - old_overlap
            if delta < 0 or np.random.rand() < np.exp(-delta / (temp + 1e-10)):
                xy[i] = new_pos
        
        # Update radii based on clearance
        for i in range(n):
            min_clear = np.inf
            # Wall clearance
            min_clear = min(min_clear, xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
            # Circle clearance
            for j in range(n):
                if j == i:
                    continue
                dist = np.sqrt((xy[i, 0] - xy[j, 0])**2 + (xy[i, 1] - xy[j, 1])**2)
                min_clear = min(min_clear, dist - r[j])
            
            # Gradually increase radius towards clearance
            r[i] = min(r[i] + 0.001, max(0.001, min_clear))
        
        # Cool down
        temp *= cooling
        
        # Evaluate and track best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_xy = xy.copy()
            best_r = r.copy()
        
        # Update contact graph every 100 iterations
        if it % 100 == 0 and n >= 3:
            try:
                tri = Delaunay(xy)
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        for j in range(i + 1, 3):
                            edges.add((simplex[i], simplex[j]))
            except Exception:
                pass
    
    return best_xy, best_r, best_total, best_valid


def _gradient_polish(xy, r, deadline):
    """
    Final gradient-based polish using L-BFGS-B with penalty terms.
    """
    n = len(xy)
    
    params = np.zeros(3 * n)
    params[0::3] = xy[:, 0]
    params[1::3] = xy[:, 1]
    params[2::3] = r
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.4))
    
    def _objective(p):
        xy_l = p[0::3].reshape(-1, 2)
        r_l = np.maximum(p[1::3], 0.001)
        
        penalty = 0.0
        # Wall violations
        for i in range(n):
            x, y, ri = xy_l[i, 0], xy_l[i, 1], r_l[i]
            penalty += max(0.0, ri - x)**2
            penalty += max(0.0, ri - (1.0 - x))**2
            penalty += max(0.0, ri - y)**2
            penalty += max(0.0, ri - (1.0 - y))**2
        
        # Overlap violations
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_l[i, 0] - xy_l[j, 0]
                dy = xy_l[i, 1] - xy_l[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r_l[i] + r_l[j]
                if dist < min_dist:
                    penalty += (min_dist - dist)**2
        
        return -np.sum(r_l) + 1000.0 * penalty
    
    best_total, best_valid = _eval_packing(xy, r)
    best_params = params.copy()
    
    try:
        result = minimize(
            _objective,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        
        opt_params = result.x
        opt_xy = opt_params[0::3].reshape(-1, 2)
        opt_r = np.maximum(opt_params[1::3], 0.001)
        
        total, valid = _eval_packing(opt_xy, opt_r)
        if valid and total > best_total:
            best_total = total
            best_valid = True
            best_params = opt_params.copy()
            
    except Exception:
        pass
    
    final_xy = best_params[0::3].reshape(-1, 2)
    final_r = np.maximum(best_params[1::3], 0.001)
    
    return final_xy, final_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False
    
    # Multi-start with diverse initializations
    n_starts = 10
    for start in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Different initialization strategies
        if start == 0:
            # Grid initialization
            side = 5
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                gx = i % side
                gy = i // side
                xy[i, 0] = (gx + 0.5) / side
                xy[i, 1] = (gy + 0.5) / side
                r[i] = 0.5 / side - 0.001
        elif start == 1:
            # Random initialization with largest empty circle placement
            xy = np.zeros((N, 2))
            r = np.zeros(N)
            for i in range(N):
                if i == 0:
                    xy[i, 0] = 0.5
                    xy[i, 1] = 0.5
                    r[i] = 0.1
                else:
                    bx, by, br = _largest_empty_circle(xy[:i], r[:i], n_samples=5000)
                    xy[i, 0] = bx
                    xy[i, 1] = by
                    r[i] = br
        else:
            # Random initialization
            xy = np.random.rand(N, 2) * 0.8 + 0.1
            r = np.random.rand(N) * 0.05 + 0.02
        
        # Force-directed packing
        if time.time() < deadline - 20.0:
            out_xy, out_r, total, valid = _force_directed_packing(
                xy, r, deadline - 20.0, max_iter=3000, initial_temp=0.01
            )
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = out_xy
                best_r = out_r
        
        # Gradient polish if we have a valid result
        if best_valid and time.time() < deadline - 10.0:
            g_xy, g_r, g_total, g_valid = _gradient_polish(best_xy, best_r, deadline - 10.0)
            if g_valid and g_total > best_total:
                best_total = g_total
                best_valid = True
                best_xy = g_xy
                best_r = g_r
    
    # If no valid result found, fall back to a simple grid
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
        return result
    
    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        # Ensure circle is inside unit square
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)
    
    # Final check
    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side
    
    return result
# EVOLVE-BLOCK-END
