"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time


def _max_radius_at(pos, others_xy, others_r):
    """Max radius for a circle at pos given others and walls."""
    x, y = pos
    # Wall constraint
    r_wall = min(x, 1.0 - x, y, 1.0 - y)
    if r_wall <= 0:
        return 0.0
    if len(others_xy) == 0:
        return r_wall
    dx = others_xy[:, 0] - x
    dy = others_xy[:, 1] - y
    dists = np.sqrt(dx * dx + dy * dy)
    r_circles = dists - others_r
    return max(0.0, min(r_wall, np.min(r_circles)))


def _find_largest_empty(n_existing, positions, radii, grid=150):
    """Find the position in [0,1]^2 where a new circle can have the largest radius."""
    gx = np.linspace(0.0, 1.0, grid)
    gy = np.linspace(0.0, 1.0, grid)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    
    # Wall clearance for each grid point
    wall = np.minimum(np.minimum(GX, 1.0 - GX), np.minimum(GY, 1.0 - GY))
    
    if n_existing == 0:
        max_r = float(np.max(wall))
        idx = int(np.argmax(wall))
        return float(GX.ravel()[idx]), float(GY.ravel()[idx]), max_r
    
    # Distance to each existing circle
    # For efficiency, process in chunks if needed, but 150x150=22500 points x 26 circles is fine
    best_r = wall.copy()
    for k in range(n_existing):
        dx = GX - positions[k, 0]
        dy = GY - positions[k, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        clearance = dist - radii[k]
        best_r = np.minimum(best_r, clearance)
    
    # Clip to 0
    best_r = np.maximum(best_r, 0.0)
    max_r = float(np.max(best_r))
    idx = int(np.argmax(best_r))
    px = float(GX.ravel()[idx])
    py = float(GY.ravel()[idx])
    return px, py, max_r


def _optimize_single_circle(i, positions, radii, grid=200, fine_grid=40):
    """
    Find the best position and radius for circle i, given all others fixed.
    Returns (best_x, best_y, best_r).
    """
    n = len(radii)
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    others_xy = positions[mask]
    others_r = radii[mask]
    
    # Coarse grid
    gx = np.linspace(0.0, 1.0, grid)
    gy = np.linspace(0.0, 1.0, grid)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    wall = np.minimum(np.minimum(GX, 1.0 - GX), np.minimum(GY, 1.0 - GY))
    
    if len(others_xy) == 0:
        max_r = float(np.max(wall))
        idx = int(np.argmax(wall))
        return float(GX.ravel()[idx]), float(GY.ravel()[idx]), max_r
    
    best_r = wall.copy()
    for k in range(len(others_xy)):
        dx = GX - others_xy[k, 0]
        dy = GY - others_xy[k, 1]
        dist = np.sqrt(dx * dx + dy * dy)
        clearance = dist - others_r[k]
        best_r = np.minimum(best_r, clearance)
    
    best_r = np.maximum(best_r, 0.0)
    max_r = float(np.max(best_r))
    idx = int(np.argmax(best_r))
    bx = float(GX.ravel()[idx])
    by = float(GY.ravel()[idx])
    
    # Fine grid around the best point
    step = 1.0 / grid
    margin = 3 * step
    fgx = np.linspace(max(0, bx - margin), min(1, bx + margin), fine_grid)
    fgy = np.linspace(max(0, by - margin), min(1, by + margin), fine_grid)
    FGX, FGY = np.meshgrid(fgx, fgy, indexing='ij')
    fwall = np.minimum(np.minimum(FGX, 1.0 - FGX), np.minimum(FGY, 1.0 - FGY))
    
    fbest_r = fwall.copy()
    for k in range(len(others_xy)):
        fdx = FGX - others_xy[k, 0]
        fdy = FGY - others_xy[k, 1]
        fdist = np.sqrt(fdx * fdx + fdy * fdy)
        fclearance = fdist - others_r[k]
        fbest_r = np.minimum(fbest_r, fclearance)
    
    fbest_r = np.maximum(fbest_r, 0.0)
    fmax_r = float(np.max(fbest_r))
    fidx = int(np.argmax(fbest_r))
    fbx = float(FGX.ravel()[fidx])
    fby = float(FGY.ravel()[fidx])
    
    if fmax_r > max_r:
        return fbx, fby, fmax_r
    return bx, by, max_r


def _relax_all(positions, radii, deadline, max_sweeps=50, grid=150, fine_grid=30):
    """
    Iteratively re-optimize each circle in sequence.
    Returns (positions, radii, total).
    """
    n = len(radii)
    best_total = float(np.sum(radii))
    best_pos = positions.copy()
    best_rad = radii.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            new_x, new_y, new_r = _optimize_single_circle(
                i, positions, radii, grid=grid, fine_grid=fine_grid)
            if new_r > radii[i] + 1e-12:
                positions[i, 0] = new_x
                positions[i, 1] = new_y
                radii[i] = new_r
                improved = True
        
        total = float(np.sum(radii))
        if total > best_total + 1e-12:
            best_total = total
            best_pos = positions.copy()
            best_rad = radii.copy()
        
        if not improved:
            break
    
    return best_pos, best_rad, best_total


def _greedy_insertion(deadline, grid_coarse=120, grid_fine=200, relax_grid=150, relax_fine=30):
    """
    Sequential greedy insertion: add circles one by one at the largest empty spot,
    then do a full relaxation pass.
    """
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    n_existing = 0
    
    # Insert circles one by one
    for i in range(N):
        if time.time() > deadline:
            break
        if n_existing == 0:
            # First circle: center of square
            positions[i, 0] = 0.5
            positions[i, 1] = 0.5
            radii[i] = 0.5
        else:
            # Find largest empty spot
            px, py, pr = _find_largest_empty(
                n_existing, positions[:n_existing], radii[:n_existing], grid=grid_coarse)
            # Only place if it fits
            if pr > 0.005:
                positions[i, 0] = px
                positions[i, 1] = py
                radii[i] = pr
                n_existing += 1
            else:
                # Fallback: tiny circle at center
                positions[i, 0] = 0.5
                positions[i, 1] = 0.5
                radii[i] = 0.005
                n_existing += 1
    
    # Now do full relaxation
    sub_deadline = deadline - 5
    if time.time() < sub_deadline:
        positions, radii, total = _relax_all(
            positions, radii, sub_deadline,
            max_sweeps=80, grid=relax_grid, fine_grid=relax_fine)
    
    return positions, radii, float(np.sum(radii))


def _perturbed_greedy(deadline, seed, perturb_scale=0.1, grid_coarse=120, relax_grid=150, relax_fine=30):
    """
    Greedy insertion with random perturbation of insertion order and initial positions.
    """
    rng = np.random.RandomState(seed)
    positions = np.zeros((N, 2))
    radii = np.zeros(N)
    n_existing = 0
    
    # Randomize the order of insertion
    order = list(range(N))
    rng.shuffle(order)
    
    for idx in order:
        if time.time() > deadline:
            break
        if n_existing == 0:
            # First circle: slightly perturbed center
            positions[idx, 0] = 0.5 + rng.normal(0, 0.05)
            positions[idx, 1] = 0.5 + rng.normal(0, 0.05)
            positions[idx, 0] = np.clip(positions[idx, 0], 0.1, 0.9)
            positions[idx, 1] = np.clip(positions[idx, 1], 0.1, 0.9)
            radii[idx] = 0.3
        else:
            # Find largest empty spot with some noise
            px, py, pr = _find_largest_empty(
                n_existing, positions[:n_existing], radii[:n_existing], grid=grid_coarse)
            # Add small perturbation
            px += rng.normal(0, 0.02)
            py += rng.normal(0, 0.02)
            px = np.clip(px, 0.01, 0.99)
            py = np.clip(py, 0.01, 0.99)
            if pr > 0.005:
                positions[idx, 0] = px
                positions[idx, 1] = py
                radii[idx] = pr
                n_existing += 1
            else:
                positions[idx, 0] = 0.5
                positions[idx, 1] = 0.5
                radii[idx] = 0.005
                n_existing += 1
    
    # Full relaxation
    sub_deadline = deadline - 5
    if time.time() < sub_deadline:
        positions, radii, total = _relax_all(
            positions, radii, sub_deadline,
            max_sweeps=60, grid=relax_grid, fine_grid=relax_fine)
    
    return positions, radii, float(np.sum(radii))


def _multi_start_greedy(deadline):
    """
    Run multiple greedy insertions with different strategies and return the best.
    """
    start_time = time.time()
    best_pos = None
    best_rad = None
    best_total = 0.0
    
    # Strategy 1: Pure greedy (center-out)
    if time.time() < deadline - 30:
        pos, rad, tot = _greedy_insertion(deadline - 20, grid_coarse=120, relax_grid=150, relax_fine=30)
        if tot > best_total:
            best_total = tot
            best_pos = pos
            best_rad = rad
    
    # Strategy 2: Perturbed greedy with different seeds
    n_perturb = 4
    per_time = max(10, (deadline - time.time() - 10) / n_perturb)
    for s in range(n_perturb):
        if time.time() > deadline - 10:
            break
        sub_dl = time.time() + per_time
        pos, rad, tot = _perturbed_greedy(sub_dl, seed=42 + s, grid_coarse=100, relax_grid=140, relax_fine=25)
        if tot > best_total:
            best_total = tot
            best_pos = pos
            best_rad = rad
    
    # Final high-precision relaxation on the best result
    if best_pos is not None and time.time() < deadline - 5:
        final_pos, final_rad, final_tot = _relax_all(
            best_pos, best_rad, deadline - 2,
            max_sweeps=100, grid=200, fine_grid=40)
        if final_tot > best_total:
            best_total = final_tot
            best_pos = final_pos
            best_rad = final_rad
    
    return best_pos, best_rad, best_total


def _validate_and_fix(positions, radii):
    """Ensure all circles are valid (inside square, non-overlapping)."""
    n = len(radii)
    for i in range(n):
        # Clip radius to fit in square
        max_r = min(positions[i, 0], 1.0 - positions[i, 0],
                    positions[i, 1], 1.0 - positions[i, 1])
        radii[i] = min(radii[i], max_r)
        radii[i] = max(radii[i], 0.001)
        positions[i, 0] = np.clip(positions[i, 0], radii[i], 1.0 - radii[i])
        positions[i, 1] = np.clip(positions[i, 1], radii[i], 1.0 - radii[i])
    
    # Check for overlaps and shrink if needed
    for it in range(10):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-9:
                    # Shrink the smaller circle
                    if radii[i] < radii[j]:
                        radii[i] = max(0.001, dist / 2 - 1e-6)
                    else:
                        radii[j] = max(0.001, dist / 2 - 1e-6)
                    overlap_found = True
        if not overlap_found:
            break
    
    return positions, radii


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_pos, best_rad, best_total = _multi_start_greedy(deadline)
    
    if best_pos is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        positions = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            positions[i, 0] = (gx + 0.5) / side
            positions[i, 1] = (gy + 0.5) / side
            radii[i] = 0.45 / side
        best_pos = positions
        best_rad = radii
        best_total = float(np.sum(radii))
    
    # Validate and fix
    best_pos, best_rad = _validate_and_fix(best_pos, best_rad)
    
    # Build result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i, 0] = best_pos[i, 0]
        result[i, 1] = best_pos[i, 1]
        result[i, 2] = best_rad[i]
    
    return result
# EVOLVE-BLOCK-END
