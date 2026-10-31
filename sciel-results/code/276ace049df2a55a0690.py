"""Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Geometric Contact Graph / Linear Programming.
1. Initialize with diverse layouts (hexagonal, grid, random).
2. Iteratively optimize positions by maximizing the minimum clearance 
   (distance to boundary or other circles minus current radius).
3. For each position configuration, solve an exact Linear Program (LP) 
   to find the maximum sum of radii subject to non-overlap and boundary constraints.
4. This separates the non-linear positional problem from the linear radial problem,
   allowing for tighter packing than penalty methods."""

import numpy as np
import time
from scipy.optimize import linprog

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dx = pos[:, 0, np.newaxis] - pos[np.newaxis, :, 0]
    dy = pos[:, 1, np.newaxis] - pos[np.newaxis, :, 1]
    dist_sq = dx * dx + dy * dy
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist_sq[mask] < (min_dist[mask] - tol) ** 2):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(50):
        valid = True
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        # Overlap check
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    k = dist / (sum_r + 1e-12) * 0.9999
                    if k < 1.0:
                        radii[i] = max(radii[i] * k, 1e-5)
                        radii[j] = max(radii[j] * k, 1e-5)
                        valid = False
        if valid:
            break
    return pos, radii

def _solve_lp_radii(pos):
    """
    Solve LP to maximize sum of radii given positions.
    Variables: r_0, ..., r_{N-1}
    Objective: max sum(r_i) -> min -sum(r_i)
    Constraints:
    1. r_i + r_j <= d_ij for all i < j
    2. r_i <= x_i, 1-x_i, y_i, 1-y_i for all i
    3. r_i >= 0
    """
    c = -np.ones(N)
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints: r_i + r_j <= d_ij
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            row = np.zeros(N)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(d)
            
    # Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # We can add these as individual constraints or just bound r_i.
    # Since r_i appears in multiple pairwise constraints, we can't just bound it simply 
    # unless we take the min. But linprog handles bounds.
    # However, the pairwise constraints already limit the sum.
    # Let's add explicit boundary constraints to be safe, or just use bounds.
    # Using bounds is more efficient.
    
    bounds = []
    for i in range(N):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        bounds.append((0.0, max_r))
        
    if not A_ub:
        # No pairs? N=1
        res = linprog(c, A_ub=None, b_ub=None, bounds=bounds, method='highs')
    else:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
    if res.success:
        return res.x
    else:
        # Fallback: return zeros or small values
        return np.zeros(N)

def _optimize_positions(pos, radii, max_iters=1000, step_size=0.01):
    """
    Iteratively update positions to maximize clearance.
    Clearance for circle i = min( dist(i, j) - (r_i + r_j) for all j, dist(i, boundary) - r_i )
    We want to maximize the minimum clearance.
    This is a non-convex problem, so we use a heuristic:
    Move each circle in the direction that increases its local clearance.
    """
    pos = pos.copy()
    radii = radii.copy()
    
    for it in range(max_iters):
        # Calculate forces based on clearance gradients
        # Approximate gradient of min-clearance
        grad_pos = np.zeros_like(pos)
        
        for i in range(N):
            # Boundary forces
            # If close to boundary, push away
            x, y = pos[i]
            r = radii[i]
            
            # Left
            if x - r < 0.05:
                grad_pos[i, 0] += 1.0
            # Right
            if 1.0 - x - r < 0.05:
                grad_pos[i, 0] -= 1.0
            # Bottom
            if y - r < 0.05:
                grad_pos[i, 1] += 1.0
            # Top
            if 1.0 - y - r < 0.05:
                grad_pos[i, 1] -= 1.0
            
            # Neighbor forces
            # Push away from neighbors that are close
            for j in range(N):
                if i == j: continue
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                
                # If dist < sum_r + margin, push away
                margin = 0.05
                if dist < sum_r + margin:
                    # Force magnitude proportional to violation
                    if dist > 1e-6:
                        factor = (sum_r + margin - dist) / dist
                        grad_pos[i, 0] += factor * dx
                        grad_pos[i, 1] += factor * dy
        
        # Normalize and update
        norm = np.linalg.norm(grad_pos, axis=1, keepdims=True)
        norm[norm == 0] = 1
        grad_pos /= norm
        
        pos += step_size * grad_pos
        
        # Project back to valid region (roughly)
        # Ensure centers are at least r away from boundaries
        for i in range(N):
            r = radii[i]
            pos[i, 0] = np.clip(pos[i, 0], r, 1.0 - r)
            pos[i, 1] = np.clip(pos[i, 1], r, 1.0 - r)
            
        # Decay step size
        step_size *= 0.995
        if step_size < 1e-5:
            break
            
    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.10
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_grid():
    """Grid initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid = 25, plus 1
    side = 5
    step = 1.0 / side
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= N: break
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            pos[k] = [x, y]
            radii[k] = step / 2 * 0.9
            k += 1
    if k < N:
        pos[k] = [0.5, 0.5]
        radii[k] = 0.01
    return pos, radii

def _init_random(seed=0):
    """Random initialization."""
    rng = np.random.RandomState(seed)
    pos = rng.uniform(0.1, 0.9, (N, 2))
    radii = rng.uniform(0.05, 0.1, N)
    # Simple overlap resolution
    for _ in range(100):
        improved = False
        for i in range(N):
            for j in range(i+1, N):
                dx = pos[i,0] - pos[j,0]
                dy = pos[i,1] - pos[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j]:
                    # Shrink both
                    k = dist / (radii[i] + radii[j] + 1e-6) * 0.99
                    radii[i] *= k
                    radii[j] *= k
                    improved = True
        if not improved: break
    return pos, radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_pos = None
    best_radii = None
    
    # List of initializations
    inits = [
        ("hex", _init_hexagonal),
        ("grid", _init_grid),
        ("rand1", lambda: _init_random(1)),
        ("rand2", lambda: _init_random(2)),
        ("rand3", lambda: _init_random(3)),
    ]
    
    for name, init_fn in inits:
        if time.time() > deadline:
            break
            
        pos, radii = init_fn()
        
        # Phase 1: Optimize positions using clearance heuristic
        # Run multiple iterations of position optimization
        for _ in range(3):
            pos, radii = _optimize_positions(pos, radii, max_iters=500)
            
            # Phase 2: Solve LP for radii
            radii_lp = _solve_lp_radii(pos)
            if np.any(radii_lp > 0):
                radii = radii_lp
                
            # Post-process to ensure validity
            pos_pp, radii_pp = _post_process(pos, radii)
            
            if _is_valid(pos_pp, radii_pp):
                score = np.sum(radii_pp)
                if score > best_score:
                    best_score = score
                    best_pos = pos_pp.copy()
                    best_radii = radii_pp.copy()
                    
        # If time permits, do a more refined local search on the best so far
        if time.time() < deadline - 30 and best_pos is not None:
            # Perturb best and re-optimize
            rng = np.random.RandomState(int(time.time()) % 1000)
            for _ in range(2):
                if time.time() > deadline - 10:
                    break
                pos_p = best_pos + rng.normal(0, 0.02, (N, 2))
                rad_p = best_radii * (1 + rng.normal(0, 0.1, N))
                pos_p, rad_p = _post_process(pos_p, rad_p)
                
                pos_o, rad_o = _optimize_positions(pos_p, rad_p, max_iters=300)
                rad_lp = _solve_lp_radii(pos_o)
                if np.any(rad_lp > 0):
                    rad_o = rad_lp
                
                pos_pp, rad_pp = _post_process(pos_o, rad_o)
                if _is_valid(pos_pp, rad_pp):
                    score = np.sum(rad_pp)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_pp.copy()
                        best_radii = rad_pp.copy()

    # Final Fallback
    if best_pos is None or not _is_valid(best_pos, best_radii):
        pos, radii = _init_hexagonal()
        pos, radii = _post_process(pos, radii)
        best_pos, best_radii = pos, radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
