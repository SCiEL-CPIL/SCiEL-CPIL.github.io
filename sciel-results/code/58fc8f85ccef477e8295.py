"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 26

# EVOLVE-BLOCK-START
from scipy.optimize import linprog
from scipy.spatial import cKDTree

def _pairwise_constraints(circles):
    """Build constraint matrix and vector for pairwise non-overlap (in terms of radii).
    circles: (n, 3) array of (x, y, r)
    Returns A_eq, b_eq such that A_eq @ r <= b_eq for all pairs.
    We use the constraint: r_i + r_j <= dist_ij
    """
    n = len(circles)
    pos = circles[:, :2]
    # Compute all pairwise distances
    diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    A_rows = []
    b_vals = []
    for i, j in zip(iu[0], iu[1]):
        d = dists[i, j]
        if d < 1e-12:
            continue
        # r_i + r_j <= d
        row = np.zeros(n)
        row[i] = 1.0
        row[j] = 1.0
        A_rows.append(row)
        b_vals.append(d)
    if not A_rows:
        return None, None
    A = np.array(A_rows)
    b = np.array(b_vals)
    return A, b


def _boundary_constraints(n, pos):
    """Constraints from boundary: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    This is r_i <= b_i for each i.
    """
    A = np.eye(n)
    b = np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), np.minimum(pos[:, 1], 1 - pos[:, 1]))
    return A, b


def _solve_radii_lp(pos):
    """Given positions, solve LP to maximize sum of radii subject to non-overlap and boundary.
    Returns optimal radii or None if infeasible.
    """
    n = len(pos)
    # Pairwise constraints
    A_pair, b_pair = _pairwise_constraints(np.column_stack([pos, np.zeros(n)]))
    # Boundary constraints
    A_bound, b_bound = _boundary_constraints(n, pos)
    
    # Combine constraints
    if A_pair is not None:
        A = np.vstack([A_pair, A_bound])
        b = np.concatenate([b_pair, b_bound])
    else:
        A = A_bound
        b = b_bound
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Bounds: 0 <= r_i
    bounds = [(0, None)] * n
    
    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    if res.status == 0:
        return res.x
    return None


def _gradient_ascent(circles, pos, radii, steps=500, lr=0.001, temperature=1.0):
    """
    Gradient ascent on positions using a smooth surrogate for the objective.
    Objective: sum(r_i) where r_i is determined by LP given positions.
    We approximate the gradient by perturbing positions and re-solving LP.
    Actually, let's use a simpler approach: use the current radii as fixed and 
    optimize positions to maximize minimum distance ratio.
    
    Better approach: use a penalty-based method where we define a smooth objective
    that approximates the LP solution.
    
    Let's use a different strategy: coordinate ascent with LP radius updates.
    """
    n = len(circles)
    pos = pos.copy()
    r = radii.copy()
    
    for step in range(steps):
        # Update positions via gradient descent on a smooth objective
        # Objective: maximize sum(r_i) where r_i = min(boundary_i, min_j (d_ij - r_j))
        # This is complex. Let's use a simpler heuristic: 
        # For each circle, try to move it to increase the minimum constraint violation.
        
        # Compute current constraints
        diffs = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs ** 2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # For each i, the limiting constraint is min over j of (dists[i,j] - r[j]) and boundary
        # Let's do a simple local search: for each circle, try moving in random directions
        # and keep the move if it allows larger radii via LP.
        
        improved = False
        for i in range(n):
            best_pos = pos[i].copy()
            best_r_sum = np.sum(r)
            
            # Try several random perturbations
            for _ in range(20):
                trial_pos = pos.copy()
                delta = np.random.randn(2) * 0.01
                trial_pos[i] = np.clip(pos[i] + delta, 0, 1)
                
                # Solve LP for this new position
                trial_r = _solve_radii_lp(trial_pos)
                if trial_r is not None and np.sum(trial_r) > best_r_sum + 1e-6:
                    best_pos = trial_pos[i].copy()
                    best_r_sum = np.sum(trial_r)
                    improved = True
            
            pos[i] = best_pos
        
        # Update radii
        r = _solve_radii_lp(pos)
        if r is None:
            break
    
    return pos, r


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    # Initial positions: random
    pos = np.random.rand(N, 2)
    
    # Initial radii: small
    r = np.full(N, 0.01)
    circles = np.column_stack([pos, r])
    
    best_pos = pos.copy()
    best_r = r.copy()
    best_sum = np.sum(best_r)
    
    # Phase 1: Iterative local improvement
    for iteration in range(500):
        if time.time() > deadline:
            break
        
        # Solve LP for current positions
        r = _solve_radii_lp(pos)
        if r is None:
            # Perturb positions to fix infeasibility
            pos += np.random.randn(N, 2) * 0.005
            pos = np.clip(pos, 0, 1)
            continue
        
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_r = r.copy()
        
        # Local search: try moving each circle
        improved = False
        for i in range(N):
            if time.time() > deadline:
                break
            
            best_local_pos = pos[i].copy()
            best_local_sum = current_sum
            
            # Try several perturbations
            for attempt in range(30):
                trial_pos = pos.copy()
                # Random direction
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction) + 1e-12
                # Try different step sizes
                for step_size in [0.005, 0.01, 0.02, 0.002]:
                    trial_pos[i] = np.clip(pos[i] + direction * step_size, 0, 1)
                    trial_r = _solve_radii_lp(trial_pos)
                    if trial_r is not None:
                        trial_sum = np.sum(trial_r)
                        if trial_sum > best_local_sum + 1e-7:
                            best_local_sum = trial_sum
                            best_local_pos = trial_pos[i].copy()
                            improved = True
            
            pos[i] = best_local_pos
        
        # If no improvement, try a larger perturbation
        if not improved:
            # Random restart of some positions
            for _ in range(5):
                idx = np.random.randint(0, N)
                pos[idx] = np.random.rand(2)
    
    # Final: use best found
    pos = best_pos
    r = best_r
    
    # Final polish: try more iterations with smaller steps
    for iteration in range(200):
        if time.time() > deadline:
            break
        
        current_sum = np.sum(r)
        
        for i in range(N):
            if time.time() > deadline:
                break
            
            best_local_pos = pos[i].copy()
            best_local_sum = current_sum
            
            for attempt in range(50):
                trial_pos = pos.copy()
                direction = np.random.randn(2)
                direction /= np.linalg.norm(direction) + 1e-12
                for step_size in [0.001, 0.002, 0.005]:
                    trial_pos[i] = np.clip(pos[i] + direction * step_size, 0, 1)
                    trial_r = _solve_radii_lp(trial_pos)
                    if trial_r is not None:
                        trial_sum = np.sum(trial_r)
                        if trial_sum > best_local_sum + 1e-8:
                            best_local_sum = trial_sum
                            best_local_pos = trial_pos[i].copy()
            
            pos[i] = best_local_pos
        
        r = _solve_radii_lp(pos)
        if r is None:
            break
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = pos.copy()
            best_r = r.copy()
    
    pos = best_pos
    r = best_r
    
    # Ensure validity: clip and verify
    circles = np.column_stack([pos, r])
    
    # Verify and fix any issues
    for i in range(N):
        x, y, ri = circles[i]
        if ri < 0:
            ri = 0
        x = np.clip(x, ri, 1 - ri) if ri > 0 else np.clip(x, 0, 1)
        y = np.clip(y, ri, 1 - ri) if ri > 0 else np.clip(y, 0, 1)
        circles[i] = [x, y, ri]
    
    # Final check for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            min_dist = circles[i, 2] + circles[j, 2]
            if dist < min_dist - 1e-9:
                # Shrink radii slightly
                scale = dist / (min_dist + 1e-12)
                circles[i, 2] *= scale
                circles[j, 2] *= scale
    
    return circles
# EVOLVE-BLOCK-END
