"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _eval_packing(xy, r):
    """
    Evaluate a packing: returns (total_radius, is_valid).
    Valid means all circles inside [0,1]^2 and disjoint interiors.
    """
    n = len(xy)
    # Check walls
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
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
    return float(np.sum(r)), True


def _is_valid_config(xy, r):
    """Fast validity check for MC steps."""
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist_sq = dx * dx + dy * dy
            min_dist = r[i] + r[j]
            if dist_sq < min_dist * min_dist - 1e-12:
                return False
    return True


def _mc_hard_disk_packing(n, deadline, seed=0):
    """
    Monte Carlo simulation of hard disks.
    1. Start with small radii.
    2. Grow radii slowly.
    3. Use Metropolis moves (position perturbations) to maintain valid config.
    4. Track best sum of radii.
    """
    rng = np.random.RandomState(seed)
    xy = rng.uniform(0.1, 0.9, size=(n, 2))
    r = np.full(n, 0.01)
    
    # Ensure initial validity
    if not _is_valid_config(xy, r):
        # Retry with smaller radii or different positions
        for _ in range(10):
            xy = rng.uniform(0.2, 0.8, size=(n, 2))
            r = np.full(n, 0.005)
            if _is_valid_config(xy, r):
                break
    
    best_total = np.sum(r)
    best_xy = xy.copy()
    best_r = r.copy()
    
    # Parameters
    start_r = 0.01
    max_r_target = 0.15 # We won't grow to this, but it's a limit
    growth_rate = 0.0002 # Increase per successful step
    move_size = 0.02
    
    steps = 0
    max_steps = 200000 # Hard cap, but deadline is primary
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Try to grow radii slightly if system is stable
        # We attempt to increase all radii by a tiny amount
        dr = growth_rate
        new_r = r + dr
        
        # Check if this growth is valid with current positions
        # If not, we need to move particles to make room
        # Strategy: Attempt a move. If move succeeds, keep it. 
        # Periodically attempt a "growth" step.
        
        # Simpler strategy for MC:
        # 1. Randomly pick a particle.
        # 2. Propose new position.
        # 3. If valid, accept.
        # 4. Separately, try to increase radii.
        
        # Let's use a combined approach:
        # Every K steps, try to increase all radii by epsilon.
        # If invalid, revert radii and try to fix positions via MC moves.
        
        # Actually, a better approach for "maximize sum of radii":
        # Treat radii as part of the state.
        # Propose: Pick particle i.
        #   Option A: Move particle i.
        #   Option B: Change radius of particle i (grow/shrink).
        
        i = rng.randint(n)
        move_type = rng.random()
        
        if move_type < 0.7: # Move position
            new_xy = xy.copy()
            new_r = r.copy()
            # Gaussian move
            dx = rng.randn() * move_size
            dy = rng.randn() * move_size
            new_xy[i, 0] += dx
            new_xy[i, 1] += dy
            
            # Clip to box (with radius margin)
            ri = r[i]
            new_xy[i, 0] = np.clip(new_xy[i, 0], ri, 1.0 - ri)
            new_xy[i, 1] = np.clip(new_xy[i, 1], ri, 1.0 - ri)
            
            if _is_valid_config(new_xy, new_r):
                xy = new_xy
                # Accept move
            else:
                # Reject
                pass
                
        else: # Change radius
            new_xy = xy.copy()
            new_r = r.copy()
            # Try to grow
            if rng.random() < 0.6:
                new_r[i] += 0.005
            else:
                new_r[i] -= 0.005
            
            new_r[i] = max(0.001, new_r[i])
            
            if _is_valid_config(new_xy, new_r):
                xy = new_xy
                r = new_r
                # Accept radius change
            else:
                # Reject
                pass
        
        current_total = np.sum(r)
        if current_total > best_total:
            best_total = current_total
            best_xy = xy.copy()
            best_r = r.copy()
            
        # Adaptive move size
        if steps % 1000 == 0:
            # Reduce move size slightly to settle
            move_size *= 0.98
            
    return best_xy, best_r, best_total


def _best_fit_local_search(xy, r, deadline):
    """
    Local search: For each circle, try to increase its radius as much as possible
    while maintaining validity, or move it to a better spot.
    """
    n = len(xy)
    current_xy = xy.copy()
    current_r = r.copy()
    best_total = np.sum(current_r)
    best_xy = current_xy.copy()
    best_r = current_r.copy()
    
    # Simple iterative improvement
    for _ in range(500):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(n):
            # Try to increase radius
            original_r = current_r[i]
            # Binary search for max radius
            low = original_r
            high = min(current_xy[i, 0], 1.0 - current_xy[i, 0], 
                       current_xy[i, 1], 1.0 - current_xy[i, 1])
            
            # Check if we can increase
            # Try a few steps
            for _ in range(10):
                mid = (low + high) / 2
                test_r = current_r.copy()
                test_r[i] = mid
                if _is_valid_config(current_xy, test_r):
                    low = mid
                else:
                    high = mid
                    
            if low > original_r + 1e-6:
                current_r[i] = low
                improved = True
                if np.sum(current_r) > best_total:
                    best_total = np.sum(current_r)
                    best_xy = current_xy.copy()
                    best_r = current_r.copy()
            
            # Try moving the circle to a random nearby spot and re-maximizing radius
            # This helps escape local optima where a circle is stuck
            if rng_move := np.random.random() < 0.1:
                # Save state
                save_xy = current_xy.copy()
                save_r = current_r.copy()
                
                # Random move
                dx = np.random.randn() * 0.05
                dy = np.random.randn() * 0.05
                new_x = np.clip(current_xy[i, 0] + dx, current_r[i], 1.0 - current_r[i])
                new_y = np.clip(current_xy[i, 1] + dy, current_r[i], 1.0 - current_r[i])
                
                test_xy = current_xy.copy()
                test_xy[i, 0] = new_x
                test_xy[i, 1] = new_y
                
                if _is_valid_config(test_xy, current_r):
                    # Now try to increase radius again
                    current_xy = test_xy
                    # Repeat radius maximization for this circle
                    low = current_r[i]
                    high = min(current_xy[i, 0], 1.0 - current_xy[i, 0], 
                               current_xy[i, 1], 1.0 - current_xy[i, 1])
                    for _ in range(10):
                        mid = (low + high) / 2
                        test_r = current_r.copy()
                        test_r[i] = mid
                        if _is_valid_config(current_xy, test_r):
                            low = mid
                        else:
                            high = mid
                    if low > current_r[i]:
                        current_r[i] = low
                        improved = True
                        if np.sum(current_r) > best_total:
                            best_total = np.sum(current_r)
                            best_xy = current_xy.copy()
                            best_r = current_r.copy()
                    else:
                        # Revert if no gain
                        current_xy = save_xy
                        current_r = save_r
                else:
                    # Revert
                    current_xy = save_xy
                    current_r = save_r

        if not improved:
            break
            
    return best_xy, best_r, best_total


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=200):
    """
    Gradient-based polish using L-BFGS-B on the smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
    for i in range(n):
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((0.001, 0.5))  # r

    # Adaptive beta strategy
    betas = [50.0, 100.0]
    
    current_xy = xy.copy()
    current_r = r.copy()
    best_total, _ = _eval_packing(current_xy, current_r)

    def _smooth_objective(v, n, beta=50.0):
        x = v[0:n]
        y = v[n:2*n]
        r = v[2*n:3*n]

        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        wall_pen = np.sum(np.logaddexp(0.0, beta * wall_viol) / beta)

        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = r[i] + r[j] - dist
                total_viol += np.logaddexp(0.0, beta * viol) / beta

        return -np.sum(r) + 100.0 * (wall_pen + total_viol)

    def _smooth_grad(v, n, beta=50.0):
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        r = v[2*n:3*n].copy()

        grad = np.zeros(3 * n)

        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = r - wall_min
        sig_wall = 1.0 / (1.0 + np.exp(-beta * wall_viol))

        for i in range(n):
            s = sig_wall[i]
            if s < 1e-10:
                continue
            dists = np.array([x[i], 1.0 - x[i], y[i], 1.0 - y[i]])
            min_idx = int(np.argmin(dists))
            if min_idx == 0:
                grad[2*n + i] += 100.0 * s
                grad[i] -= 100.0 * s
            elif min_idx == 1:
                grad[2*n + i] += 100.0 * s
                grad[i] += 100.0 * s
            elif min_idx == 2:
                grad[2*n + i] += 100.0 * s
                grad[n + i] -= 100.0 * s
            else:
                grad[2*n + i] += 100.0 * s
                grad[n + i] += 100.0 * s

        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                if dist_sq < 1e-20:
                    continue
                dist = np.sqrt(dist_sq)
                viol = r[i] + r[j] - dist
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s < 1e-10:
                    continue
                factor = 100.0 * s
                grad[i] += factor * dx / dist
                grad[n + i] += factor * dy / dist
                grad[j] -= factor * dx / dist
                grad[n + j] -= factor * dy / dist
                grad[2*n + i] += factor
                grad[2*n + j] += factor

        grad[2*n:3*n] -= 1.0
        return grad

    for beta in betas:
        if time.time() > deadline - 5:
            break
            
        def obj(v, b=beta):
            if time.time() > deadline:
                return float('inf')
            return _smooth_objective(v, n, b)

        def grad(v, b=beta):
            if time.time() > deadline:
                return np.zeros(3 * n)
            return _smooth_grad(v, n, b)

        try:
            result = minimize(obj, np.concatenate([current_xy[:, 0], current_xy[:, 1], current_r]), 
                              jac=grad, method='L-BFGS-B',
                              bounds=bounds,
                              options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                       'maxfun': 5000})
        except Exception:
            continue

        v_opt = result.x
        new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
        new_r = v_opt[2*n:3*n]

        total, valid = _eval_packing(new_xy, new_r)
        if valid and total > best_total:
            best_total = total
            current_xy = new_xy
            current_r = new_r

    return current_xy, current_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple MC runs with different seeds
    # MC is stochastic, so multiple restarts help find better jammed states
    for seed in range(10):
        if time.time() > deadline - 20:
            break
        
        out_xy, out_r, total = _mc_hard_disk_packing(N, deadline - 20, seed=seed)
        if total > best_total:
            best_total = total
            best_valid = True
            best_xy = out_xy
            best_r = out_r

    # Phase 2: Local Search (Best Fit) to expand circles
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            polished_xy, polished_r, total = _best_fit_local_search(best_xy, best_r, deadline - 10)
            if total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

    # Phase 3: Gradient Polish for final squeezing
    if best_valid and best_xy is not None:
        if time.time() < deadline - 5:
            final_xy, final_r = _gradient_polish(best_xy, best_r, deadline - 5)
            total, valid = _eval_packing(final_xy, final_r)
            if valid and total > best_total:
                best_total = total
                best_xy = final_xy
                best_r = final_r

    # Fallback
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
