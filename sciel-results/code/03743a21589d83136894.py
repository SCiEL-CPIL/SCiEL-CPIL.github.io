"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import scipy.optimize as opt

N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt


def _make_initial_positions(n, rng):
    """Create initial circle positions with some structure."""
    positions = np.zeros((n, 3))
    
    # Start with a grid-like arrangement
    side = int(np.ceil(np.sqrt(n)))
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n:
                break
            positions[idx, 0] = (i + 0.5) / side
            positions[idx, 1] = (j + 0.5) / side
            positions[idx, 2] = 0.5 / side * 0.95
            idx += 1
    
    # Add small random perturbations
    positions[:, :2] += rng.normal(0, 0.01, size=(n, 2))
    # Clamp positions to stay in bounds
    positions[:, 0] = np.clip(positions[:, 0], 0.01, 0.99)
    positions[:, 1] = np.clip(positions[:, 1], 0.01, 0.99)
    positions[:, 2] = np.clip(positions[:, 2], 0.01, 0.4)
    
    return positions


def _compute_objective_and_penalty(pos, n):
    """
    Compute the negative sum of radii (to minimize) plus penalties for violations.
    pos: (n, 3) array of (x, y, r)
    Returns: (objective_value, penalty_value, is_valid)
    """
    x = pos[:, 0]
    y = pos[:, 1]
    r = pos[:, 2]
    
    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalties
    penalty = 0.0
    is_valid = True
    
    # Wall constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    wall_viol = np.maximum(0, r - x) + np.maximum(0, r - (1 - x)) + \
                np.maximum(0, r - y) + np.maximum(0, r - (1 - y))
    penalty += 1000 * np.sum(wall_viol)
    if np.any(wall_viol > 1e-6):
        is_valid = False
    
    # Pairwise non-overlap: dist(i,j) >= r_i + r_j
    # Vectorized computation
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            viol = np.maximum(0, min_dist - dist)
            penalty += 1000 * viol
            if viol > 1e-6:
                is_valid = False
    
    # Radius positivity
    neg_r = np.maximum(0, -r)
    penalty += 1000 * np.sum(neg_r)
    if np.any(neg_r > 1e-6):
        is_valid = False
    
    return obj, penalty, is_valid


def _objective_wrapper(pos, n):
    """Wrapper for scipy minimize."""
    obj, pen, _ = _compute_objective_and_penalty(pos, n)
    return obj + pen


def _gradient_wrapper(pos, n):
    """Numerical gradient for better optimization."""
    eps = 1e-7
    grad = np.zeros_like(pos)
    base_val, base_pen, _ = _compute_objective_and_penalty(pos, n)
    base = base_val + base_pen
    
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.copy()
            pos_plus[i, j] += eps
            val_plus, pen_plus, _ = _compute_objective_and_penalty(pos_plus, n)
            grad[i, j] = (val_plus + pen_plus - base) / eps
    
    return grad


def _refine_with_gradient_descent(pos, n, max_iters=500, lr=0.001, deadline=None):
    """Simple gradient descent refinement."""
    pos = pos.copy()
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient_wrapper(pos, n)
        pos -= lr * grad
        
        # Project back to feasible region approximately
        pos[:, 0] = np.clip(pos[:, 0], pos[:, 2], 1 - pos[:, 2])
        pos[:, 1] = np.clip(pos[:, 1], pos[:, 2], 1 - pos[:, 2])
        pos[:, 2] = np.maximum(pos[:, 2], 0.001)
    
    return pos


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_score = -np.inf
    
    # Multiple restarts with different initial configurations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if restart == 0:
            pos = _make_initial_positions(N, rng)
        elif restart == 1:
            # More spread out
            pos = _make_initial_positions(N, rng)
            pos[:, :2] *= 0.8
            pos[:, 2] *= 0.8
        elif restart == 2:
            # Random uniform
            pos = np.zeros((N, 3))
            pos[:, 0] = rng.uniform(0.1, 0.9, N)
            pos[:, 1] = rng.uniform(0.1, 0.9, N)
            pos[:, 2] = rng.uniform(0.02, 0.15, N)
        elif restart == 3:
            # Hexagonal-ish pattern
            side = 6
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    offset = 0.5 / side if i % 2 else 0
                    pos[idx, 0] = (j + offset + 0.5) / side
                    pos[idx, 1] = (i + 0.5) / side
                    pos[idx, 2] = 0.5 / side * 0.9
                    idx += 1
        else:
            # Perturbed grid
            pos = _make_initial_positions(N, rng)
            pos[:, :2] += rng.normal(0, 0.03, size=(N, 2))
            pos[:, 0] = np.clip(pos[:, 0], 0.01, 0.99)
            pos[:, 1] = np.clip(pos[:, 1], 0.01, 0.99)
        
        # Optimize using L-BFGS-B
        try:
            result = opt.minimize(
                fun=_objective_wrapper,
                x0=pos,
                args=(N,),
                method='L-BFGS-B',
                jac=_gradient_wrapper,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            pos_opt = result.x
        except Exception:
            pos_opt = pos
        
        # Evaluate
        obj, pen, is_valid = _compute_objective_and_penalty(pos_opt, N)
        score = -obj  # sum of radii
        
        # If invalid, try to repair
        if not is_valid:
            # Try gradient descent to fix
            pos_repaired = _refine_with_gradient_descent(pos_opt, N, max_iters=200, deadline=deadline)
            obj_r, pen_r, is_valid_r = _compute_objective_and_penalty(pos_repaired, N)
            score_r = -obj_r
            if is_valid_r or score_r > score:
                pos_opt = pos_repaired
                score = score_r
                obj, pen, is_valid = obj_r, pen_r, is_valid_r
        
        if score > best_score:
            best_score = score
            best_pos = pos_opt.copy()
    
    # Final polish with tighter tolerance
    if best_pos is not None and time.time() < deadline - 10:
        try:
            result = opt.minimize(
                fun=_objective_wrapper,
                x0=best_pos,
                args=(N,),
                method='L-BFGS-B',
                jac=_gradient_wrapper,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
            )
            pos_polished = result.x
            obj_p, pen_p, is_valid_p = _compute_objective_and_penalty(pos_polished, N)
            score_p = -obj_p
            if is_valid_p and score_p > best_score:
                best_pos = pos_polished
                best_score = score_p
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_pos = np.array(cells[:N])
    
    # Final validation and cleanup
    x = best_pos[:, 0]
    y = best_pos[:, 1]
    r = best_pos[:, 2]
    
    # Ensure strict feasibility with small margin
    margin = 1e-6
    r = np.minimum(r, x - margin)
    r = np.minimum(r, 1 - x - margin)
    r = np.minimum(r, y - margin)
    r = np.minimum(r, 1 - y - margin)
    r = np.maximum(r, 0.001)
    
    # Fix any remaining overlaps by shrinking radii
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r_sum = dist - margin
            if r[i] + r[j] > max_r_sum:
                # Shrink both proportionally
                total_r = r[i] + r[j]
                if total_r > 0:
                    scale = max_r_sum / total_r
                    r[i] *= scale
                    r[j] *= scale
    
    # Final clamp
    r = np.maximum(r, 0.001)
    r = np.minimum(r, np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y)) - margin)
    r = np.maximum(r, 0.001)
    
    return np.column_stack([x, y, r])
# EVOLVE-BLOCK-END
