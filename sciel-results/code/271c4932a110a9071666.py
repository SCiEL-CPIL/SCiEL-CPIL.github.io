"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26
import numpy as np
from scipy.optimize import minimize
import time


def _eval_packing(x):
    """Evaluate packing quality: sum of radii if valid, else -inf."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    # Boundary check
    if np.any(rs < 0) or np.any(xs < -1e-9) or np.any(xs > 1 + 1e-9) or np.any(ys < -1e-9) or np.any(ys > 1 + 1e-9):
        return -np.inf
    if np.any(rs > xs + 1e-9) or np.any(rs > 1 - xs + 1e-9) or np.any(rs > ys + 1e-9) or np.any(rs > 1 - ys + 1e-9):
        return -np.inf
    # Non-overlap check
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            if d2 < (rs[i] + rs[j])**2 - 1e-9:
                return -np.inf
    return np.sum(rs)


def _max_radius_for_circle(x, y, others_xy, others_r):
    """Max radius for a circle at (x,y) given others."""
    wall_clear = min(x, 1.0 - x, y, 1.0 - y)
    if wall_clear <= 0:
        return 0.0
    max_r = wall_clear
    for k in range(len(others_r)):
        ox, oy = others_xy[k, 0], others_xy[k, 1]
        d = np.sqrt((x - ox)**2 + (y - oy)**2)
        available = d - others_r[k]
        if available < max_r:
            max_r = available
    return max(0.0, max_r)


def _best_fit_local_search(pos, rng, deadline, n_passes=50):
    """Iteratively place each circle at its best position/radius."""
    xs = pos[:, 0].copy()
    ys = pos[:, 1].copy()
    rs = pos[:, 2].copy()
    best_val = _eval_packing(pos.flatten())
    if best_val == -np.inf:
        # Fix feasibility first
        for i in range(N):
            rs[i] = min(rs[i], xs[i], 1 - xs[i], ys[i], 1 - ys[i], 0.49)
        best_val = _eval_packing(pos.flatten())
    
    for p in range(n_passes):
        if time.time() > deadline:
            break
        improved = False
        order = rng.permutation(N)
        for i in order:
            if time.time() > deadline:
                break
            # Try to expand radius at current position
            others_xy = np.vstack([xs[np.arange(N) != i], ys[np.arange(N) != i]]).T
            others_r = rs[np.arange(N) != i]
            max_r_here = _max_radius_for_circle(xs[i], ys[i], others_xy, others_r)
            if max_r_here > rs[i] + 1e-8:
                rs[i] = max_r_here
                improved = True
            
            # Try moving to better positions: sample nearby points
            best_r = rs[i]
            best_x, best_y = xs[i], ys[i]
            n_samples = 20
            for _ in range(n_samples):
                # Random offset in a local neighborhood
                dx = rng.uniform(-0.08, 0.08)
                dy = rng.uniform(-0.08, 0.08)
                nx = np.clip(xs[i] + dx, 0.01, 0.99)
                ny = np.clip(ys[i] + dy, 0.01, 0.99)
                others_xy = np.vstack([xs[np.arange(N) != i], ys[np.arange(N) != i]]).T
                others_r = rs[np.arange(N) != i]
                max_r = _max_radius_for_circle(nx, ny, others_xy, others_r)
                if max_r > best_r + 1e-6:
                    best_r = max_r
                    best_x, best_y = nx, ny
            # Also try corner-anchored and edge-anchored positions
            for cx, cy in [(0.05, 0.05), (0.95, 0.05), (0.05, 0.95), (0.95, 0.95),
                           (0.5, 0.05), (0.5, 0.95), (0.05, 0.5), (0.95, 0.5),
                           (0.5, 0.5)]:
                others_xy = np.vstack([xs[np.arange(N) != i], ys[np.arange(N) != i]]).T
                others_r = rs[np.arange(N) != i]
                max_r = _max_radius_for_circle(cx, cy, others_xy, others_r)
                if max_r > best_r + 1e-6:
                    best_r = max_r
                    best_x, best_y = cx, cy
            
            if best_r > rs[i] + 1e-6:
                xs[i], ys[i], rs[i] = best_x, best_y, best_r
                improved = True
        
        # Update pos
        pos[:, 0] = xs
        pos[:, 1] = ys
        pos[:, 2] = rs
        cur_val = _eval_packing(pos.flatten())
        if cur_val > best_val:
            best_val = cur_val
        if not improved and p > 5:
            break
    
    return pos, best_val


def _smooth_penalty_objective(x, penalty_weight=100.0, soft_beta=50.0):
    """Smooth penalty objective: minimize -sum(r) + penalty for violations."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    
    # Base objective
    obj = -np.sum(rs)
    
    # Boundary violations (soft)
    for i in range(N):
        v = np.maximum(0, rs[i] - xs[i]) + np.maximum(0, rs[i] - (1 - xs[i])) + \
            np.maximum(0, rs[i] - ys[i]) + np.maximum(0, rs[i] - (1 - ys[i]))
        obj += penalty_weight * v
    
    # Overlap violations (soft using log-sum-exp style)
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            d = np.sqrt(max(d2, 1e-12))
            overlap = (rs[i] + rs[j]) - d
            if overlap > 0:
                obj += penalty_weight * overlap
    
    return obj


def _smooth_penalty_grad(x, penalty_weight=100.0):
    """Gradient of smooth penalty objective."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    gx = np.zeros(3 * N)
    gy = np.zeros(3 * N)
    gr = np.zeros(3 * N)
    
    # Base gradient: d(-sum(r))/dr_i = -1
    gr[:] = -1.0
    
    # Boundary gradients
    for i in range(N):
        if rs[i] > xs[i]:
            gr[3*i+2] += penalty_weight
            gx[3*i] -= penalty_weight
        if rs[i] > 1 - xs[i]:
            gr[3*i+2] += penalty_weight
            gx[3*i] += penalty_weight
        if rs[i] > ys[i]:
            gr[3*i+2] += penalty_weight
            gy[3*i+1] -= penalty_weight
        if rs[i] > 1 - ys[i]:
            gr[3*i+2] += penalty_weight
            gy[3*i+1] += penalty_weight
    
    # Overlap gradients
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx*dx + dy*dy + 1e-12)
            overlap = (rs[i] + rs[j]) - d
            if overlap > 0:
                # Gradient w.r.t. positions
                gx[3*i] += penalty_weight * dx / d
                gy[3*i+1] += penalty_weight * dy / d
                gx[3*j] -= penalty_weight * dx / d
                gy[3*j+1] -= penalty_weight * dy / d
                # Gradient w.r.t. radii
                gr[3*i+2] += penalty_weight
                gr[3*j+2] += penalty_weight
    
    return np.concatenate([gx, gy, gr]).reshape(3*N)


def _gradient_polish(x, deadline, penalty_weight=100.0):
    """L-BFGS-B polish with smooth penalty."""
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    def obj(x):
        return _smooth_penalty_objective(x, penalty_weight)
    
    def grad(x):
        return _smooth_penalty_grad(x, penalty_weight)
    
    try:
        res = minimize(obj, x, jac=grad, method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if _is_valid(res.x):
            return res.x
    except Exception:
        pass
    return x


def _is_valid(x, tol=1e-7):
    """Check if configuration is valid."""
    pos = x.reshape(N, 3)
    xs, ys, rs = pos[:, 0], pos[:, 1], pos[:, 2]
    if np.any(rs < 0):
        return False
    if np.any(xs < -tol) or np.any(xs > 1 + tol) or np.any(ys < -tol) or np.any(ys > 1 + tol):
        return False
    if np.any(rs > xs + tol) or np.any(rs > 1 - xs + tol) or np.any(rs > ys + tol) or np.any(rs > 1 - ys + tol):
        return False
    for i in range(N):
        for j in range(i + 1, N):
            d2 = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
            if d2 < (rs[i] + rs[j])**2 - tol:
                return False
    return True


def _sum_radii(x):
    return np.sum(x.reshape(N, 3)[:, 2])


def _initial_grid_config(rng):
    """Generate a perturbed grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    pos = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        pos[i, 0] = (col + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 1] = (row + 0.5) / side + rng.uniform(-0.02, 0.02)
        pos[i, 2] = 0.5 / side - 0.01
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos.flatten()


def _initial_ring_config(rng):
    """Generate a ring-based initial configuration."""
    pos = np.zeros((N, 3))
    pos[0] = [0.5, 0.5, 0.12]
    idx = 1
    for k in range(6):
        angle = 2 * np.pi * k / 6
        pos[idx, 0] = 0.5 + 0.28 * np.cos(angle)
        pos[idx, 1] = 0.5 + 0.28 * np.sin(angle)
        pos[idx, 2] = 0.07
        idx += 1
    for k in range(12):
        angle = 2 * np.pi * k / 12 + np.pi / 12
        pos[idx, 0] = 0.5 + 0.40 * np.cos(angle)
        pos[idx, 1] = 0.5 + 0.40 * np.sin(angle)
        pos[idx, 2] = 0.04
        idx += 1
    corners = [(0.08, 0.08), (0.92, 0.08), (0.08, 0.92), (0.92, 0.92),
               (0.5, 0.06), (0.5, 0.94), (0.06, 0.5)]
    for k in range(7):
        pos[idx, 0] = corners[k][0]
        pos[idx, 1] = corners[k][1]
        pos[idx, 2] = 0.03
        idx += 1
    for i in range(N):
        pos[i, 0] = np.clip(pos[i, 0], pos[i, 2], 1 - pos[i, 2])
        pos[i, 1] = np.clip(pos[i, 1], pos[i, 2], 1 - pos[i, 2])
    return pos.flatten()


def _sa_refine(x, rng, deadline):
    """Simple simulated annealing refinement in-place."""
    pos = x.reshape(N, 3)
    current_val = _sum_radii(x)
    if current_val == -np.inf or not _is_valid(x):
        # Fix feasibility
        for i in range(N):
            pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1], 0.49)
        current_val = _sum_radii(pos.flatten())
    T = 0.01
    step = 0.01
    n_iter = 1500
    for it in range(n_iter):
        if time.time() > deadline:
            break
        i = rng.integers(N)
        dim = rng.integers(3)
        old = pos[i, dim]
        pos[i, dim] += rng.normal(0, step)
        if dim == 0:
            pos[i, 0] = np.clip(pos[i, 0], 0, 1)
        elif dim == 1:
            pos[i, 1] = np.clip(pos[i, 1], 0, 1)
        else:
            pos[i, 2] = np.clip(pos[i, 2], 0, 0.5)
        
        if _is_valid(pos.flatten()):
            new_val = _sum_radii(pos.flatten())
            delta = new_val - current_val
            if delta > 0 or (T > 1e-6 and rng.random() < np.exp(delta / T)):
                current_val = new_val
            else:
                pos[i, dim] = old
        else:
            pos[i, dim] = old
        T *= 0.999
        step *= 0.9995
    x[:] = pos.flatten()


def construct():
    deadline = time.time() + 140  # 10s safety margin for 150s limit
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = -np.inf
    
    # Phase 1: Best-fit local search from multiple inits
    n_inits = 4
    for init_i in range(n_inits):
        if time.time() > deadline - 40:
            break
        if init_i % 2 == 0:
            x0 = _initial_grid_config(rng)
        else:
            x0 = _initial_ring_config(rng)
        
        pos = x0.reshape(N, 3)
        # Ensure feasibility
        for i in range(N):
            pos[i, 2] = min(pos[i, 2], pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1], 0.49)
        
        # Run best-fit local search
        pos, val = _best_fit_local_search(pos, rng, deadline - 30, n_passes=30)
        
        if _is_valid(pos.flatten()):
            v = _sum_radii(pos.flatten())
            if v > best_val:
                best_val = v
                best_x = pos.flatten().copy()
    
    # Phase 2: Gradient polish on best result
    if best_x is not None and time.time() < deadline - 20:
        polished = _gradient_polish(best_x.copy(), deadline - 15)
        if _is_valid(polished):
            v = _sum_radii(polished)
            if v > best_val:
                best_val = v
                best_x = polished.copy()
    
    # Phase 3: SA refinement
    if best_x is not None and time.time() < deadline - 5:
        _sa_refine(best_x, rng, deadline - 3)
        if _is_valid(best_x):
            v = _sum_radii(best_x)
            if v > best_val:
                best_val = v
    
    # Phase 4: Final gradient polish
    if best_x is not None and time.time() < deadline - 2:
        polished = _gradient_polish(best_x.copy(), deadline - 1, penalty_weight=500.0)
        if _is_valid(polished):
            v = _sum_radii(polished)
            if v > best_val:
                best_val = v
                best_x = polished.copy()
    
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_x.reshape(N, 3)
