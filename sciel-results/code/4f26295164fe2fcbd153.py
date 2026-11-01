"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_initialization(n):
    """Generate hexagonal close-packed initial positions."""
    pts = []
    r = 0.5 / n ** 0.5  # approximate radius
    dx = 2 * r
    dy = 2 * r * np.sqrt(3) / 2
    row = 0
    while len(pts) < n:
        for col in range(n + 2):
            x = (col + 0.5) * dx
            y = (row + 0.5) * dy
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
            if len(pts) >= n:
                break
        row += 1
        if row * dy > 1.2:
            break
    # If we didn't get enough, fill with grid
    while len(pts) < n:
        side = int(np.ceil(np.sqrt(n)))
        for i in range(side):
            for j in range(side):
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                if [x, y] not in pts:
                    pts.append([x, y])
                    break
            if len(pts) >= n:
                break
    return np.array(pts[:n])


def _softmin(x, alpha=100):
    """Smooth approximation of min using log-sum-exp."""
    x = np.asarray(x)
    m = np.min(x)
    return m - np.log(np.sum(np.exp(alpha * (x - m)))) / alpha


def _clearance(x, y, r):
    """Compute clearance from boundaries and other circles."""
    # Boundary clearances
    boundary_clear = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    return boundary_clear - r


def _objective(params, n, alpha=100):
    """Objective: maximize sum of radii with soft penalty for overlaps."""
    pos = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Sum of radii (we want to maximize)
    sum_r = np.sum(radii)
    
    # Penalty for boundary violations
    boundary_viol = np.maximum(0, radii - np.minimum(np.minimum(pos[:, 0], 1 - pos[:, 0]), 
                                                      np.minimum(pos[:, 1], 1 - pos[:, 1])))
    boundary_pen = np.sum(boundary_viol ** 2)
    
    # Penalty for circle overlaps using soft-min
    overlap_pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            min_dist = radii[i] + radii[j]
            # Soft penalty when dist < min_dist
            gap = dist - min_dist
            if gap < 0:
                overlap_pen += gap ** 2
            elif gap < 0.1:
                # Smooth transition
                overlap_pen += (0.1 - gap) ** 2 * 0.1
    
    return -(sum_r) + 10 * boundary_pen + 5 * overlap_pen


def _is_valid(arr, tol=1e-9):
    """Check if packing is valid."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if x - r < -tol or x + r > 1 + tol or y - r < -tol or y + r > 1 + tol or r < 0:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(arr[i, :2] - arr[j, :2])
            if dist < arr[i, 2] + arr[j, 2] - tol:
                return False
    return True


def _clean(arr):
    """Clean up small numerical errors."""
    arr = arr.copy()
    for i in range(len(arr)):
        x, y, r = arr[i]
        r = max(r, 1e-10)
        x = np.clip(x, r, 1 - r)
        y = np.clip(y, r, 1 - r)
        arr[i] = [x, y, r]
    return arr


def _optimize_start(init_pos, init_radii, time_budget, alpha=100):
    """Run optimization from a starting point."""
    n = len(init_pos)
    x0 = np.concatenate([init_pos.flatten(), init_radii])
    
    def obj_and_grad(params):
        val = _objective(params, n, alpha)
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad[i] = (_objective(params_plus, n, alpha) - val) / eps
        return val, grad
    
    # Use L-BFGS-B for bounded optimization
    bounds = []
    for i in range(2*n):
        bounds.append((0, 1))
    for i in range(n):
        bounds.append((1e-10, 0.5))
    
    best_x = x0.copy()
    best_val = _objective(x0, n, alpha)
    
    start_time = time.time()
    max_iter = 200
    
    for restart in range(3):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 10:
            break
        
        res = minimize(
            lambda p: _objective(p, n, alpha),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
        
        # Add small perturbation for next restart
        x0 = best_x + np.random.randn(len(best_x)) * 0.01
        x0 = np.clip(x0, [0]*2*n + [1e-10]*n, [1]*2*n + [0.5]*n)
        
        max_iter = max(50, max_iter // 2)
    
    # Extract result
    pos = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    result = np.column_stack([pos, radii])
    
    # Final polish: local refinement
    result = _polish(result, time_budget - (time.time() - start_time))
    
    return result


def _polish(arr, time_budget):
    """Final high-precision polish pass."""
    n = len(arr)
    if time_budget < 1:
        return arr
    
    start_time = time.time()
    best_val = -np.sum(arr[:, 2])  # negative sum of radii
    best_arr = arr.copy()
    
    # Local optimization with tighter tolerance
    x0 = np.concatenate([arr[:, :2].flatten(), arr[:, 2]])
    
    bounds = []
    for i in range(2*n):
        bounds.append((0, 1))
    for i in range(n):
        bounds.append((1e-10, 0.5))
    
    for _ in range(5):
        elapsed = time.time() - start_time
        if elapsed > time_budget - 2:
            break
        
        res = minimize(
            lambda p: _objective(p, n, 50),
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            pos = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            best_arr = np.column_stack([pos, radii])
        
        x0 = res.x
    
    return best_arr


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Hexagonal initialization
    init_pos = _hex_initialization(N)
    init_radii = np.full(N, 0.08)
    
    # Try multiple initializations
    best_result = None
    best_score = -np.inf
    
    # First try: hexagonal with equal radii
    result = _optimize_start(init_pos, init_radii, 40)
    if _is_valid(result):
        score = np.sum(result[:, 2])
        if score > best_score:
            best_score = score
            best_result = result
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return _clean(best_result) if best_result is not None else _clean(np.column_stack([init_pos, init_radii]))
    
    # Second try: staggered grid with varying radii
    side = 6
    pts = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            if i % 2 == 1:
                y += 0.25 / side
            pts.append([x, y])
    while len(pts) < N:
        pts.append([0.5 + np.random.randn() * 0.05, 0.5 + np.random.randn() * 0.05])
    pts = np.array(pts[:N])
    radii2 = np.full(N, 0.07)
    
    result2 = _optimize_start(pts, radii2, 30)
    if _is_valid(result2):
        score2 = np.sum(result2[:, 2])
        if score2 > best_score:
            best_score = score2
            best_result = result2
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return _clean(best_result) if best_result is not None else _clean(np.column_stack([init_pos, init_radii]))
    
    # Third try: random initialization
    np.random.seed(42)
    pts3 = np.random.rand(N, 2) * 0.8 + 0.1
    radii3 = np.full(N, 0.06)
    
    result3 = _optimize_start(pts3, radii3, 25)
    if _is_valid(result3):
        score3 = np.sum(result3[:, 2])
        if score3 > best_score:
            best_score = score3
            best_result = result3
    
    if best_result is None:
        # Fallback to simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return _clean(best_result)
# EVOLVE-BLOCK-END
