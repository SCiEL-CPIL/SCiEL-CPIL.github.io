"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hexagonal_init(W, H, n_circles):
    """Generate hexagonal lattice positions for circles in rectangle [0,W]x[0,H]."""
    # Estimate radius from area: n * pi * r^2 <= W * H
    # r_est = sqrt(W * H / (n_circles * pi)) * 0.95 (some margin)
    r_est = np.sqrt(W * H / (n_circles * np.pi)) * 0.85
    
    # Hexagonal grid parameters
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    positions = []
    # Generate rows
    row = 0
    while True:
        y = r_est + row * dy
        if y + r_est > H + 1e-9:
            break
        offset = r_est if row % 2 == 1 else 0.0
        x = offset + r_est
        while x + r_est <= W + 1e-9:
            positions.append((x, y))
            x += dx
        row += 1
        if len(positions) > 200:
            break
    
    if len(positions) < n_circles:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(n_circles)))
        positions = []
        for i in range(side):
            for j in range(side):
                positions.append(((i + 0.5) / side * W, (j + 0.5) / side * H))
    
    # Take first n_circles, or if we have more, subsample
    if len(positions) > n_circles:
        # Uniformly subsample
        indices = np.linspace(0, len(positions) - 1, n_circles, dtype=int)
        positions = [positions[i] for i in indices]
    elif len(positions) < n_circles:
        # Pad with grid
        side = int(np.ceil(np.sqrt(n_circles - len(positions))))
        extra = []
        for i in range(side):
            for j in range(side):
                extra.append(((i + 0.5) / side * W, (j + 0.5) / side * H))
        positions = positions + extra[:n_circles - len(positions)]
    
    return np.array(positions[:n_circles])


def _pack_single(W, H, init_pos, time_budget):
    """Optimize circle packing for given rectangle dimensions."""
    n = len(init_pos)
    
    # Initial radii: estimate from area
    r_init = np.sqrt(W * H / (n * np.pi)) * 0.8
    r_init = np.full(n, r_init)
    
    # Variables: [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = init_pos[i, 0]
        x0[3*i+1] = init_pos[i, 1]
        x0[3*i+2] = r_init[i]
    
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, W))      # x
        bounds.append((0, H))      # y
        bounds.append((1e-6, min(W, H)/2))  # r
    
    # Precompute pair indices
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_idx = np.array(pairs)  # shape (m, 2)
    
    def objective(vars_):
        elapsed = time.time() - t_start
        if elapsed > time_budget:
            return 1e10  # signal timeout
        
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        
        # Sum of radii (to maximize, so negate)
        sum_r = np.sum(r)
        
        # Overlap penalty
        # For each pair, check if distance < r_i + r_j
        dx = x[pair_idx[:, 0]] - x[pair_idx[:, 1]]
        dy = y[pair_idx[:, 0]] - y[pair_idx[:, 1]]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = r[pair_idx[:, 0]] + r[pair_idx[:, 1]]
        overlap = np.maximum(0, r_sum - dist)
        overlap_pen = np.sum(overlap**2)
        
        # Boundary penalty
        bx = np.maximum(0, r - x) + np.maximum(0, r - (W - x))
        by = np.maximum(0, r - y) + np.maximum(0, r - (H - y))
        bound_pen = np.sum(bx**2) + np.sum(by**2)
        
        penalty = 50.0 * (overlap_pen + bound_pen)
        
        return -sum_r + penalty
    
    # Gradient (numerical approximation would be slow, use analytical)
    def gradient(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        
        grad = np.zeros(3 * n)
        
        # d(-sum_r)/dr_i = -1
        grad[2::3] += -1.0
        
        # Overlap gradient
        dx = x[pair_idx[:, 0]] - x[pair_idx[:, 1]]
        dy = y[pair_idx[:, 0]] - y[pair_idx[:, 1]]
        dist = np.sqrt(dx**2 + dy**2) + 1e-12
        r_sum = r[pair_idx[:, 0]] + r[pair_idx[:, 1]]
        overlap = np.maximum(0, r_sum - dist)
        
        if np.any(overlap > 0):
            # d(overlap^2)/dx_i for circle i in pair
            # overlap = r_i + r_j - dist
            # d(overlap^2)/dx_i = 2*overlap * (-d(dist)/dx_i) = 2*overlap * (dx_ij / dist)
            # where dx_ij = x_i - x_j
            # For circle i (index 0 in pair): d(dist)/dx_i = dx/dist, so d(overlap^2)/dx_i = -2*overlap*dx/dist
            # Wait: overlap = r_sum - dist, d(overlap)/dx_i = -d(dist)/dx_i = -(x_i - x_j)/dist = (x_j - x_i)/dist
            # d(overlap^2)/dx_i = 2*overlap * (x_j - x_i)/dist = 2*overlap * (-dx)/dist
            # where dx = x_i - x_j
            
            # Penalty contribution: 50 * overlap^2
            # d/dx_i [50 * overlap^2] = 50 * 2 * overlap * d(overlap)/dx_i = 100 * overlap * (x_j - x_i)/dist
            # = 100 * overlap * (-dx)/dist
            
            i_idx = pair_idx[:, 0]
            j_idx = pair_idx[:, 1]
            
            # For circle i (i_idx): gradient wrt x_i
            gx_i = 100.0 * overlap * (-dx) / dist
            gy_i = 100.0 * overlap * (-dy) / dist
            gr_i = 100.0 * overlap * 1.0  # d(overlap)/dr_i = 1
            
            # For circle j (j_idx): gradient wrt x_j
            gx_j = 100.0 * overlap * (dx) / dist
            gy_j = 100.0 * overlap * (dy) / dist
            gr_j = 100.0 * overlap * 1.0
            
            # Accumulate using np.add.at
            np.add.at(grad[0::3], i_idx, gx_i)
            np.add.at(grad[1::3], i_idx, gy_i)
            np.add.at(grad[2::3], i_idx, gr_i)
            np.add.at(grad[0::3], j_idx, gx_j)
            np.add.at(grad[1::3], j_idx, gy_j)
            np.add.at(grad[2::3], j_idx, gr_j)
        
        # Boundary gradient
        # bx_i = max(0, r_i - x_i) + max(0, r_i - (W - x_i))
        # d(bx_i^2)/dx_i:
        #   if x_i < r_i: d/dx_i [50*(r_i - x_i)^2] = 50 * 2*(r_i-x_i)*(-1) = -100*(r_i-x_i)
        #   if x_i > W - r_i: d/dx_i [50*(x_i - (W-r_i))^2] = 100*(x_i - W + r_i)
        
        left_viol = np.maximum(0, r - x)
        right_viol = np.maximum(0, r - (W - x))
        bottom_viol = np.maximum(0, r - y)
        top_viol = np.maximum(0, r - (H - y))
        
        grad[0::3] += -100.0 * left_viol + 100.0 * right_viol
        grad[1::3] += -100.0 * bottom_viol + 100.0 * top_viol
        grad[2::3] += 100.0 * (left_viol + right_viol + bottom_viol + top_viol)
        
        return grad
    
    result = minimize(
        objective, x0, jac=gradient, method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
    )
    
    # Extract result
    x = result.x[0::3]
    y = result.x[1::3]
    r = result.x[2::3]
    
    # Check validity: ensure no overlaps and within bounds
    valid = True
    for i in range(n):
        if x[i] < -1e-9 or x[i] > W + 1e-9 or y[i] < -1e-9 or y[i] > H + 1e-9:
            valid = False
            break
        if r[i] <= 0:
            valid = False
            break
    
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: shrink radii to make valid
        scale = 0.95
        r = r * scale
        # Check again, shrink more if needed
        for _ in range(10):
            ok = True
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
            scale *= 0.95
            r = r * scale
    
    return np.column_stack([x, y, r]), np.sum(r)


def construct():
    global t_start
    t_start = time.time()
    total_budget = 110.0  # safety margin before 120s limit
    
    best_result = None
    best_score = -1e10
    
    # Try multiple aspect ratios
    # W + H = 2, so H = 2 - W
    # W in (0, 2)
    aspect_ratios = np.linspace(0.3, 1.7, 15)
    
    for W in aspect_ratios:
        H = 2.0 - W
        if H <= 0.01:
            continue
        
        elapsed = time.time() - t_start
        if elapsed > total_budget - 5.0:
            break
        
        per_attempt_budget = min(8.0, (total_budget - elapsed) / max(1, len(aspect_ratios) - int(np.searchsorted(aspect_ratios, W))))
        if per_attempt_budget < 1.0:
            break
        
        init_pos = _hexagonal_init(W, H, N)
        result, score = _pack_single(W, H, init_pos, per_attempt_budget)
        
        if score > best_score:
            best_score = score
            best_result = result
    
    # If we still have time, try to polish the best result
    if best_result is not None:
        elapsed = time.time() - t_start
        if elapsed < total_budget - 10.0:
            # Try a few more aspect ratios near the best one
            pass
    
    if best_result is None:
        # Fallback to seed
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
