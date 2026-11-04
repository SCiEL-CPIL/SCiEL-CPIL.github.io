import numpy as np
import time
from scipy.optimize import minimize

N = 26
START_TIME = None

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return 145.0 - _elapsed()

def _feasible_score(centers, radii, tol=1e-8):
    """Return sum of radii if feasible, else -inf."""
    n = len(centers)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < -tol:
            return -np.inf
        if x < -tol or x > 1 + tol or y < -tol or y > 1 + tol:
            return -np.inf
        if r > x + tol or r > 1 - x + tol or r > y + tol or r > 1 - y + tol:
            return -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return -np.inf
    return float(np.sum(radii))

def _soft_objective(params, beta=30.0):
    """
    params: shape (3N,) = [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    Objective: maximize sum(r) - penalty
    Penalty uses soft-max for smoothness.
    """
    n = N
    x = params[0:n]
    y = params[n:2*n]
    r = params[2*n:3*n]
    
    # Soft boundary: for each circle, the constraint is r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # We use: penalty = soft_max(0, r_i - x_i, r_i - (1-x_i), r_i - y_i, r_i - (1-y_i))
    # soft_max of values v_0..v_k = (1/beta) * log(sum(exp(beta * v_k)))
    
    total_penalty = 0.0
    total_grad = np.zeros(3*n)
    
    # Boundary penalties
    for i in range(n):
        vals = np.array([0.0, r[i] - x[i], r[i] - (1 - x[i]), r[i] - y[i], r[i] - (1 - y[i])])
        # soft_max
        max_val = np.max(vals)
        if max_val > 1e-10:
            exp_vals = np.exp(beta * (vals - max_val))
            soft_max_val = max_val + np.log(np.sum(exp_vals)) / beta
            total_penalty += soft_max_val
            # Gradient
            # d(soft_max)/d(vals) = exp_vals / sum(exp_vals)
            probs = exp_vals / np.sum(exp_vals)
            # d(vals)/d(params)
            # vals[0] = 0, vals[1] = r_i - x_i, vals[2] = r_i - (1-x_i), vals[3] = r_i - y_i, vals[4] = r_i - (1-y_i)
            # d(vals[1])/dx_i = -1, d(vals[1])/dr_i = 1
            # d(vals[2])/dx_i = 1, d(vals[2])/dr_i = 1
            # d(vals[3])/dy_i = -1, d(vals[3])/dr_i = 1
            # d(vals[4])/dy_i = 1, d(vals[4])/dr_i = 1
            total_grad[i] += -probs[1] + probs[2]  # dx_i
            total_grad[n + i] += -probs[3] + probs[4]  # dy_i
            total_grad[2*n + i] += probs[1] + probs[2] + probs[3] + probs[4]  # dr_i
    
    # Pairwise penalties
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < 1e-12:
                continue
            overlap = r[i] + r[j] - dist
            if overlap > 1e-10:
                # soft penalty: overlap^2 smoothed
                # Use: penalty = (1/beta) * log(1 + exp(beta * overlap))
                # This is a smooth approximation to max(0, overlap)
                # Actually let's use a simpler smooth penalty
                # penalty = overlap^2 / (2 * smooth_dist)
                # Or just use the exponential soft-max approach
                # Let's use: penalty = (1/beta) * log(1 + exp(beta * overlap))
                # Gradient: d/doverlap = exp(beta*overlap)/(1+exp(beta*overlap)) = sigmoid(beta*overlap)
                # d/d(params) = sigmoid * d(overlap)/d(params)
                
                # For numerical stability
                if beta * overlap > 50:
                    sig = 1.0
                elif beta * overlap < -50:
                    sig = 0.0
                else:
                    sig = 1.0 / (1.0 + np.exp(-beta * overlap))
                
                pen = (1.0 / beta) * np.log(1.0 + np.exp(np.clip(beta * overlap, -50, 50)))
                total_penalty += pen
                total_grad[i] += sig * dx / dist
                total_grad[n + i] += sig * dy / dist
                total_grad[2*n + i] += sig
                total_grad[j] -= sig * dx / dist
                total_grad[n + j] -= sig * dy / dist
                total_grad[2*n + j] += sig
    
    value = np.sum(r) - total_penalty
    return -value, -total_grad  # Negate for minimization

def _project_feasible(centers, radii, tol=1e-9):
    """Shrink radii to make packing feasible."""
    radii = radii.copy()
    for _ in range(500):
        feasible = True
        for i in range(N):
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            if radii[i] > max_r + tol:
                radii[i] = max(0.0, max_r)
                feasible = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    excess = radii[i] + radii[j] - dist
                    if excess > 0:
                        radii[i] = max(0.0, radii[i] - excess / 2.0)
                        radii[j] = max(0.0, radii[j] - excess / 2.0)
                        feasible = False
        if feasible:
            break
    return radii

def _score(centers, radii):
    """Return the feasible score."""
    return _feasible_score(centers, radii)

def _hex_packing_init(seed=42):
    """Hexagonal packing initialization."""
    rng = np.random.default_rng(seed)
    centers = []
    radii = []
    
    # Try different hexagonal arrangements
    for r0 in [0.085, 0.09, 0.095]:
        dy = r0 * 2 * 0.866
        centers = []
        for row in range(7):
            x_offset = 0 if row % 2 == 0 else r0
            for col in range(7):
                x = r0 + col * 2 * r0 + x_offset
                y = r0 + row * dy
                if x < 1 - r0 + 0.01 and y < 1 - r0 + 0.01:
                    centers.append((x, y))
                if len(centers) >= N:
                    break
            if len(centers) >= N:
                break
        if len(centers) >= N:
            break
    
    # Fill remaining with random
    while len(centers) < N:
        centers.append((rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)))
    
    centers = np.array(centers[:N])
    radii = np.zeros(N)
    
    # Compute initial radii
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(N):
            if j == i:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist / 2.0)
        radii[i] = max(0.0, max_r)
    
    return centers, radii

def _structured_init(seed=42):
    """Structured grid initialization with jitter."""
    rng = np.random.default_rng(seed)
    centers = []
    radii = []
    
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            x = (i + 0.5) / 5.0 + rng.uniform(-0.02, 0.02)
            y = (j + 0.5) / 5.0 + rng.uniform(-0.02, 0.02)
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers.append((x, y))
    
    # Add 6 more
    gap_positions = [
        (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9),
        (0.5, 0.1), (0.5, 0.9)
    ]
    for x, y in gap_positions:
        if len(centers) >= N:
            break
        x += rng.uniform(-0.02, 0.02)
        y += rng.uniform(-0.02, 0.02)
        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        centers.append((x, y))
    
    centers = np.array(centers[:N])
    radii = np.zeros(N)
    
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(N):
            if j == i:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist / 2.0)
        radii[i] = max(0.0, max_r)
    
    return centers, radii

def _random_init(seed=42):
    """Random initialization."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.1, 0.9, size=(N, 2))
    radii = np.zeros(N)
    
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(N):
            if j == i:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist / 2.0)
        radii[i] = max(0.0, max_r)
    
    return centers, radii

def _grid_init():
    """Simple grid initialization."""
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append(((i + 0.5) / 5.0, (j + 0.5) / 5.0))
    centers.append((0.5, 0.5))
    
    centers = np.array(centers[:N])
    radii = np.zeros(N)
    
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(N):
            if j == i:
                continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, dist / 2.0)
        radii[i] = max(0.0, max_r)
    
    return centers, radii

def _optimize_from(centers, radii, deadline):
    """Run L-BFGS-B optimization from a starting point."""
    params = np.concatenate([centers[:, 0], centers[:, 1], radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N
    
    def objective_and_grad(p):
        val, grad = _soft_objective(p, beta=30.0)
        return val, grad
    
    try:
        result = minimize(
            objective_and_grad,
            params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < objective_and_grad(params)[0]:
            opt_params = result.x
            opt_centers = np.column_stack([opt_params[0:N], opt_params[N:2*N]])
            opt_radii = opt_params[2*N:3*N]
            opt_radii = _project_feasible(opt_centers, opt_radii)
            return opt_centers, opt_radii
    except Exception:
        pass
    
    return centers, radii

def _local_search_polish(centers, radii, deadline):
    """Local search: move each circle to maximize its radius."""
    centers = centers.copy()
    radii = radii.copy()
    rng = np.random.default_rng(42)
    
    for iteration in range(100):
        if _elapsed() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = centers[i].copy()
            best_r = radii[i]
            
            # Try random perturbations
            for step in [0.01, 0.02, 0.05]:
                for _ in range(20):
                    angle = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(0, step)
                    nx = centers[i, 0] + dist * np.cos(angle)
                    ny = centers[i, 1] + dist * np.sin(angle)
                    nx = np.clip(nx, 0, 1)
                    ny = np.clip(ny, 0, 1)
                    
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    for j in range(N):
                        if j == i:
                            continue
                        dx = nx - centers[j, 0]
                        dy = ny - centers[j, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 1e-12:
                            max_r = min(max_r, d - radii[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-12:
                        best_r = max_r
                        best_pos = np.array([nx, ny])
                        improved = True
            
            centers[i] = best_pos
            radii[i] = best_r
        
        if not improved:
            break
    
    return centers, radii

def _grow_radii(centers, radii, deadline):
    """Try to grow radii by adjusting positions."""
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(50):
        if _elapsed() > deadline:
            break
        
        improved = False
        for i in range(N):
            # Try to increase radius by moving slightly
            for _ in range(10):
                angle = np.random.uniform(0, 2 * np.pi)
                step = np.random.uniform(0, 0.01)
                nx = centers[i, 0] + step * np.cos(angle)
                ny = centers[i, 1] + step * np.sin(angle)
                nx = np.clip(nx, 0, 1)
                ny = np.clip(ny, 0, 1)
                
                # Check if we can increase radius
                new_r = radii[i] + 0.001
                feasible = True
                max_r = min(nx, 1 - nx, ny, 1 - ny)
                if new_r > max_r:
                    feasible = False
                else:
                    for j in range(N):
                        if j == i:
                            continue
                        dx = nx - centers[j, 0]
                        dy = ny - centers[j, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d < new_r + radii[j]:
                            feasible = False
                            break
                
                if feasible:
                    centers[i] = [nx, ny]
                    radii[i] = new_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return centers, radii

def construct():
    global START_TIME
    START_TIME = time.time()
    deadline = 145.0  # Safety margin
    
    best_score = -np.inf
    best_centers = None
    best_radii = None
    
    # Generate initial configurations
    inits = []
    
    c, r = _hex_packing_init(42)
    inits.append((c, r, "hex42"))
    
    c, r = _hex_packing_init(123)
    inits.append((c, r, "hex123"))
    
    c, r = _structured_init(42)
    inits.append((c, r, "struct42"))
    
    c, r = _structured_init(2020)
    inits.append((c, r, "struct2020"))
    
    c, r = _random_init(42)
    inits.append((c, r, "rand42"))
    
    c, r = _random_init(789)
    inits.append((c, r, "rand789"))
    
    c, r = _grid_init()
    inits.append((c, r, "grid"))
    
    # Optimize each initialization
    for idx, (centers, radii, name) in enumerate(inits):
        if _elapsed() > deadline:
            break
        
        try:
            # Phase 1: L-BFGS-B optimization
            c_opt, r_opt = _optimize_from(centers, radii, deadline)
            r_opt = _project_feasible(c_opt, r_opt)
            
            score = _feasible_score(c_opt, r_opt)
            if score > best_score:
                best_score = score
                best_centers = c_opt.copy()
                best_radii = r_opt.copy()
            
            # Phase 2: Local search polish
            if _elapsed() < deadline - 15:
                c_ls, r_ls = _local_search_polish(best_centers, best_radii, deadline)
                r_ls = _project_feasible(c_ls, r_ls)
                score_ls = _feasible_score(c_ls, r_ls)
                if score_ls > best_score:
                    best_score = score_ls
                    best_centers = c_ls.copy()
                    best_radii = r_ls.copy()
            
            # Phase 3: Grow radii
            if _elapsed() < deadline - 5:
                c_gr, r_gr = _grow_radii(best_centers, best_radii, deadline)
                r_gr = _project_feasible(c_gr, r_gr)
                score_gr = _feasible_score(c_gr, r_gr)
                if score_gr > best_score:
                    best_score = score_gr
                    best_centers = c_gr.copy()
                    best_radii = r_gr.copy()
                    
        except Exception:
            pass
    
    # Final validation
    if best_centers is not None:
        final_score, final_valid = _feasible_score(best_centers, best_radii), True
        if _feasible_score(best_centers, best_radii) < 0:
            best_radii = _project_feasible(best_centers, best_radii)
            if _feasible_score(best_centers, best_radii) < 0:
                # Fallback
                side = int(np.ceil(np.sqrt(N)))
                cells = []
                for i in range(side):
                    for j in range(side):
                        if len(cells) >= N:
                            break
                        x = (i + 0.5) / side
                        y = (j + 0.5) / side
                        r = 0.5 / side - 1e-3
                        cells.append([x, y, r])
                return np.array(cells)
    
    if best_centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return np.column_stack((best_centers, best_radii))
