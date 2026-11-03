import numpy as np
import time
from scipy.optimize import minimize

N = 200
HALF = 0.5
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _compute_radius(c):
    """Compute the valid radius for a configuration. Returns 0 if invalid."""
    c = np.asarray(c)
    if c.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    
    # Check inside square
    if np.any(np.abs(c) > HALF + 1e-12):
        return 0.0
    
    # Compute min pairwise distance / 2
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Compute min wall margin
    margins = HALF - np.max(np.abs(c), axis=1)
    min_margin = np.min(margins)
    
    return min(min_pair, min_margin)

def _soft_min_neg(x, beta):
    """
    Objective: -soft_min(distances)
    We want to maximize min(distances).
    soft_min(d) = -1/beta * log(sum(exp(-beta * d)))
    So -soft_min(d) = 1/beta * log(sum(exp(-beta * d)))
    
    To avoid overflow, we use the log-sum-exp trick.
    Let M = min(d).
    sum(exp(-beta * d)) = exp(-beta * M) * sum(exp(-beta * (d - M)))
    log(sum(...)) = -beta * M + log(sum(exp(-beta * (d - M))))
    
    Objective = 1/beta * (-beta * M + log(sum(exp(-beta * (d - M)))))
             = -M + 1/beta * log(sum(exp(-beta * (d - M))))
    
    As beta -> inf, this approaches -M.
    We minimize this objective.
    """
    c = x.reshape(N, 2)
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    
    # Wall distances
    # Distance from point i to nearest wall
    # d_wall_i = min(HALF - |x_i|, HALF - |y_i|)
    abs_c = np.abs(c)
    d_wall = np.minimum(HALF - abs_c[:, 0], HALF - abs_c[:, 1])
    
    # Combine all distances into one vector
    # We need to be careful with the number of distances.
    # Pairs: N*(N-1)/2 ~ 20000
    # Walls: N = 200
    # Total ~ 20200. This is manageable.
    
    # Extract upper triangle of d
    iu = np.triu_indices(N, k=1)
    d_pairs = d[iu]
    
    # d_all = concatenate(d_pairs, d_wall)
    # But we need to handle the log-sum-exp efficiently.
    
    # Let's compute the soft min of d_pairs and d_wall separately?
    # No, min(A, B) = min(min(A), min(B)).
    # But soft_min is not linear.
    # soft_min(A U B) = soft_min(concat(A, B))
    
    # Let's just build the full vector.
    # To save memory, we can compute in chunks or just accept the ~20k array.
    # 20k floats is tiny.
    
    d_all = np.concatenate([d_pairs, d_wall])
    
    # Log-sum-exp
    # L = log(sum(exp(-beta * d_all)))
    # M = min(d_all)
    # L = -beta * M + log(sum(exp(-beta * (d_all - M))))
    
    M = np.min(d_all)
    exp_terms = np.exp(-beta * (d_all - M))
    S = np.sum(exp_terms)
    
    if S == 0:
        return 0.0 # Should not happen
    
    L = -beta * M + np.log(S)
    
    obj = 1.0 / beta * L
    # obj approx -M
    
    # Gradient
    # d(obj)/dx = 1/beta * dL/dx
    # L = -beta * M + log(S)
    # dL/dx = -beta * dM/dx + (1/S) * dS/dx
    # S = sum(exp(-beta * d_k))
    # dS/dx = sum(-beta * exp(-beta * d_k) * dd_k/dx)
    # Let w_k = exp(-beta * d_k) / S  (weights)
    # dS/dx = -beta * sum(w_k * S * dd_k/dx) ? No.
    # dS/dx = sum(-beta * exp(-beta * d_k) * dd_k/dx)
    # (1/S) * dS/dx = -beta * sum( (exp(-beta * d_k)/S) * dd_k/dx )
    #                = -beta * sum( w_k * dd_k/dx )
    
    # So dL/dx = -beta * dM/dx - beta * sum(w_k * dd_k/dx)
    
    # dM/dx is the gradient of the minimum.
    # M = min(d_all). The gradient is the gradient of the argmin term.
    # If we use the soft min, the "effective" gradient is the weighted average of gradients of all terms.
    # Actually, for the soft min, the gradient is:
    # grad(soft_min) = - sum(w_k * grad(d_k))
    # Wait, let's re-derive.
    # f(x) = -1/beta log(sum exp(-beta d_k))
    # grad f = -1/beta * 1/S * dS/dx
    #         = -1/beta * 1/S * sum(-beta exp(-beta d_k) grad d_k)
    #         = sum( (exp(-beta d_k)/S) * grad d_k )
    #         = sum(w_k * grad d_k)
    
    # So we just need the weighted average of the gradients of all distances.
    
    # Gradients of d_pairs:
    # d_k = ||c_i - c_j||
    # grad_{c_i} d_k = (c_i - c_j) / d_k
    # grad_{c_j} d_k = (c_j - c_i) / d_k
    
    # Gradients of d_wall:
    # d_wall_i = min(HALF - |x_i|, HALF - |y_i|)
    # This is non-smooth.
    # For the soft min approach, we should use a smooth approximation of the wall distance too?
    # Or just use the subgradient?
    # If we use the exact min, the gradient is undefined at the kink.
    # Let's use a smooth wall distance:
    # d_wall_i = -1/beta_w log( exp(-beta_w (HALF - |x_i|)) + exp(-beta_w (HALF - |y_i|)) )
    # This is the soft min of the two wall distances.
    # Let's stick to the simple min for now and use subgradient (average of active gradients).
    # Or better, just use the gradient of the term that is currently minimal.
    # In the soft min context, the weights w_k will naturally handle the "softness".
    # But the wall distance itself is a min.
    # Let's approximate the wall distance gradient as follows:
    # If HALF - |x_i| < HALF - |y_i|, grad is (-sign(x_i), 0)
    # If HALF - |y_i| < HALF - |x_i|, grad is (0, -sign(y_i))
    # If equal, average.
    
    # Let's compute the gradients for pairs first.
    grad = np.zeros_like(c)
    
    # Weights for pairs
    # w_pair_k = exp(-beta * d_k) / S_total
    # S_total = sum_pairs(exp(-beta d_k)) + sum_walls(exp(-beta d_wall))
    
    # To be efficient, let's compute the numerator for the gradient sum directly.
    # Num = sum_pairs( exp(-beta d_k) * grad d_k ) + sum_walls( exp(-beta d_wall_i) * grad d_wall_i )
    # Grad = Num / S_total
    
    # Let's compute S_pairs and S_walls separately to avoid building the huge d_all array if possible?
    # No, d_all is small.
    
    # Recalculate S and weights from d_all
    # w_all = exp(-beta * (d_all - M)) / S
    # This is already computed in exp_terms / S
    
    # Map back to pairs and walls
    n_pairs = len(d_pairs)
    w_pairs = exp_terms[:n_pairs] / S
    w_walls = exp_terms[n_pairs:] / S
    
    # Gradient from pairs
    # grad_pairs = sum_k w_k * (c_i - c_j)/d_k for i, j
    # We can vectorize this.
    # diff[i, j] = c_i - c_j
    # d[i, j] = distance
    # We need sum over j for each i.
    
    # Let's create a matrix of weights W_pair of shape (N, N)
    # W_pair[i, j] = w_k if (i, j) is the k-th pair, 0 otherwise.
    # Since we only have upper triangle, we need to fill both (i, j) and (j, i) with the same weight.
    
    W = np.zeros((N, N))
    W[iu] = w_pairs
    W = W + W.T  # Symmetrize
    
    # Now, grad_i = sum_j W[i, j] * (c_i - c_j) / d[i, j]
    # Note: d[i, i] is inf, W[i, i] is 0.
    # d_safe = np.where(d > 0, d, 1.0)
    d_safe = np.where(d > 1e-12, d, 1.0)
    
    # Factor F[i, j] = W[i, j] / d[i, j]
    F = W / d_safe
    
    # grad_i = sum_j F[i, j] * (c_i - c_j)
    #         = c_i * sum_j F[i, j] - sum_j F[i, j] * c_j
    #         = c_i * F[i, :].sum() - F[i, :] @ c
    
    row_sums = F.sum(axis=1)
    Fc = F @ c
    
    grad_pairs = c * row_sums[:, None] - Fc
    
    # Gradient from walls
    # For each i, we have weight w_walls[i].
    # We need grad of d_wall_i.
    # d_wall_i = min(HALF - |x_i|, HALF - |y_i|)
    # Let dx = HALF - |x_i|, dy = HALF - |y_i|
    # If dx < dy: grad = (-sign(x_i), 0)
    # If dy < dx: grad = (0, -sign(y_i))
    # If dx == dy: grad = (-0.5*sign(x_i), -0.5*sign(y_i))
    
    abs_c = np.abs(c)
    dx = HALF - abs_c[:, 0]
    dy = HALF - abs_c[:, 1]
    
    grad_walls = np.zeros_like(c)
    
    # Vectorized logic
    # mask_x = dx < dy
    # mask_y = dy < dx
    # mask_eq = (dx == dy)
    
    # To handle floating point equality, use a small epsilon
    eps = 1e-12
    mask_x = dx < dy - eps
    mask_y = dy < dx - eps
    mask_eq = ~(mask_x | mask_y)
    
    # sign(x_i) is 1 if x_i > 0, -1 if x_i < 0, 0 if x_i == 0
    sx = np.sign(c[:, 0])
    sy = np.sign(c[:, 1])
    
    # If x_i is 0, sign is 0. But derivative of |x| at 0 is undefined.
    # Usually we take 0 or subgradient.
    # If x=0, dx = HALF. dy = HALF - |y|.
    # If y != 0, dy < HALF (assuming |y|>0). So dx > dy. So mask_y is true.
    # So the x-component gradient is 0 anyway.
    
    grad_walls[mask_x, 0] = -sx[mask_x]
    grad_walls[mask_y, 1] = -sy[mask_y]
    
    # For equal case, average
    grad_walls[mask_eq, 0] = -0.5 * sx[mask_eq]
    grad_walls[mask_eq, 1] = -0.5 * sy[mask_eq]
    
    # Weight by w_walls
    grad_walls *= w_walls[:, None]
    
    grad_total = grad_pairs + grad_walls
    
    return obj, grad_total.ravel()

def _initial_hex_grid(s, ox, oy):
    """Generate hex grid points, select N, return array (N, 2)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    cols = int(np.ceil(1.0 / dx)) + 4
    rows = int(np.ceil(1.0 / dy)) + 4
    xs = np.arange(-2, cols) * dx + ox
    ys = np.arange(-2, rows) * dy + oy
    
    pts = []
    for j, x in enumerate(xs):
        yoff = (dy / 2.0) if (j % 2 == 1) else 0.0
        for i, y in enumerate(ys):
            pts.append((x, y + yoff))
    arr = np.array(pts)
    
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    if arr.shape[0] > N:
        # Select N points furthest from each other (greedy farthest point sampling)
        selected = []
        min_dists = np.full(arr.shape[0], np.inf)
        idxs = np.arange(arr.shape[0])
        rng = np.random.default_rng(42)
        start = rng.choice(idxs)
        selected.append(start)
        
        for _ in range(N - 1):
            last = arr[selected[-1]]
            d = np.linalg.norm(arr - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        arr = arr[selected]
        
    return arr

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_r = 0.0
    
    # Initial configurations
    configs = []
    
    # 1. Hex grid
    s_hex = 2.0 * 0.0366 * 1.05
    c_hex = _initial_hex_grid(s_hex, 0.0, 0.0)
    if c_hex.shape[0] == N:
        configs.append(c_hex)
        
    # 2. Hex grid shifted
    c_hex2 = _initial_hex_grid(s_hex, 0.01, 0.01)
    if c_hex2.shape[0] == N:
        configs.append(c_hex2)
        
    # 3. Random
    c_rand = rng.uniform(-HALF + 0.04, HALF - 0.04, (N, 2))
    configs.append(c_rand)
        
    # 4. Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if grid.shape[0] >= N:
        c_grid = grid[:N]
        configs.append(c_grid)

    # Annealing schedule for beta
    # Start low, increase to high
    betas = [50, 100, 200, 500, 1000, 2000, 5000, 10000]
    
    # Bounds for L-BFGS-B
    bounds = [(-HALF, HALF)] * (2 * N)
    
    for c_init in configs:
        if time.time() > DEADLINE:
            break
            
        # Verify initial config
        r_current = _compute_radius(c_init)
        if r_current <= 0:
            c_init = np.clip(c_init, -HALF + 0.03, HALF - 0.03)
            r_current = _compute_radius(c_init)
            if r_current <= 0:
                continue
                
        x0 = c_init.ravel()
        current_x = x0.copy()
        
        # Run annealing
        for beta in betas:
            if time.time() > DEADLINE:
                break
                
            try:
                res = minimize(
                    _soft_min_neg,
                    current_x,
                    args=(beta,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-6, 'disp': False}
                )
                
                current_x = res.x
                c_opt = current_x.reshape(N, 2)
                
                r_opt = _compute_radius(c_opt)
                
                if r_opt > best_r:
                    best_r = r_opt
                    best_c = c_opt.copy()
                    
            except Exception as e:
                # If optimization fails, keep current_x
                pass

    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
    else:
        # Final polish with very high beta
        if time.time() < DEADLINE - 10:
            beta_final = 20000
            try:
                res = minimize(
                    _soft_min_neg,
                    best_c.ravel(),
                    args=(beta_final,),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
                )
                c_polished = res.x.reshape(N, 2)
                r_polished = _compute_radius(c_polished)
                if r_polished > best_r:
                    best_r = r_polished
                    best_c = c_polished
            except Exception:
                pass

    return best_c

if __name__ == "__main__":
    construct()
