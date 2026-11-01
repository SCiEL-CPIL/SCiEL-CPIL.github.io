import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
DEADLINE = 145.0  # Safety margin before 150s limit

def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r

def _softplus(x, beta):
    """Numerically stable softplus: log(1 + exp(beta*x))/beta"""
    # Clip x to avoid overflow in exp
    x_clipped = np.clip(x, -50/beta, 50/beta)
    return np.log1p(np.exp(beta * x_clipped)) / beta

def _sigmoid(x, beta):
    """Numerically stable sigmoid: 1/(1+exp(-beta*x))"""
    x_clipped = np.clip(x, -50/beta, 50/beta)
    return 1.0 / (1.0 + np.exp(-beta * x_clipped))

def _packing_objective_and_grad(x, beta, lambda_r):
    """
    Objective: sum of softplus violations - lambda_r * r
    x: flattened array [c_0x, c_0y, ..., c_{N-1}x, c_{N-1}y, r]
    """
    c = x[:-1].reshape(N, 2)
    r = x[-1]
    
    # Pairwise distances
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, 1e-12) # Avoid div by zero
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = 2.0 * r - pair_dist
    
    # Wall distances
    d_left = c[:, 0] + HALF
    d_right = HALF - c[:, 0]
    d_bottom = c[:, 1] + HALF
    d_top = HALF - c[:, 1]
    
    wall_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = r - wall_dist
    
    # Objective
    obj = np.sum(_softplus(pair_viol, beta)) + np.sum(_softplus(wall_viol, beta)) - lambda_r * r
    
    # Gradients
    grad = np.zeros_like(x)
    
    # Pair gradients
    # d/dx softplus(beta*(2r - d)) = sigmoid(beta*(2r-d)) * beta * d(2r-d)/dx
    # d(2r-d)/dr = 2
    # d(2r-d)/dc_i = - (c_i - c_j)/d
    sig_pair = _sigmoid(pair_viol, beta)
    
    # Gradient w.r.t r from pairs
    grad_r_pairs = np.sum(sig_pair * 2.0)
    
    # Gradient w.r.t c from pairs
    grad_c = np.zeros_like(c)
    # We need to accumulate gradients for each point
    # sig_pair is vector of length N*(N-1)/2
    # We can use the full matrix for easier vectorization
    sig_full = np.zeros((N, N))
    sig_full[iu] = sig_pair
    sig_full = sig_full + sig_full.T
    
    # grad_c[i] = sum_j sigmoid(beta*(2r-d_ij)) * (c_i - c_j)/d_ij
    # Note: (c_i - c_j)/d_ij is the unit vector from j to i.
    # The derivative of dist w.r.t c_i is (c_i - c_j)/dist.
    # So d(2r - dist)/dc_i = -(c_i - c_j)/dist.
    # Thus d obj/dc_i = sig * (-(c_i - c_j)/dist) = sig * (c_j - c_i)/dist.
    
    # Vectorized calculation of gradients
    # diff is (N, N, 2). dist is (N, N).
    # unit_vec[i, j] = diff[i, j] / dist[i, j] = (c_i - c_j) / dist_ij
    unit_vec = diff / dist[:, :, np.newaxis]
    
    # grad_c[i] = - sum_j sig_full[i, j] * unit_vec[i, j]
    # Wait, let's re-verify the sign.
    # obj = softplus(beta * (2r - d_ij))
    # d obj/d c_i = sigmoid(beta * (2r - d_ij)) * beta * d(2r - d_ij)/d c_i
    # d(2r - d_ij)/d c_i = - d(d_ij)/d c_i = - (c_i - c_j)/d_ij
    # So d obj/d c_i = - sigmoid(...) * (c_i - c_j)/d_ij
    
    grad_c = -np.sum(sig_full[:, :, np.newaxis] * unit_vec, axis=1)
    
    # Wall gradients
    sig_wall = _sigmoid(wall_viol, beta)
    
    # Gradient w.r.t r from walls
    grad_r_walls = np.sum(sig_wall)
    
    # Gradient w.r.t c from walls
    # d(r - w_i)/dr = 1
    # d(r - w_i)/dc_i = - d(w_i)/dc_i
    # w_i = min(d_left, d_right, d_bottom, d_top)
    # If closest is left (d_left), w_i = x + 0.5, dw/dx = 1, dw/dy = 0
    # If closest is right (d_right), w_i = 0.5 - x, dw/dx = -1, dw/dy = 0
    # If closest is bottom (d_bottom), w_i = y + 0.5, dw/dx = 0, dw/dy = 1
    # If closest is top (d_top), w_i = 0.5 - y, dw/dx = 0, dw/dy = -1
    
    grad_wall_c = np.zeros_like(c)
    for i in range(N):
        dl = d_left[i]
        dr = d_right[i]
        db = d_bottom[i]
        dt = d_top[i]
        wmin = wall_dist[i]
        s = sig_wall[i]
        
        # Soft assignment if close to tie, but hard assignment is usually fine for L-BFGS
        # To be smooth, we could use softmax over the 4 walls, but let's stick to hard for speed
        # and rely on the optimizer to handle non-smoothness or use a very small beta initially.
        # Actually, hard min is non-differentiable. Let's use a soft-min for the wall distance too?
        # No, the wall distance is a linear function of c. The min of linear functions is concave.
        # We are penalizing r - w_i.
        # Let's just use the hard gradient. L-BFGS can handle mild non-smoothness.
        
        if abs(dl - wmin) < 1e-10:
            grad_wall_c[i, 0] -= s * 1.0  # d(r - (x+0.5))/dx = -1
        if abs(dr - wmin) < 1e-10:
            grad_wall_c[i, 0] -= s * (-1.0) # d(r - (0.5-x))/dx = 1
        if abs(db - wmin) < 1e-10:
            grad_wall_c[i, 1] -= s * 1.0   # d(r - (y+0.5))/dy = -1
        if abs(dt - wmin) < 1e-10:
            grad_wall_c[i, 1] -= s * (-1.0) # d(r - (0.5-y))/dy = 1
            
    grad[:-1] = (grad_c + grad_wall_c).ravel()
    grad[-1] = grad_r_pairs + grad_r_walls - lambda_r
    
    return float(obj), grad

def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice initialization."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)

def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)

def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_r = 0.0
    
    # Initializations
    inits = [
        _hex_init(rng, jitter=0.0),
        _hex_init(rng, jitter=0.01),
        _grid_init(rng, jitter=0.01),
        _random_init(rng),
    ]
    
    # Optimization parameters
    beta_schedule = [50, 100, 200, 500, 1000]
    lambda_r = 10.0  # Weight for maximizing r
    
    for init in inits:
        if time.time() > deadline - 20.0:
            break
            
        current_pts = init
        current_r = _compute_r(current_pts)
        
        if current_r > best_r:
            best_r = current_r
            best_pts = current_pts.copy()
            
        # Optimize starting from this init
        x0 = np.concatenate([current_pts.ravel(), [current_r]])
        
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
                
            def obj_grad(x):
                return _packing_objective_and_grad(x, beta, lambda_r)
                
            try:
                res = minimize(
                    lambda x: obj_grad(x)[0],
                    x0,
                    jac=lambda x: obj_grad(x)[1],
                    method='L-BFGS-B',
                    bounds=[(-HALF, HALF)] * (2 * N) + [(0.0, HALF)],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                
                x_opt = res.x
                c_opt = x_opt[:-1].reshape(N, 2)
                r_opt = x_opt[-1]
                
                # Clip to ensure validity
                c_opt = np.clip(c_opt, -HALF, HALF)
                
                r_achieved = _compute_r(c_opt)
                
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_pts = c_opt.copy()
                    
                # Use the optimized position as start for next beta
                x0 = np.concatenate([c_opt.ravel(), [r_achieved]])
                
            except Exception as e:
                # If optimization fails, break beta loop
                break
                
    # Final polish: try to push r a bit higher with a very high beta
    if best_pts is not None and time.time() < deadline - 2.0:
        x0 = np.concatenate([best_pts.ravel(), [best_r]])
        beta_high = 2000
        try:
            res = minimize(
                lambda x: _packing_objective_and_grad(x, beta_high, lambda_r)[0],
                x0,
                jac=lambda x: _packing_objective_and_grad(x, beta_high, lambda_r)[1],
                method='L-BFGS-B',
                bounds=[(-HALF, HALF)] * (2 * N) + [(0.0, HALF)],
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c_opt = res.x[:-1].reshape(N, 2)
            c_opt = np.clip(c_opt, -HALF, HALF)
            r_achieved = _compute_r(c_opt)
            if r_achieved > best_r:
                best_r = r_achieved
                best_pts = c_opt
        except Exception:
            pass
            
    return best_pts
