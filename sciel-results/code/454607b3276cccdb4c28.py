"""
Pack 200 equal circles in a unit square using direct constrained NLP (SLSQP).
Strategy: Maximize radius r subject to explicit non-overlap and boundary constraints.
"""
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
    # Use efficient chunking or direct for N=200 (200x200 is small)
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    # Compute min wall margin
    # margin_i = HALF - max(|x_i|, |y_i|)
    margins = HALF - np.max(np.abs(c), axis=1)
    min_margin = np.min(margins)
    
    return min(min_pair, min_margin)

def _initial_hex_grid(s, ox, oy):
    """Generate hex grid points, select N, return array (N, 2)."""
    dx = s
    dy = s * np.sqrt(3.0) / 2.0
    # Generate a large enough grid
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
    
    # Filter to square
    keep = np.all(np.abs(arr) <= HALF, axis=1)
    arr = arr[keep]
    
    # If we have more than N, select N that maximize min distance (greedy) or just take first N?
    # Greedy is better for initial quality.
    if arr.shape[0] > N:
        selected = []
        min_dists = np.full(arr.shape[0], np.inf)
        # Start with center-ish point or random
        # Let's pick the point closest to (0,0) for stability, or random
        idxs = np.arange(arr.shape[0])
        rng = np.random.default_rng(42)
        start = rng.choice(idxs)
        selected.append(start)
        
        for _ in range(N - 1):
            last = arr[selected[-1]]
            d = np.linalg.norm(arr - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            # Mask already selected
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        arr = arr[selected]
        
    return arr

def _slsqp_objective(x):
    """
    x: flattened array of shape (2N+1).
    x[:2N] = centers
    x[-1] = radius r
    Objective: -r (minimize negative radius)
    """
    r = x[-1]
    return -r

def _slsqp_grad(x):
    """
    Gradient of -r w.r.t x.
    Only the last component (r) has derivative -1.
    Centers have derivative 0 in the objective.
    """
    grad = np.zeros_like(x)
    grad[-1] = -1.0
    return grad

def _slsqp_constraints(x):
    """
    Returns list of constraint dicts.
    Constraint form: fun(x) >= 0
    """
    c = x[:2*N].reshape(N, 2)
    r = x[-1]
    
    cons = []
    
    # 1. Boundary constraints:
    # x_i - r >= -HALF  => x_i + HALF - r >= 0
    # -x_i - r >= -HALF => -x_i + HALF - r >= 0
    # So: HALF - |x_ik| - r >= 0 is not linear.
    # Wait, SLSQP supports nonlinear constraints.
    # Constraint: x_i - r >= -HALF  => x_i + HALF - r >= 0
    # Constraint: -x_i - r >= -HALF => -x_i + HALF - r >= 0
    # These are linear in x and r.
    
    # Actually, the constraint is: center must be in [ -HALF + r, HALF - r ]
    # So: x_i - r + HALF >= 0
    # And: -x_i - r + HALF >= 0
    
    # Let's add these as one vectorized constraint if possible, or loop.
    # SLSQP accepts 'ineq' with 'fun' returning array.
    
    # Boundary ineq:
    def bnd_fun(x_flat):
        c_f = x_flat[:2*N].reshape(N, 2)
        r_f = x_flat[-1]
        # x + HALF - r >= 0
        vals1 = c_f[:, 0] + HALF - r_f
        vals2 = -c_f[:, 0] + HALF - r_f
        vals3 = c_f[:, 1] + HALF - r_f
        vals4 = -c_f[:, 1] + HALF - r_f
        return np.concatenate([vals1, vals2, vals3, vals4])
    
    def bnd_jac(x_flat):
        # Jacobian of bnd_fun w.r.t x_flat
        # Shape: (4N, 2N+1)
        J = np.zeros((4*N, 2*N+1))
        # For x_i, y_i, r
        # d/dx_i of (x_i + H - r) = 1
        # d/dx_i of (-x_i + H - r) = -1
        # d/dy_i of (y_i + H - r) = 1
        # d/dy_i of (-y_i + H - r) = -1
        # d/dr of all = -1
        
        for i in range(N):
            idx_cx = 2*i
            idx_cy = 2*i + 1
            idx_r = 2*N
            
            # Row 0: x_i + H - r
            J[i, idx_cx] = 1.0
            J[i, idx_r] = -1.0
            
            # Row 1: -x_i + H - r
            J[N+i, idx_cx] = -1.0
            J[N+i, idx_r] = -1.0
            
            # Row 2: y_i + H - r
            J[2*N+i, idx_cy] = 1.0
            J[2*N+i, idx_r] = -1.0
            
            # Row 3: -y_i + H - r
            J[3*N+i, idx_cy] = -1.0
            J[3*N+i, idx_r] = -1.0
            
        return J
    
    cons.append({'type': 'ineq', 'fun': bnd_fun, 'jac': bnd_jac})
    
    # 2. Non-overlap constraints:
    # |c_i - c_j|^2 - (2r)^2 >= 0
    # d_ij^2 - 4r^2 >= 0
    
    # Number of pairs: N*(N-1)/2
    n_pairs = N * (N - 1) // 2
    
    def pair_fun(x_flat):
        c_f = x_flat[:2*N].reshape(N, 2)
        r_f = x_flat[-1]
        # Compute all pairwise squared distances
        # Efficient way:
        diff = c_f[:, None, :] - c_f[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        # Extract upper triangle
        iu = np.triu_indices(N, k=1)
        d2_vals = d2[iu]
        return d2_vals - 4.0 * r_f * r_f
    
    def pair_jac(x_flat):
        c_f = x_flat[:2*N].reshape(N, 2)
        r_f = x_flat[-1]
        J = np.zeros((n_pairs, 2*N+1))
        
        iu = np.triu_indices(N, k=1)
        i_idx, j_idx = iu
        
        for k in range(n_pairs):
            i = i_idx[k]
            j = j_idx[k]
            # d/dx_i (d_ij^2 - 4r^2) = 2 * (c_i - c_j)
            # d/dx_j (d_ij^2 - 4r^2) = 2 * (c_j - c_i) = -2 * (c_i - c_j)
            # d/dr (d_ij^2 - 4r^2) = -8r
            
            diff_ij = c_f[i] - c_f[j]
            
            J[k, 2*i] = 2.0 * diff_ij[0]
            J[k, 2*i+1] = 2.0 * diff_ij[1]
            
            J[k, 2*j] = -2.0 * diff_ij[0]
            J[k, 2*j+1] = -2.0 * diff_ij[1]
            
            J[k, 2*N] = -8.0 * r_f
            
        return J
    
    cons.append({'type': 'ineq', 'fun': pair_fun, 'jac': pair_jac})
    
    return cons

def construct():
    t0 = time.time()
    DEADLINE = TIME_LIMIT - 5.0
    
    rng = np.random.default_rng(2024)
    
    best_c = None
    best_r = 0.0
    
    # Initial radius estimate
    # Hex packing density ~ 0.9069
    # Area per circle ~ pi r^2
    # 200 * pi r^2 <= 0.9069 * 1
    # r^2 <= 0.9069 / (200 * pi)
    # r <= sqrt(0.9069 / 628.3) ~ sqrt(0.00144) ~ 0.038
    # Known best is ~0.0366
    r_init = 0.035
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Hex grid centered
    s_hex = 2.0 * r_init * 1.01  # Slightly larger spacing to ensure feasibility
    c_hex = _initial_hex_grid(s_hex, 0.0, 0.0)
    if c_hex.shape[0] == N:
        configs.append(c_hex)
        
    # 2. Hex grid shifted
    c_hex2 = _initial_hex_grid(s_hex, 0.01, 0.01)
    if c_hex2.shape[0] == N:
        configs.append(c_hex2)
        
    # 3. Rectangular grid
    # Approx sqrt(200) ~ 14.14
    # 14x14 = 196, need 200. 15x14 = 210.
    # Let's try 15x14 grid, drop 10.
    nx, ny = 15, 14
    xs = np.linspace(-HALF + r_init, HALF - r_init, nx)
    ys = np.linspace(-HALF + r_init, HALF - r_init, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    if grid_pts.shape[0] >= N:
        # Select N points with max min-distance
        # Greedy
        selected = []
        min_dists = np.full(grid_pts.shape[0], np.inf)
        start = rng.choice(grid_pts.shape[0])
        selected.append(start)
        for _ in range(N - 1):
            last = grid_pts[selected[-1]]
            d = np.linalg.norm(grid_pts - last, axis=1)
            min_dists = np.minimum(min_dists, d)
            for s_idx in selected:
                min_dists[s_idx] = -1
            next_idx = np.argmax(min_dists)
            if min_dists[next_idx] < 0:
                break
            selected.append(next_idx)
        c_rect = grid_pts[selected]
        if c_rect.shape[0] == N:
            configs.append(c_rect)
            
    # 4. Random perturbation of hex
    if len(configs) > 0:
        c_rand = configs[0] + rng.normal(0, 0.005, (N, 2))
        c_rand = np.clip(c_rand, -HALF + r_init*0.5, HALF - r_init*0.5)
        configs.append(c_rand)
        
    if not configs:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        c_fallback = grid[:N]
        c_fallback = np.clip(c_fallback, -HALF + 0.03, HALF - 0.03)
        configs.append(c_fallback)

    # Run SLSQP on each config
    for c_init in configs:
        if time.time() > DEADLINE:
            break
            
        # Initial r: slightly less than current valid r of c_init to ensure feasibility
        r_current = _compute_radius(c_init)
        if r_current <= 0:
            # Try to fix c_init by scaling? Or just skip
            # If invalid, perturb
            c_init = np.clip(c_init, -HALF + 0.03, HALF - 0.03)
            r_current = _compute_radius(c_init)
            if r_current <= 0:
                continue
                
        # Start r slightly below current to be safe, let optimizer increase
        r_start = r_current * 0.99
        
        x0 = np.concatenate([c_init.ravel(), [r_start]])
        
        # Bounds
        # Centers: [-HALF, HALF] (constraints handle tighter bound based on r)
        # r: [0, 0.5]
        bounds = [(-HALF, HALF)] * (2*N) + [(0, 0.5)]
        
        # SLSQP options
        # Maxiter: 200 iterations should be enough for N=200 if initialized well
        # ftol: 1e-8
        # maxfun: 1000
        
        try:
            res = minimize(
                _slsqp_objective,
                x0,
                jac=_slsqp_grad,
                constraints=_slsqp_constraints(x0), # Note: constraints callables are stateless, but we pass x0 for potential future use? No, fun takes x.
                bounds=bounds,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success or res.fun < 0: # res.fun is -r, so smaller is better
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt = res.x[-1]
                
                # Verify
                r_valid = _compute_radius(c_opt)
                if r_valid > best_r:
                    best_r = r_valid
                    best_c = c_opt.copy()
                    
        except Exception as e:
            # If SLSQP fails, try to use the last valid point?
            # Or just skip
            pass

    # If no valid configuration found, fallback
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N]
        best_c = np.clip(best_c, -HALF + 0.03, HALF - 0.03)
    else:
        # Final clamp to ensure validity
        best_c = np.clip(best_c, -HALF + 1e-9, HALF - 1e-9)
        
    return best_c

if __name__ == "__main__":
    construct()
