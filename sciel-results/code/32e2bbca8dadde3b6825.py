import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 50
KNOWN_BEST_R = 0.1258254895304

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Helper: Calculate the score (radius) for a set of points
    def calc_radius(pts):
        # Check validity
        if pts.shape != (N, 2):
            return -1.0
        if np.any(~np.isfinite(pts)):
            return -1.0
        norms = np.linalg.norm(pts, axis=1)
        if np.any(norms > 1.0):
            return -1.0
        # Pairwise distances
        d = cdist(pts, pts)
        # Mask diagonal
        d_min = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                if d[i, j] < d_min:
                    d_min = d[i, j]
        b_min = np.min(1.0 - norms)
        r = min(d_min / 2.0, b_min)
        return r

    # Helper: Clip points to be strictly inside unit circle
    def clip_to_circle(pts, margin=1e-10):
        pts = pts.copy()
        norms = np.linalg.norm(pts, axis=1)
        valid = norms > 0
        pts[valid] = pts[valid] * (1.0 - margin) / norms[valid, np.newaxis]
        # If any norm is 0, leave as is (origin is fine)
        return pts

    # Helper: Generate hexagonal grid initializations
    def get_hex_grid(spacing, rotation=0.0):
        points = []
        k = int(1.2 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                # Rotate
                xr = x * np.cos(rotation) - y * np.sin(rotation)
                yr = x * np.sin(rotation) + y * np.cos(rotation)
                if xr**2 + yr**2 <= 1.0:
                    points.append([xr, yr])
        points = np.array(points)
        if len(points) < N:
            return None
        # Select N points closest to origin
        dists = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(dists)[:N]
        return points[indices]

    # Helper: Generate random initialization
    def get_random_init():
        pts = np.random.rand(N, 2) * 2.0 - 1.0
        norms = np.linalg.norm(pts, axis=1)
        valid = norms > 0
        pts[valid] = pts[valid] / norms[valid, np.newaxis] * np.random.rand(N) * 0.95
        return pts

    # Helper: Objective and gradient for fixed R (penalty method)
    def objective_and_grad(c, R):
        c = c.reshape(N, 2)
        # Pairwise distances
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        
        # Boundary distances
        r_norm = np.linalg.norm(c, axis=1)
        b_dist = 1.0 - r_norm
        
        # Violations
        pair_viol = np.maximum(0.0, 2.0 * R - d)
        bound_viol = np.maximum(0.0, R - b_dist)
        
        # Energy
        E = np.sum(pair_viol**2) + np.sum(bound_viol**2)
        
        # Gradient
        grad = np.zeros_like(c)
        
        # Pairwise gradient
        coeff = np.zeros((N, N))
        mask = d < np.inf
        coeff[mask] = 2.0 * pair_viol[mask] / d[mask]
        
        sum_coeff = np.sum(coeff, axis=1)
        grad_pair = c * sum_coeff[:, np.newaxis] - (coeff @ c)
        
        # Boundary gradient
        grad_bound = np.zeros_like(c)
        valid_bound = r_norm > 1e-8
        grad_bound[valid_bound] = (2.0 * bound_viol[valid_bound] * c[valid_bound]) / r_norm[valid_bound, np.newaxis]
        
        grad = grad_pair + grad_bound
        
        return E, grad

    # Helper: Soft-min objective and gradient for maximizing r
    # r = min(d_ij/2, b_i)
    # We want to maximize r.
    # Let f_i = b_i, g_ij = d_ij/2.
    # r = min(f_i, g_ij)
    # Use log-sum-exp approximation for min: -logsumexp(-x) approximates min(x)
    # But we need gradient.
    # Alternatively, use the epigraph form or just a smooth approximation.
    # Let's stick to the penalty method for fixed R, but use a smoother objective for the "max r" part?
    # The directive suggests smooth surrogate.
    # Let's try to maximize a soft-min of the constraints.
    # h(c) = -logsumexp(-[b_1, ..., b_N, d_12/2, ...])
    # This is smooth and concave (since min is concave, -logsumexp is convex? No, min is concave, -min is convex. logsumexp is convex. -logsumexp(-x) is concave. We want to maximize a concave function, which is a convex optimization problem if the feasible region is convex. But here the feasible region is not convex. However, local optimization can work.)
    
    def soft_min_obj_and_grad(c):
        c = c.reshape(N, 2)
        r_norm = np.linalg.norm(c, axis=1)
        b = 1.0 - r_norm
        
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        g = d / 2.0
        
        # Collect all constraints into a single vector
        # b has N elements, g has N*(N-1)/2 elements
        # To save memory/time, we can compute the logsumexp in parts or just use the full vector if N is small.
        # N=50, pairs=1225. Total 1275 elements. Fine.
        
        # We want to maximize r_soft = -logsumexp(-x) / beta
        # Let's use a temperature beta.
        beta = 50.0  # Higher beta -> sharper min
        
        # x = [b, g]
        # For efficiency, let's compute the gradient directly without forming the huge vector if possible, 
        # but for N=50 it's small enough.
        
        # Actually, let's just use the penalty method for fixed R as it's robust.
        # The "soft-min" idea is good for a global search, but local optimization with fixed R is standard.
        # Let's try a different approach: maximize the minimum distance directly using a smooth approximation.
        
        # Let's return the penalty method stuff for now, as it's proven.
        # But the directive asks to investigate improvements.
        # Let's try to use the soft-min as the objective for a local search.
        
        # x = np.concatenate([b, g[upper_triangle_indices]])
        # This is getting complicated to vectorize the gradient of logsumexp.
        
        # Let's stick to the fixed-R penalty method but improve the search strategy.
        # 1. Better initialization.
        # 2. More restarts with different random seeds/rotations.
        # 3. Bisection on R.
        
        return None, None

    # We will use the fixed-R penalty method.
    def obj_fixed_R(c, R):
        E, _ = objective_and_grad(c, R)
        return E

    def jac_fixed_R(c, R):
        _, g = objective_and_grad(c, R)
        return g.flatten()

    best_points = None
    best_r = 0.0

    # Initializations
    inits = []
    
    # Hex grids with various spacings and rotations
    for s in np.linspace(0.1, 0.25, 20):
        for rot in [0.0, np.pi/6, np.pi/4, np.pi/3]:
            pts = get_hex_grid(s, rot)
            if pts is not None:
                inits.append(pts)
                
    # Random inits
    for _ in range(10):
        inits.append(get_random_init())

    # Sort inits by initial radius (higher is better) to try good ones first
    init_radii = [calc_radius(p) for p in inits]
    # Filter out invalid
    valid_inits = [(p, r) for p, r in zip(inits, init_radii) if r > 0]
    valid_inits.sort(key=lambda x: x[1], reverse=True)
    
    # If no valid inits, use a fallback
    if not valid_inits:
        valid_inits = [(get_random_init(), 0.0)]

    # For each initialization, try to optimize
    # We will do a bisection search on R for each init? That's expensive.
    # Better: For each init, do a local search to maximize r directly?
    # Or: For each init, find the best r achievable by a local search, then update global best.
    
    # Let's try a different strategy:
    # 1. For each init, run a local search that maximizes a soft-min of the constraints.
    #    This avoids the need for bisection.
    #    Objective: maximize -logsumexp(-[b_i, d_ij/2]) / beta
    #    This is equivalent to maximizing the "soft" minimum.
    
    def soft_min_objective(c):
        c = c.reshape(N, 2)
        r_norm = np.linalg.norm(c, axis=1)
        b = 1.0 - r_norm
        
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        g = d / 2.0
        
        # Combine b and g
        # To avoid large memory, we can compute logsumexp in chunks or just concatenate.
        # N=50, b has 50, g has 1225. Total 1275.
        x = np.concatenate([b, g.ravel()])
        
        beta = 100.0
        # max x for numerical stability
        x_max = np.max(x)
        # If x_max is very large, logsumexp might overflow? No, we subtract x_max.
        # logsumexp(x) = x_max + log(sum(exp(x - x_max)))
        # We want -logsumexp(-x) = -( -x_max + log(sum(exp(-x - (-x_max)))) )
        # = x_max - log(sum(exp(-(x - x_max))))
        
        exp_terms = np.exp(-(x - x_max))
        log_sum = np.log(np.sum(exp_terms))
        val = x_max - log_sum
        
        return -val / beta  # We want to minimize -val, so maximize val

    def soft_min_gradient(c):
        c = c.reshape(N, 2)
        r_norm = np.linalg.norm(c, axis=1)
        b = 1.0 - r_norm
        
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        g = d / 2.0
        
        x = np.concatenate([b, g.ravel()])
        beta = 100.0
        
        x_max = np.max(x)
        exp_terms = np.exp(-(x - x_max))
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp  # softmax weights for -x
        
        # Gradient of val = x_max - log(sum(exp(-(x - x_max))))
        # dval/dx_i = -weights_i (approximately, ignoring x_max dependence for simplicity, 
        # but x_max depends on x. Let's be precise.)
        # Actually, let's just use numerical gradient or a simpler approximation.
        # For L-BFGS-B, finite difference gradient is slower but safer.
        # Let's try to derive it properly.
        # f(x) = -logsumexp(-x)
        # df/dx_i = p_i, where p_i = exp(-x_i) / sum(exp(-x_j))
        # So d(-f)/dx_i = -p_i.
        # We want to minimize -f, so gradient is -p_i.
        
        # But x is a function of c.
        # x_b_i = 1 - ||c_i||
        # x_g_ij = ||c_i - c_j|| / 2
        
        # This is complex. Let's just use finite differences for the soft-min objective.
        # It's slower but robust.
        return None

    # Since soft_min_gradient is hard to vectorize efficiently without bugs,
    # let's stick to the fixed-R penalty method which has a known analytic gradient.
    # But we can use the soft-min idea to pick a good R for the bisection?
    # Or just use multiple restarts with the fixed-R method.
    
    # Strategy:
    # 1. For each of the top 5 initializations:
    #    a. Estimate a good R range.
    #    b. Perform bisection on R using L-BFGS-B with the penalty objective.
    #    c. Update global best.
    
    top_inits = valid_inits[:5]
    
    for init_pts, init_r in top_inits:
        if time.time() > deadline - 20:
            break
            
        # Bisection setup
        R_low = init_r
        R_high = min(0.126, init_r + 0.05)  # Don't go too high initially
        
        current_best_pts = init_pts.copy()
        current_best_r = init_r
        
        # We will try to increase R
        for _ in range(15):  # Max bisection steps per init
            if time.time() > deadline - 10:
                break
                
            R_mid = (R_low + R_high) / 2.0
            
            # Optimize for R_mid
            x0 = current_best_pts.copy()
            
            try:
                res = minimize(
                    obj_fixed_R, x0, args=(R_mid,),
                    jac=lambda c: jac_fixed_R(c, R_mid),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                E_final, _ = objective_and_grad(res.x, R_mid)
                if E_final < 1e-6:
                    R_low = R_mid
                    current_best_pts = res.x.copy()
                    current_best_r = R_mid
                else:
                    R_high = R_mid
            except Exception:
                R_high = R_mid
        
        # Update global best
        r_check = calc_radius(current_best_pts)
        if r_check > best_r:
            best_r = r_check
            best_points = current_best_pts.copy()
            
    # If we still have time, try a few more random restarts
    if best_points is None or time.time() < deadline - 30:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            init_pts = get_random_init()
            init_r = calc_radius(init_pts)
            if init_r < best_r:
                continue
                
            R_low = init_r
            R_high = min(0.126, init_r + 0.05)
            current_best_pts = init_pts.copy()
            current_best_r = init_r
            
            for _ in range(10):
                if time.time() > deadline - 5:
                    break
                R_mid = (R_low + R_high) / 2.0
                x0 = current_best_pts.copy()
                try:
                    res = minimize(
                        obj_fixed_R, x0, args=(R_mid,),
                        jac=lambda c: jac_fixed_R(c, R_mid),
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    E_final, _ = objective_and_grad(res.x, R_mid)
                    if E_final < 1e-6:
                        R_low = R_mid
                        current_best_pts = res.x.copy()
                        current_best_r = R_mid
                    else:
                        R_high = R_mid
                except Exception:
                    R_high = R_mid
                    
            r_check = calc_radius(current_best_pts)
            if r_check > best_r:
                best_r = r_check
                best_points = current_best_pts.copy()

    if best_points is None:
        # Fallback
        best_points = get_random_init()
        best_points = clip_to_circle(best_points)
        
    # Final clip to ensure validity
    best_points = clip_to_circle(best_points)
    
    return best_points
