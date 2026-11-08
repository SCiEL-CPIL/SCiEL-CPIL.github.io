import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import itertools

N = 14
TRI_INDICES = np.array(list(itertools.combinations(range(N), 3)))
I_IDX = TRI_INDICES[:, 0]
J_IDX = TRI_INDICES[:, 1]
K_IDX = TRI_INDICES[:, 2]
M = len(TRI_INDICES)

def _score(pts):
    """Compute min triangle area / convex hull area."""
    if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
        return -1e10
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
    except Exception:
        return -1e10
    if h_area < 1e-8:
        return -1e10
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    min_a = np.min(np.abs(cross)) * 0.5
    return min_a / h_area

def _soft_log_min_score_and_grad(pts, beta):
    """
    Computes the log-soft-min of triangle areas minus log(hull area) and its gradient.
    Objective to maximize: log(soft_min(areas)) - log(hull_area)
    This is equivalent to maximizing soft_min(areas) / hull_area.
    Returns: (value, gradient)
    """
    pi = pts[I_IDX]
    pj = pts[J_IDX]
    pk = pts[K_IDX]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min approximation: (1/beta) * log(sum(exp(-beta * areas)))
    # To avoid overflow, we use the log-sum-exp trick
    # max_a = np.max(areas)
    # log_sum = np.log(np.sum(np.exp(-beta * (areas - max_a)))) - beta * max_a
    # Actually, we want min, so we use -beta * areas.
    # Let z = -beta * areas.
    # log_sum_exp(z) = max(z) + log(sum(exp(z - max(z))))
    # z_max = -beta * min(areas)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return -1e10, np.zeros_like(pts)
        
    z = -beta * areas
    z_max = np.max(z)
    log_sum_exp = z_max + np.log(np.sum(np.exp(z - z_max)))
    log_soft_min = (1.0 / beta) * log_sum_exp
    
    # Weights for gradient
    # w_k = exp(z_k) / sum(exp(z_k))
    # This is the softmax of z.
    weights = np.exp(z - z_max)
    weights /= np.sum(weights)
    
    # Gradient of log_soft_min w.r.t. pts
    # d/dp log_soft_min = (1/beta) * d/dp log_sum_exp
    # = (1/beta) * sum( w_k * d/dp z_k )
    # z_k = -beta * A_k
    # d/dp z_k = -beta * dA_k/dp
    # So d/dp log_soft_min = sum( w_k * (-dA_k/dp) ) = - sum( w_k * grad_A_k )
    
    # Initialize gradient accumulator
    grad_soft = np.zeros((N, 2))
    
    # Compute gradients of areas
    # For each triangle k, grad_A_k is non-zero only for points i, j, k
    # A_k = 0.5 * sign(cross_k) * cross_k
    # grad_A_k = 0.5 * sign(cross_k) * grad_cross_k
    
    signs = np.sign(cross)
    # If cross is 0, sign is 0, which is fine (gradient is 0 or undefined, we pick 0)
    
    # We need grad_cross for each triangle
    # grad_cross_i = (y_j - y_k, x_k - x_j)
    # grad_cross_j = (y_k - y_i, x_i - x_k)
    # grad_cross_k = (y_i - y_j, x_j - x_i)
    
    # Vectorized calculation
    # pi, pj, pk are (M, 2)
    xi, yi = pi[:, 0], pi[:, 1]
    xj, yj = pj[:, 0], pj[:, 1]
    xk, yk = pk[:, 0], pk[:, 1]
    
    gxi = yj - yk
    gyi = xk - xj
    
    gxj = yk - yi
    gyj = xi - xk
    
    gxk = yi - yj
    gyk = xj - xi
    
    # grad_A_k = 0.5 * sign * grad_cross
    gxi *= 0.5 * signs
    gyi *= 0.5 * signs
    gxj *= 0.5 * signs
    gyj *= 0.5 * signs
    gxk *= 0.5 * signs
    gyk *= 0.5 * signs
    
    # Accumulate weighted gradients
    # grad_soft = - sum( w_k * grad_A_k )
    # So we add -w_k * g to grad_soft at indices i, j, k
    
    # Using np.add.at for efficient accumulation
    np.add.at(grad_soft, (I_IDX, 0), -weights * gxi)
    np.add.at(grad_soft, (I_IDX, 1), -weights * gyi)
    np.add.at(grad_soft, (J_IDX, 0), -weights * gxj)
    np.add.at(grad_soft, (J_IDX, 1), -weights * gyj)
    np.add.at(grad_soft, (K_IDX, 0), -weights * gxk)
    np.add.at(grad_soft, (K_IDX, 1), -weights * gyk)
    
    # Now compute hull area and its gradient
    try:
        hull = ConvexHull(pts)
        h_area = hull.volume
        if h_area < 1e-8:
             return -1e10, np.zeros_like(pts)
        hull_indices = hull.vertices
    except Exception:
        return -1e10, np.zeros_like(pts)
    
    # Gradient of Hull Area
    # A_hull = 0.5 * sum( p_i x p_{i+1} )
    # dA/dx_i = 0.5 * (y_{i+1} - y_{i-1})
    # dA/dy_i = 0.5 * (x_{i-1} - x_{i+1})
    
    grad_hull = np.zeros((N, 2))
    m_h = len(hull_indices)
    if m_h < 3:
        return -1e10, np.zeros_like(pts)
        
    # Get coordinates of hull vertices
    # hull_indices are indices into pts
    # We need to map local hull index to global point index
    for i in range(m_h):
        idx_curr = hull_indices[i]
        idx_next = hull_indices[(i + 1) % m_h]
        idx_prev = hull_indices[(i - 1) % m_h]
        
        x_curr, y_curr = pts[idx_curr]
        x_next, y_next = pts[idx_next]
        x_prev, y_prev = pts[idx_prev]
        
        grad_hull[idx_curr, 0] = 0.5 * (y_next - y_prev)
        grad_hull[idx_curr, 1] = 0.5 * (x_prev - x_next)
        
    # Value: log_soft_min - log(h_area)
    value = log_soft_min - np.log(h_area)
    
    # Gradient of value: grad_soft - grad_hull / h_area
    # We want to maximize value, so we minimize -value.
    # Gradient of -value: -grad_soft + grad_hull / h_area
    # Wait, the optimizer minimizes.
    # If we return (value, grad_value), minimize will use -grad_value?
    # No, scipy minimize expects (f, grad_f) where grad_f is gradient of f.
    # If we want to maximize f, we should minimize -f.
    # So we should return (-value, -grad_value).
    
    grad_value = grad_soft - grad_hull / h_area
    
    return -value, -grad_value

def _initial_regular():
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _initial_star(rng):
    """Two rings."""
    n_outer = N // 2
    n_inner = N - n_outer
    angles_out = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_in = np.linspace(0, 2 * np.pi, n_inner, endpoint=False) + np.pi / n_inner
    r_out = 1.0
    r_in = 0.6
    x = np.concatenate([r_out * np.cos(angles_out), r_in * np.cos(angles_in)])
    y = np.concatenate([r_out * np.sin(angles_out), r_in * np.sin(angles_in)])
    pts = np.column_stack([x, y])
    idx = rng.permutation(N)
    return pts[idx]

def _initial_random(rng):
    # Points in unit disk
    r = np.sqrt(rng.uniform(0, 1, N))
    theta = rng.uniform(0, 2 * np.pi, N)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_pts = _initial_regular()
    best_score = _score(best_pts)
    
    def time_ok():
        return time.time() < deadline
    
    # Initializations
    inits = [
        _initial_regular(),
        _initial_star(rng),
        _initial_random(rng),
        _initial_star(rng),
        _initial_random(rng)
    ]
    
    # Annealing schedule for beta
    # Start with small beta (smooth) and increase to large beta (sharp)
    betas = [10, 50, 100, 200, 500, 1000]
    
    # For each initialization, run the annealing process
    for init_pts in inits:
        if not time_ok():
            break
            
        current_pts = init_pts.copy()
        
        # Run L-BFGS-B for each beta
        for beta in betas:
            if not time_ok():
                break
                
            try:
                res = minimize(
                    _soft_log_min_score_and_grad,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    jac=True, # We return gradient
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
                )
                
                # Check if we improved
                # The function returns -value, so lower is better for the minimizer
                # We want to maximize value, so we check the actual score
                candidate_pts = res.x.reshape(N, 2)
                score = _score(candidate_pts)
                
                if score > best_score:
                    best_score = score
                    best_pts = candidate_pts
                    current_pts = candidate_pts
                else:
                    # If it didn't improve globally, we might still want to continue from this point
                    # if it improved locally, but let's stick to global best to avoid getting stuck
                    # in a local minimum that is worse than the current best.
                    # However, for annealing, we usually continue from the result of the previous step.
                    # Let's update current_pts to the result of the minimization, 
                    # even if it didn't beat the global best, because it might be a better 
                    # starting point for the next (sharper) beta.
                    current_pts = candidate_pts
                    
            except Exception:
                pass
                
    # Final polish with Nelder-Mead on the best candidate found
    if time_ok():
        try:
            res = minimize(
                lambda x: -_score(x.reshape(N, 2)),
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            candidate_pts = res.x.reshape(N, 2)
            score = _score(candidate_pts)
            if score > best_score:
                best_score = score
                best_pts = candidate_pts
        except Exception:
            pass

    return best_pts
