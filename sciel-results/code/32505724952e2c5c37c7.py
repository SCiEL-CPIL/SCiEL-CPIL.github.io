"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _score(cents):
    """Compute the radius for a given set of centers. Higher is better."""
    # Centers must be in [-0.5, 0.5]^2
    if not np.all(np.isfinite(cents)):
        return -np.inf
    if np.any(cents < -0.5 - 1e-12) or np.any(cents > 0.5 + 1e-12):
        return -np.inf
    
    # Boundary constraint: distance to nearest wall
    # min_i min_k (0.5 - |c_ik|)
    dist_to_wall = np.min(np.minimum(np.abs(cents), 0.5 - np.abs(cents)))
    # Actually: 0.5 - max(|x|, |y|) for each point? No, min over coordinates.
    # dist_to_wall_i = min(0.5 - x_i, 0.5 + x_i, 0.5 - y_i, 0.5 + y_i)
    # Since x_i in [-0.5, 0.5], 0.5 - |x_i| is the distance to the vertical walls.
    # 0.5 - |y_i| is the distance to the horizontal walls.
    # So min_dist_to_wall = min(0.5 - max(|x|, |y|))? No.
    # It's min over i of (min(0.5 - |x_i|, 0.5 - |y_i|)).
    # Let's compute properly.
    d_x = 0.5 - np.abs(cents[:, 0])
    d_y = 0.5 - np.abs(cents[:, 1])
    min_wall = np.min(np.minimum(d_x, d_y))
    
    # Pairwise distances
    # Use cdist for efficiency
    d = cdist(cents, cents)
    # Set diagonal to inf to ignore self-distance
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    
    return min(min_pair, min_wall)

def _objective(cents):
    """Minimize negative score (or a proxy). 
    We want to maximize min(min_pair, min_wall).
    This is equivalent to maximizing t subject to all constraints >= t.
    Or simply minimizing -t where t is the bottleneck.
    However, -min is non-smooth.
    
    A common approach for packing is to minimize the sum of squared violations 
    of a target radius, or to use a soft-min.
    
    Let's try minimizing: -min(min_pair, min_wall) is hard.
    
    Alternative: Minimize (1/r^2) * sum of violations?
    
    Let's stick to a simple gradient-based approach on the "bottleneck" value.
    We can use a soft-min approximation:
    soft_min(a, b, k) = -1/k * log(exp(-k*a) + exp(-k*b))
    
    But we have many pairs.
    min_pair = min_{i<j} d_ij / 2
    min_wall = min_i w_i
    
    Let's define the objective as minimizing:
    J(c) = - (1/k) * log( sum_{i<j} exp(-k * d_ij/2) + sum_i exp(-k * w_i) )
    This approximates -min( min_pair, min_wall ).
    Minimizing J is maximizing the soft-min.
    
    However, k needs to be large for accuracy, but large k causes numerical issues.
    
    Another strategy: 
    1. Start with a good initial guess (e.g., grid or random).
    2. Use L-BFGS-B to maximize the "minimum distance" directly? 
       SciPy's minimize can handle non-smooth functions poorly.
       
    Let's try a different objective:
    Minimize the maximum "violation" if we target a radius r.
    But we don't know r.
    
    Let's use the "max-min" objective directly with a heuristic gradient.
    Or, simpler: 
    Minimize the sum of squared distances for pairs that are too close?
    
    Actually, a very effective method for circle packing is "repulsion" based.
    Minimize: sum_{i<j} (1/d_ij)^2 + sum_i (1/w_i)^2
    This pushes circles apart and away from walls.
    As the system reaches equilibrium, the minimum distance increases.
    
    Let's try this potential energy approach.
    E(c) = sum_{i<j} (1/d_ij - 1/r_target)^2 ? No, we don't know r_target.
    
    Let's just minimize:
    E(c) = sum_{i<j} 1/d_ij^2 + sum_i 1/w_i^2
    
    This is smooth everywhere (except where d=0 or w=0, which we avoid).
    The global minimum of this energy function for fixed N in a square often corresponds to a good packing.
    """
    # Calculate pairwise distances
    d = cdist(cents, cents)
    np.fill_diagonal(d, np.inf)
    
    # Avoid division by zero or very small numbers
    # If d is very small, 1/d^2 is huge, which is good for penalty.
    # But we need to be careful with overflow.
    
    # Wall distances
    d_x = 0.5 - np.abs(cents[:, 0])
    d_y = 0.5 - np.abs(cents[:, 1])
    w = np.minimum(d_x, d_y)
    
    # Penalty for small distances
    # Add a small epsilon to avoid inf
    eps = 1e-8
    d_safe = np.maximum(d, eps)
    w_safe = np.maximum(w, eps)
    
    pair_energy = np.sum(1.0 / (d_safe ** 2))
    wall_energy = np.sum(1.0 / (w_safe ** 2))
    
    return pair_energy + wall_energy

def _gradient(cents):
    """Calculate gradient of the energy function."""
    # d = cdist(cents, cents)
    # grad_pair = sum_{j != i} 2 * (c_i - c_j) / d_ij^4
    # grad_wall = derivative of 1/w_i^2
    
    # This is expensive to compute exactly. 
    # Let's use numerical gradient or rely on L-BFGS-B's internal numerical gradient?
    # L-BFGS-B with finite differences might be slow for 100 variables.
    # Let's try analytic gradient if possible, or just use finite differences with a good step.
    
    # Actually, let's just use finite differences in scipy, it's robust.
    # But we can provide the gradient for speed.
    
    # Pairwise gradient:
    # E_pair = sum_{i<j} d_ij^{-2}
    # dE_pair/dc_i = sum_{j != i} -2 * d_ij^{-3} * (c_i - c_j) / d_ij = -2 * sum_{j != i} (c_i - c_j) / d_ij^4
    
    d = cdist(cents, cents)
    np.fill_diagonal(d, np.inf)
    d_safe = np.maximum(d, 1e-8)
    
    # Vectorized gradient for pairs
    # diff = cents[:, None, :] - cents[None, :, :]
    # grad_pair = -2 * np.sum(diff / d_safe[:, :, None]**4, axis=1)
    # Note: d_safe is (N, N). diff is (N, N, 2).
    # We need to handle the diagonal (i=j) which is inf, so 1/inf^4 = 0.
    
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    d4 = d_safe[:, :, np.newaxis] ** 4
    # Avoid division by zero on diagonal (though d is inf there)
    # np.where(d_safe > 1e-7, 1.0/d4, 0.0)
    inv_d4 = np.zeros_like(d_safe)
    mask = d_safe > 1e-7
    inv_d4[mask] = 1.0 / (d_safe[mask] ** 4)
    
    grad_pair = -2.0 * np.sum(diff * inv_d4[:, :, np.newaxis], axis=1)
    
    # Wall gradient:
    # E_wall = sum_i (min(d_xi, d_yi))^{-2}
    # Let w_i = min(d_xi, d_yi)
    # dE_wall/dc_i = -2 * w_i^{-3} * dw_i/dc_i
    
    d_x = 0.5 - np.abs(cents[:, 0])
    d_y = 0.5 - np.abs(cents[:, 1])
    w = np.minimum(d_x, d_y)
    w_safe = np.maximum(w, 1e-8)
    
    grad_wall = np.zeros_like(cents)
    
    # For each point i:
    # if d_xi < d_yi, w_i = d_xi = 0.5 - |x_i|
    #   if x_i > 0, dw/dx = -1, dw/dy = 0
    #   if x_i < 0, dw/dx = 1, dw/dy = 0
    # if d_yi < d_xi, w_i = d_yi = 0.5 - |y_i|
    #   if y_i > 0, dw/dy = -1, dw/dx = 0
    #   if y_i < 0, dw/dy = 1, dw/dx = 0
    
    # dE/dx_i = -2 * w_i^{-3} * dw_i/dx_i
    # dE/dy_i = -2 * w_i^{-3} * dw_i/dy_i
    
    coeff = -2.0 / (w_safe ** 3)
    
    # dw/dx
    dw_dx = np.zeros(N)
    dw_dy = np.zeros(N)
    
    # Case 1: d_x < d_y (x is closer to wall)
    mask_x = d_x < d_y
    # For these points, w = d_x
    # dw_dx = -sign(x)
    dw_dx[mask_x] = -np.sign(cents[mask_x, 0])
    # dw_dy = 0
    
    # Case 2: d_y <= d_x (y is closer to wall)
    mask_y = d_y <= d_x
    # For these points, w = d_y
    # dw_dy = -sign(y)
    dw_dy[mask_y] = -np.sign(cents[mask_y, 1])
    # dw_dx = 0
    
    grad_wall[:, 0] = coeff * dw_dx
    grad_wall[:, 1] = coeff * dw_dy
    
    return grad_pair + grad_wall

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # Initial guess: Grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    init_cents = grid[:N].copy()
    
    # Add some jitter to break symmetry
    np.random.seed(42)
    init_cents += np.random.normal(0, 0.01, init_cents.shape)
    
    # Clip to bounds
    init_cents = np.clip(init_cents, -0.49, 0.49)
    
    best_cents = init_cents.copy()
    best_score = _score(best_cents)
    
    # Optimization
    # We will run L-BFGS-B to minimize the energy.
    # We might need to restart with different initializations if we get stuck in local minima.
    
    def objective_and_grad(x):
        x = x.reshape(-1, 2)
        val = _objective(x)
        grad = _gradient(x)
        return val, grad.flatten()
    
    # Bounds for L-BFGS-B
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Run optimization
    # maxiter should be high, but we check time.
    # We can use a callback to check time.
    
    def callback(xk):
        nonlocal best_cents, best_score
        current_score = _score(xk.reshape(-1, 2))
        if current_score > best_score:
            best_score = current_score
            best_cents = xk.reshape(-1, 2).copy()
        return time.time() > deadline
    
    res = minimize(
        objective_and_grad,
        init_cents.flatten(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8},
        callback=callback
    )
    
    # Check result
    if res.x is not None:
        res_cents = res.x.reshape(-1, 2)
        res_score = _score(res_cents)
        if res_score > best_score:
            best_score = res_score
            best_cents = res_cents
    
    # If time remains, try a few more random restarts
    while time.time() < deadline - 5.0:
        # Random initialization
        np.random.seed(int(time.time() * 1000) % 10000)
        rand_cents = np.random.uniform(-0.45, 0.45, (N, 2))
        rand_cents = np.clip(rand_cents, -0.49, 0.49)
        
        res = minimize(
            objective_and_grad,
            rand_cents.flatten(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8},
            callback=callback
        )
        
        if res.x is not None:
            res_cents = res.x.reshape(-1, 2)
            res_score = _score(res_cents)
            if res_score > best_score:
                best_score = res_score
                best_cents = res_cents
    
    # Final polish: try to increase radius by local moves?
    # The energy minimization should have done a good job.
    # Ensure validity
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents

# EVOLVE-BLOCK-END
