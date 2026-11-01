"""
Strategy: Multi-start simulated annealing with a smooth soft-minimum objective,
followed by L-BFGS polishing and a radius push phase.

This is materially different from the binary-search feasibility approach:
- Instead of testing feasibility for a fixed radius, we directly maximize a
  smooth surrogate for the minimum distance (soft-min of pairwise distances
  and wall distances).
- We use simulated annealing with vectorized updates to explore the landscape
  broadly and escape local minima.
- We then polish with L-BFGS on the same smooth objective.
- Finally, we do a "push" phase: take the best configuration, slightly increase
  the effective radius target, and re-optimize to see if we can squeeze out more.
- Multiple random restarts with different initial configurations (hex, grid,
  random) are used to improve robustness.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin before 150s kill

def _compute_radius(cents):
    """Compute the packing radius for a given set of centers."""
    n = len(cents)
    if n == 0:
        return 0.0
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _soft_min_obj(x, alpha=50.0):
    """
    Smooth objective: negative of the soft-minimum of all pairwise distances/2
    and wall distances. Higher alpha = sharper min approximation.
    """
    c = x.reshape(N, 2)
    
    # Wall distances (distance from each center to nearest wall)
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Combine all constraints into one array
    # Wall constraints: 2N values
    # Pair constraints: N*(N-1)/2 values (upper triangle)
    wall_vals = np.concatenate([wx, wy])
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    # Soft minimum: -log(sum(exp(-alpha * vals))) / alpha
    # For numerical stability, subtract max
    m = np.max(all_vals)
    log_sum = np.log(np.sum(np.exp(-alpha * (all_vals - m))))
    soft_min = m - log_sum / alpha
    
    return -soft_min

def _soft_min_grad(x, alpha=50.0):
    """
    Gradient of the soft-min objective.
    """
    c = x.reshape(N, 2)
    grad = np.zeros_like(c)
    
    # Wall distances
    wx = 0.5 - np.abs(c[:, 0])
    wy = 0.5 - np.abs(c[:, 1])
    
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    wall_vals = np.concatenate([wx, wy])
    iu = np.triu_indices(N, k=1)
    pair_vals = dists[iu]
    
    all_vals = np.concatenate([wall_vals, pair_vals])
    
    m = np.max(all_vals)
    exp_vals = np.exp(-alpha * (all_vals - m))
    sum_exp = np.sum(exp_vals)
    
    # Weights for soft min
    weights = exp_vals / sum_exp
    
    # Gradient of soft_min w.r.t. each value
    # d(soft_min)/d(val_k) = -weights[k]
    # So d(-soft_min)/d(val_k) = weights[k]
    
    # Wall x gradients
    # val_k = 0.5 - |c_i,x| for i in 0..N-1
    # d(val_k)/d(c_i,x) = -sign(c_i,x)
    # So contribution to grad[i,0] = weights[k] * (-sign(c_i,x))
    wx_w = weights[:N]
    wy_w = weights[N:2*N]
    
    # For x: val = 0.5 - |x|, d(val)/dx = -sign(x)
    # grad[i,0] += wx_w[i] * (-sign(c[i,0]))
    # But wait: objective is -soft_min, so we want to maximize soft_min.
    # d(-soft_min)/d(c_i,x) = -d(soft_min)/d(c_i,x)
    # d(soft_min)/d(c_i,x) = sum_k weights[k] * d(val_k)/d(c_i,x)
    # For wall x: d(val_k)/d(c_i,x) = -sign(c_i,x)
    # So d(-soft_min)/d(c_i,x) = -sum_k weights[k] * d(val_k)/d(c_i,x)
    # = -wx_w[i] * (-sign(c_i,x)) = wx_w[i] * sign(c_i,x)
    
    # Actually let me re-derive. Objective f = -soft_min.
    # soft_min = m - log(sum(exp(-alpha*(all_vals - m))))/alpha
    # df/d(all_vals_k) = -d(soft_min)/d(all_vals_k)
    # d(soft_min)/d(all_vals_k) = -weights[k] (where weights = exp(-alpha*(v-m))/sum)
    # So df/d(all_vals_k) = weights[k]
    
    # Now chain rule:
    # For wall x of circle i: all_vals[i] = 0.5 - |c_i,x|
    # d(all_vals[i])/d(c_i,x) = -sign(c_i,x)
    # df/dc_i,x = df/d(all_vals[i]) * d(all_vals[i])/dc_i,x = weights[i] * (-sign(c_i,x))
    
    # For wall y of circle i: all_vals[N+i] = 0.5 - |c_i,y|
    # d(all_vals[N+i])/d(c_i,y) = -sign(c_i,y)
    # df/dc_i,y = weights[N+i] * (-sign(c_i,y))
    
    # For pair (i,j) with i<j: all_vals[2N + idx] = dists[i,j]/2 = ||c_i - c_j||/2
    # d(all_vals)/dc_i = (c_i - c_j) / (2 * ||c_i - c_j||) = (c_i - c_j) / (2 * dists_full[i,j])
    # where dists_full[i,j] = 2 * dists[i,j]
    # Actually dists[i,j] here is already divided by 2, so full distance = 2*dists[i,j]
    # d(||c_i-c_j||/2)/dc_i = (c_i - c_j) / (2 * ||c_i-c_j||)
    
    # Let's use the full distance matrix for clarity
    diff_full = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_full = np.sqrt(np.sum(diff_full**2, axis=2))
    np.fill_diagonal(dist_full, 1e-10)
    
    # Pair values in all_vals correspond to upper triangle
    # For pair (i,j), i<j: value = dist_full[i,j]/2
    # d(value)/dc_i = (c_i - c_j) / (2 * dist_full[i,j])  ... wait
    # d(||c_i - c_j|| / 2) / dc_i = (1/2) * (c_i - c_j) / ||c_i - c_j||
    
    # df/dc_i = sum over all k involving i: weights[k] * d(all_vals[k])/dc_i
    
    # Wall contributions
    grad[:, 0] -= wx_w * np.sign(c[:, 0])
    grad[:, 1] -= wy_w * np.sign(c[:, 1])
    
    # Pair contributions
    # For each i, sum over j != i:
    # If i < j: weight = weights[2N + pair_idx(i,j)]
    #   d(all_vals)/dc_i = (c_i - c_j) / (2 * dist_full[i,j])
    #   d(all_vals)/dc_j = (c_j - c_i) / (2 * dist_full[i,j])
    
    # Let's build a weight matrix for pairs
    pair_w = np.zeros((N, N))
    pair_idx = 0
    for i in range(N):
        for j in range(i+1, N):
            w = weights[2*N + pair_idx]
            pair_w[i, j] = w
            pair_w[j, i] = w
            pair_idx += 1
    
    # Gradient from pairs:
    # grad[i] += sum_j pair_w[i,j] * (c_i - c_j) / (2 * dist_full[i,j])
    # Note: this is for df/dc_i where f = -soft_min
    # df/d(all_vals_k) = weights[k], and d(all_vals_k)/dc_i for pair (i,j) is (c_i-c_j)/(2*dist_full[i,j])
    
    # Vectorized:
    # For each i: grad[i] = sum_j pair_w[i,j] * (c_i - c_j) / (2 * dist_full[i,j])
    # = sum_j pair_w[i,j] / (2 * dist_full[i,j]) * (c_i - c_j)
    
    # Let coeff[i,j] = pair_w[i,j] / (2 * dist_full[i,j])
    coeff = pair_w / (2.0 * dist_full)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_pairs[i] = sum_j coeff[i,j] * (c_i - c_j)
    # = c_i * sum_j coeff[i,j] - sum_j coeff[i,j] * c_j
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)  # (N, 1)
    grad_pairs = c * row_sum - coeff @ c  # (N, 2)
    
    grad += grad_pairs
    
    return grad.flatten()

def _hex_init(n_rows=None):
    """Create a hexagonal packing arrangement."""
    if n_rows is None:
        n_rows = int(np.round(np.sqrt(N * 2 / np.sqrt(3))))
    
    best_cents = None
    best_r = 0
    
    for nr in range(4, 12):
        base = N // nr
        extra = N % nr
        row_counts = [base + (1 if i < extra else 0) for i in range(nr)]
        
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (nr - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        
        if r <= best_r:
            continue
        
        centers = []
        y_spacing = r * np.sqrt(3.0)
        total_y_span = (nr - 1) * y_spacing
        y_start = -0.5 + r + total_y_span / 2.0
        
        for row_idx in range(nr):
            rc = row_counts[row_idx]
            y = y_start + row_idx * y_spacing
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = r if (row_idx % 2 == 1) else 0.0
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        
        centers = np.array(centers[:N])
        r_actual = _compute_radius(centers)
        
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
    
    return best_cents

def _grid_init():
    """Create a simple grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N]

def _random_init():
    """Random initialization inside the square with some margin."""
    margin = 0.05
    return np.random.uniform(-0.5 + margin, 0.5 - margin, size=(N, 2))

def _simulated_annealing(x0, n_steps=5000, t_start=0.01, t_end=0.0001, alpha=50.0, time_limit=60.0):
    """
    Vectorized simulated annealing to maximize the soft-min objective.
    """
    x = x0.copy()
    current_obj = _soft_min_obj(x, alpha)
    best_x = x.copy()
    best_obj = current_obj
    
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        t = t_start * (t_end / t_start) ** (step / n_steps)
        
        # Generate random perturbation
        dx = np.random.normal(0, t, size=2 * N)
        x_new = x + dx
        
        # Clip to bounds
        x_new = np.clip(x_new, -0.5, 0.5)
        
        # Evaluate
        new_obj = _soft_min_obj(x_new, alpha)
        delta = new_obj - current_obj
        
        # Accept if better or with probability exp(delta/t)
        if delta > 0 or np.random.random() < np.exp(delta / max(t, 1e-10)):
            x = x_new
            current_obj = new_obj
            if current_obj > best_obj:
                best_obj = current_obj
                best_x = x.copy()
    
    return best_x, best_obj

def _lbfgs_polish(x0, alpha=50.0, time_limit=20.0):
    """Polish the configuration using L-BFGS-B on the soft-min objective."""
    start_time = time.time()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    try:
        result = minimize(
            _soft_min_obj,
            x0,
            args=(alpha,),
            jac=_soft_min_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 30}
        )
        
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            return x0
        
        return result.x
    except Exception:
        return x0

def _push_phase(x0, alpha=50.0, n_pushes=3, time_limit=20.0):
    """
    Push phase: try to improve by slightly increasing the effective radius.
    This works by re-optimizing with a higher alpha (sharper min) to focus
    on the tightest constraints.
    """
    x = x0.copy()
    start_time = time.time()
    
    for push in range(n_pushes):
        if time.time() - start_time > time_limit:
            break
        
        # Increase alpha to focus on tighter constraints
        alpha_push = alpha * (1.5 ** push)
        
        x = _lbfgs_polish(x, alpha=alpha_push, time_limit=time_limit / (n_pushes + 1))
    
    return x

def construct():
    start_time = time.time()
    
    # Generate initial configurations
    inits = []
    inits.append(_hex_init())
    inits.append(_grid_init())
    
    # Add a few random inits
    for _ in range(3):
        inits.append(_random_init())
    
    best_cents = None
    best_r = 0.0
    
    # Process each initial configuration
    for init in inits:
        if time.time() - start_time > DEADLINE - 10:
            break
        
        x = init.reshape(-1)
        
        # Simulated annealing
        x, obj = _simulated_annealing(x, n_steps=3000, t_start=0.008, t_end=0.00005, 
                                       alpha=50.0, time_limit=40.0)
        
        # L-BFGS polish
        x = _lbfgs_polish(x, alpha=50.0, time_limit=15.0)
        
        # Push phase
        x = _push_phase(x, alpha=50.0, n_pushes=3, time_limit=15.0)
        
        cents = x.reshape(N, 2)
        r = _compute_radius(cents)
        
        if r > best_r:
            best_r = r
            best_cents = cents.copy()
    
    # Final polish on the best configuration
    if best_cents is not None and time.time() - start_time < DEADLINE - 5:
        x = best_cents.reshape(-1)
        x = _lbfgs_polish(x, alpha=100.0, time_limit=10.0)
        cents = x.reshape(N, 2)
        r = _compute_radius(cents)
        if r > best_r:
            best_r = r
            best_cents = cents.copy()
    
    # Ensure validity
    if best_cents is None:
        best_cents = _hex_init()
    
    best_cents = np.clip(best_cents, -0.5, 0.5)
    
    return best_cents
