"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=100.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=100.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _overlap_and_grad(h, alpha):
    """Compute overlap and gradient of smooth max w.r.t. h."""
    shifted = np.zeros(N)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    smax = _smooth_max(overlap, alpha)
    w = _smooth_max_grad(overlap, alpha)
    
    grad_h = np.zeros(N)
    for i in range(N - 1):
        if w[i] > 1e-15:
            grad_h[i] += w[i] * (1.0 - h[i+1])
            grad_h[i+1] += w[i] * (-h[i])
    if w[N-1] > 1e-15:
        grad_h[N-1] += w[N-1]
    
    return overlap, smax, grad_h, w

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if s < 1e-10:
        h = np.full(N, 0.5)
        return h
    h = h * (target / s)
    h = np.clip(h, 0.0, 1.0)
    # Fix residual iteratively
    for _ in range(20):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        else:
            mask = h > 0.0
            if mask.any():
                h[mask] += diff / mask.sum()
            else:
                break
        h = np.clip(h, 0.0, 1.0)
    return h

def _two_block_init():
    """Analytic 2-block initial guess."""
    A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.382
    B = 1.0 - A  # ~0.618
    h0 = np.zeros(N)
    half = N // 2
    h0[:half] = A
    h0[half:] = B
    return h0

def _knots_to_h(knots):
    """
    Convert knot parameters to a full h vector.
    knots: array of shape (K, 2) where each row is (position_fraction, value).
    We use linear interpolation between knots, then project to feasible.
    """
    K = len(knots)
    positions = knots[:, 0] * (N - 1)  # map to index space
    values = knots[:, 1]
    
    # Ensure positions are sorted
    order = np.argsort(positions)
    positions = positions[order]
    values = values[order]
    
    # Create grid for interpolation
    x_grid = np.arange(N, dtype=float)
    
    # Linear interpolation
    h = np.interp(x_grid, positions, values, left=values[0], right=values[-1])
    
    return _project_to_feasible(h)

def _optimize_knots(deadline, num_knots=6):
    """
    Optimize knot positions and values using differential evolution.
    """
    def objective_knots(params):
        # params: [pos_0, val_0, pos_1, val_1, ..., pos_{K-1}, val_{K-1}]
        K = num_knots
        knots = params.reshape(K, 2)
        h = _knots_to_h(knots)
        return _score(h)
    
    # Bounds: positions in [0, 1], values in [0, 1]
    bounds = []
    for _ in range(num_knots):
        bounds.append((0.0, 1.0))  # position
        bounds.append((0.0, 1.0))  # value
    
    # Initial guess: uniform positions, two-block values
    init_knots = np.zeros((num_knots, 2))
    for i in range(num_knots):
        init_knots[i, 0] = i / (num_knots - 1)
        # Interpolate between A and B
        frac = i / (num_knots - 1)
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        init_knots[i, 1] = A + frac * (B - A)
    
    try:
        result = differential_evolution(
            objective_knots,
            bounds=bounds,
            x0=init_knots.flatten(),
            maxiter=50,
            popsize=15,
            tol=1e-10,
            seed=42,
            polish=False,
            time_budget=deadline - time.time() - 10.0
        )
        
        if result.fun < float('inf'):
            best_knots = result.x.reshape(num_knots, 2)
            h_best = _knots_to_h(best_knots)
            return h_best, result.fun
    except Exception:
        pass
    
    return None, float('inf')

def _gradient_descent(h_init, deadline):
    """Gradient-based local search with adaptive step size."""
    h = h_init.copy()
    best_score = _score(h)
    best_h = h.copy()
    
    for iteration in range(5000):
        if time.time() > deadline - 5:
            break
        
        # Compute gradient of smooth max objective
        alpha = 500.0
        overlap, smax, grad_h, w = _overlap_and_grad(h, alpha)
        
        # Project gradient onto feasible subspace (sum constraint)
        grad_centered = grad_h - np.mean(grad_h)
        
        # Line search with backtracking
        step_size = max(0.0001, 0.01 * (1.0 - iteration / 5000.0) ** 0.5)
        improved = False
        
        for shrink in range(20):
            trial_step = step_size * (0.5 ** shrink)
            if trial_step < 1e-10:
                break
            h_trial = h - trial_step * grad_centered
            h_trial = np.clip(h_trial, 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            
            if sc < best_score - 1e-15:
                best_score = sc
                best_h = h_trial.copy()
                h = h_trial.copy()
                improved = True
                break
        
        if not improved and iteration % 100 == 0:
            # Try random perturbation
            rng = np.random.default_rng(iteration)
            h_trial = h.copy()
            idxs = rng.choice(N, size=5, replace=False)
            for i in idxs:
                h_trial[i] = np.clip(h_trial[i] + rng.uniform(-0.01, 0.01), 0.0, 1.0)
            h_trial = _project_to_feasible(h_trial)
            sc = _score(h_trial)
            if sc < best_score - 1e-15:
                best_score = sc
                best_h = h_trial.copy()
                h = h_trial.copy()
    
    return best_h, best_score

def _slsqp_polish(h_init, t_init, deadline):
    """SLSQP polish with epigraph formulation."""
    def objective_polish(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        smax = _smooth_max(overlap, alpha=500.0)
        return t + 1000.0 * np.maximum(0.0, smax - t)**2
    
    x0 = np.concatenate([h_init, [t_init]])
    bounds_ext = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    
    def constraint_sum(params):
        return np.sum(params[:N]) - N/2.0
    
    def constraint_epigraph(params):
        h = params[:N]
        t = params[N]
        shifted = np.zeros(N)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        return t - np.max(overlap)
    
    constraints_ext = [
        {'type': 'eq', 'fun': constraint_sum},
        {'type': 'ineq', 'fun': constraint_epigraph}
    ]
    
    try:
        result = minimize(
            objective_polish,
            x0,
            method='SLSQP',
            bounds=bounds_ext,
            constraints=constraints_ext,
            options={'maxiter': 300, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.fun < 100.0:
            h_opt = result.x[:N].copy()
            h_opt = _project_to_feasible(h_opt)
            new_score = _score(h_opt)
            if np.all(h_opt >= 0.0) and np.all(h_opt <= 1.0):
                return h_opt, new_score
    except Exception:
        pass
    
    return None, float('inf')

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Phase 0: Start with analytic 2-block solution
    h0 = _two_block_init()
    h_final = h0.copy()
    current_score = _score(h0)
    
    # Phase 1: Knot-based global optimization
    # Try different numbers of knots to capture different transition shapes
    for num_knots in [4, 6, 8]:
        if time.time() > deadline - 20.0:
            break
        
        h_knots, score_knots = _optimize_knots(deadline, num_knots)
        if h_knots is not None and score_knots < current_score:
            h_final = h_knots.copy()
            current_score = score_knots
    
    # Phase 2: Gradient descent local search
    if time.time() < deadline - 10.0:
        h_grad, score_grad = _gradient_descent(h_final, deadline)
        if score_grad < current_score:
            h_final = h_grad.copy()
            current_score = score_grad
    
    # Phase 3: SLSQP polish
    if time.time() < deadline - 5.0:
        h_slsqp, score_slsqp = _slsqp_polish(h_final, current_score + 0.01, deadline)
        if h_slsqp is not None and score_slsqp < current_score:
            h_final = h_slsqp.copy()
            current_score = score_slsqp
    
    # Final validation
    h_final = _project_to_feasible(h_final)
    final_score = _score(h_final)
    
    if not (np.all(h_final >= -1e-8) and np.all(h_final <= 1.0 + 1e-8)):
        h_final = np.full(N, 0.5)
    
    return h_final
