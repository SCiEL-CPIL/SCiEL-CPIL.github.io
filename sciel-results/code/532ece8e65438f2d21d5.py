"""Erdos minimum overlap - optimize step heights to minimize max overlap with shifted 1-h."""
N = 100
import numpy as np
from scipy.optimize import minimize
import time

def _overlap_max(h, shift):
    """Compute max overlap of h with shifted (1-h) at given shift."""
    # h is length N, 1-h is length N
    # shifted 1-h: for shift s, overlap at position i is h[i] * (1-h[(i+s) % N])
    # Actually, "overlap" likely means the sum or max of pointwise products
    # Let's compute pointwise product and take max
    h_shifted = np.roll(1.0 - h, shift)
    prod = h * h_shifted
    return np.max(prod)

def _objective_and_grad(params, t_init):
    """Epigraph objective: minimize t + penalty for t < max_overlap"""
    N_local = len(params) - 1
    h = params[:N_local]
    t = params[N_local]
    
    # Compute all shifts
    max_ov = 0.0
    for s in range(1, N_local):
        h_shifted = np.roll(1.0 - h, s)
        prod = h * h_shifted
        ov = np.max(prod)
        if ov > max_ov:
            max_ov = ov
    
    # Smooth epigraph: t + max(0, max_ov - t) with softplus-like smoothing
    # Use log-sum-exp style soft max
    if max_ov > t:
        diff = max_ov - t
        obj = t + diff + 0.5 * diff * diff  # quadratic penalty
    else:
        obj = t
    
    return obj

def _construct_h_from_params(params):
    """Extract h from params, ensuring constraints."""
    N_local = len(params) - 1
    h = params[:N_local].copy()
    # Clip to [0, 1]
    h = np.clip(h, 0.0, 1.0)
    # Enforce sum = N/2
    s = np.sum(h)
    if s > 0:
        h = h * (N_local / 2.0) / s
        h = np.clip(h, 0.0, 1.0)
    return h

def _score(h):
    """Compute the score (max overlap over all shifts)."""
    N_local = len(h)
    max_ov = 0.0
    for s in range(1, N_local):
        h_shifted = np.roll(1.0 - h, s)
        prod = h * h_shifted
        ov = np.max(prod)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _make_initial_guess(method='uniform'):
    """Generate diverse initial guesses."""
    if method == 'uniform':
        return np.full(N, 0.5)
    elif method == 'alternating':
        h = np.zeros(N)
        h[0::2] = 0.7
        h[1::2] = 0.3
        # Normalize to sum = N/2
        s = np.sum(h)
        h = h * (N / 2.0) / s
        return np.clip(h, 0.0, 1.0)
    elif method == 'smooth_wave':
        x = np.linspace(0, 2 * np.pi, N, endpoint=False)
        h = 0.5 + 0.3 * np.sin(x)
        h = np.clip(h, 0.0, 1.0)
        s = np.sum(h)
        h = h * (N / 2.0) / s
        return np.clip(h, 0.0, 1.0)
    elif method == 'block':
        h = np.zeros(N)
        h[:N//4] = 0.8
        h[N//4:N//2] = 0.2
        h[N//2:3*N//4] = 0.8
        h[3*N//4:] = 0.2
        s = np.sum(h)
        h = h * (N / 2.0) / s
        return np.clip(h, 0.0, 1.0)
    else:
        return np.full(N, 0.5)

def _optimize_from_init(h0, deadline):
    """Run SLSQP optimization from initial guess h0."""
    # Parameters: h (N values) + t (1 value)
    n_params = N + 1
    x0 = np.concatenate([h0, [_score(h0)]])
    
    def obj(x):
        h = x[:N]
        t = x[N]
        # Enforce bounds softly
        h_clipped = np.clip(h, 0.0, 1.0)
        max_ov = 0.0
        for s in range(1, N):
            h_shifted = np.roll(1.0 - h_clipped, s)
            prod = h_clipped * h_shifted
            ov = np.max(prod)
            if ov > max_ov:
                max_ov = ov
        # Epigraph: minimize t subject to t >= max_ov
        # Penalty formulation
        if max_ov > t:
            return t + 100.0 * (max_ov - t) ** 2 + 0.01 * (max_ov - t)
        else:
            return t + 0.01 * (t - max_ov)
    
    def constraints(x):
        h = x[:N]
        # Sum constraint: sum(h) = N/2
        return np.sum(h) - N / 2.0
    
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]
    cons = [{'type': 'eq', 'fun': constraints}]
    
    try:
        result = minimize(
            obj, x0, method='SLSQP',
            bounds=bounds, constraints=cons,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        if result.success or result.fun < obj(x0):
            h_opt = np.clip(result.x[:N], 0.0, 1.0)
            # Re-normalize
            s = np.sum(h_opt)
            if s > 0:
                h_opt = h_opt * (N / 2.0) / s
                h_opt = np.clip(h_opt, 0.0, 1.0)
            return h_opt
    except:
        pass
    
    return h0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Try multiple initialization strategies
    init_methods = ['uniform', 'alternating', 'smooth_wave', 'block']
    
    for method in init_methods:
        if time.time() > deadline - 30.0:
            break
        
        h0 = _make_initial_guess(method)
        h_opt = _optimize_from_init(h0, deadline)
        sc = _score(h_opt)
        
        if sc < best_score:
            best_score = sc
            best_h = h_opt.copy()
    
    # Additional refinement passes on best solution
    if time.time() < deadline - 10.0:
        for _ in range(3):
            if time.time() > deadline - 10.0:
                break
            h_opt = _optimize_from_init(best_h, deadline)
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
    
    # Final check
    final_h = np.clip(best_h, 0.0, 1.0)
    s = np.sum(final_h)
    if s > 0:
        final_h = final_h * (N / 2.0) / s
        final_h = np.clip(final_h, 0.0, 1.0)
    
    return final_h
