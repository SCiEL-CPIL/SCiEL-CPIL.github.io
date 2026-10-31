"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill
TARGET_SUM = N / 2.0


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    if n == 0:
        return 0.0
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds.
    Uses iterative water-filling style correction for better bound handling."""
    if target is None:
        target = TARGET_SUM
    h = np.asarray(h, dtype=float).copy()
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - h.sum()
    if abs(diff) > 1e-14:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _smooth_max_and_grad(x, beta):
    """
    Compute soft max of x and its gradient using log-sum-exp.
    Returns (value, gradient_vector).
    """
    x = np.asarray(x, dtype=float)
    max_x = np.max(x)
    if max_x < 1e-12:
        return 0.0, np.zeros_like(x)
    
    # Shift for numerical stability
    shifted = x - max_x
    weights = np.exp(beta * shifted)
    wsum = np.sum(weights)
    
    if wsum == 0:
        return max_x, np.zeros_like(x)
        
    value = max_x + math.log(wsum) / beta
    grad = weights / wsum
    return value, grad


def _objective_and_grad(h, beta):
    """
    Compute the smooth objective value and its gradient.
    Objective: smooth_max(overlap) + penalty * (sum(h) - N/2)^2
    """
    n = len(h)
    h = np.clip(h, 0.0, 1.0)
    
    # Calculate overlap vector: o_i = h_i * (1 - h_{i+1})
    overlap = np.zeros(n)
    if n > 1:
        overlap[:-1] = h[:-1] * (1.0 - h[1:])
        overlap[-1] = h[-1]
    else:
        overlap[0] = h[0]
        
    val_smooth, grad_overlap = _smooth_max_and_grad(overlap, beta)
    
    # Gradient of overlap w.r.t h
    grad_h = np.zeros(n)
    if n > 1:
        # Contribution from o_i to h_i
        grad_h[:-1] += grad_overlap[:-1] * (1.0 - h[1:])
        # Contribution from o_i to h_{i+1}
        grad_h[1:] += grad_overlap[:-1] * (-h[:-1])
        # Contribution from o_{N-1} to h_{N-1}
        grad_h[-1] += grad_overlap[-1] * 1.0
    else:
        grad_h[0] += grad_overlap[0] * 1.0
        
    # Penalty term for sum constraint
    penalty_weight = 100.0
    sum_error = h.sum() - TARGET_SUM
    penalty_val = penalty_weight * sum_error * sum_error
    penalty_grad = 2.0 * penalty_weight * sum_error
    
    total_val = val_smooth + penalty_val
    total_grad = grad_h + penalty_grad
    
    return total_val, total_grad


def _optimize_direct(h0, deadline, beta_start=50.0, beta_end=500.0, max_iter=500):
    """
    Optimize h using L-BFGS-B with analytic gradients and adaptive beta.
    """
    h = h0.copy()
    h = _fix_sum(h)
    current_score = _score(h)
    
    # We will run a few stages with increasing beta
    stages = [
        (beta_start, max_iter // 2),
        (beta_end // 2, max_iter // 4),
        (beta_end, max_iter // 4)
    ]
    
    for beta, iters in stages:
        if time.time() > deadline - 2.0:
            break
            
        def obj_func(x):
            val, _ = _objective_and_grad(x, beta)
            return val
        
        def grad_func(x):
            _, grad = _objective_and_grad(x, beta)
            return grad
        
        try:
            res = minimize(
                obj_func, 
                h, 
                jac=grad_func,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * N,
                options={'maxiter': iters, 'ftol': 1e-14, 'gtol': 1e-10, 'iprint': 0}
            )
            h_new = res.x
            h_new = _fix_sum(h_new)
            new_score = _score(h_new)
            if new_score < current_score:
                h = h_new
                current_score = new_score
        except Exception:
            pass
            
    return current_score, h


def _local_search_polish(h, max_iters=200, deadline=None):
    """
    Deterministic local search to polish the solution.
    Tries small adjustments and swaps.
    """
    h = h.copy()
    current_score = _score(h)
    
    if deadline is None:
        deadline = time.time() + 5.0
        
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        improved = False
        # Try small adjustments
        for i in range(N):
            for delta in [0.005, -0.005, 0.01, -0.01, 0.002, -0.002]:
                h_trial = h.copy()
                h_trial[i] += delta
                if h_trial[i] < 0 or h_trial[i] > 1:
                    continue
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if improved:
                break
        
        # Try swaps
        if not improved:
            for i in range(N - 1):
                h_trial = h.copy()
                h_trial[i], h_trial[i+1] = h_trial[i+1], h_trial[i]
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
        if not improved:
            break
            
    return current_score, h


def _sinusoidal_h(A, phi, A2=0.0, phi2=0.0):
    """Generate step heights from sinusoidal model with optional 2nd harmonic.
    h_i = 0.5 + A*sin(2*pi*i/N + phi) + A2*sin(4*pi*i/N + phi2)"""
    i = np.arange(N)
    h = 0.5 + A * np.sin(2.0 * np.pi * i / N + phi)
    if abs(A2) > 1e-12:
        h += A2 * np.sin(4.0 * np.pi * i / N + phi2)
    return h


def _optimize_sinusoidal():
    """
    Optimize sinusoidal parameters (A, phi, A2, phi2) via:
    1. Coarse grid search over (A, phi)
    2. Local Nelder-Mead refinement on 4 params
    """
    best_score = float('inf')
    best_h = None

    # Phase 1: Coarse grid search
    A_grid = np.linspace(0.05, 0.45, 25)
    phi_grid = np.linspace(0.0, 2.0 * np.pi, 50)

    best_A = 0.25
    best_phi = 0.0
    best_grid_score = float('inf')

    for A in A_grid:
        for phi in phi_grid:
            h = _sinusoidal_h(A, phi)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_A = A
                best_phi = phi

    # Phase 2: Local refinement with Nelder-Mead on 4 params
    def obj_sin(params):
        A, phi, A2, phi2 = params
        A = np.clip(A, 0.0, 0.5)
        A2 = np.clip(A2, -0.3, 0.3)
        h = _sinusoidal_h(A, phi, A2, phi2)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0 = np.array([best_A, best_phi, 0.0, 0.0])
    try:
        res = minimize(obj_sin, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_A = float(res.x[0])
            best_phi = float(res.x[1])
            best_A2 = float(res.x[2])
            best_phi2 = float(res.x[3])
    except Exception:
        best_A2 = 0.0
        best_phi2 = 0.0

    # Phase 3: Try multiple starting points for local refinement
    for A_init in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        for phi_init in [0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]:
            for A2_init in [0.0, 0.05, -0.05]:
                x0 = np.array([A_init, phi_init, A2_init, 0.0])
                try:
                    res = minimize(obj_sin, x0, method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.fun < best_grid_score:
                        best_grid_score = res.fun
                        best_A = float(res.x[0])
                        best_phi = float(res.x[1])
                        best_A2 = float(res.x[2])
                        best_phi2 = float(res.x[3])
                except Exception:
                    pass

    # Build best sinusoidal h
    h_sin = _sinusoidal_h(best_A, best_phi, best_A2, best_phi2)
    h_sin = np.clip(h_sin, 0.0, 1.0)
    h_sin = _fix_sum(h_sin)
    best_score = _score(h_sin)
    best_h = h_sin.copy()

    return best_score, best_h, (best_A, best_phi, best_A2, best_phi2)


def _optimize_multiblock(num_blocks=3):
    """
    Optimize a multi-block step function with num_blocks equal-width segments.
    Each segment has a height, and we optimize those heights.
    """
    best_score = float('inf')
    best_h = None

    # Equal-width blocks
    block_width = N / num_blocks
    # Initial guess: alternating high/low
    heights = np.zeros(num_blocks)
    for i in range(num_blocks):
        heights[i] = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / num_blocks)

    def obj_heights(hs):
        hs = np.clip(hs, 0.0, 1.0)
        # Build full h vector
        h = np.zeros(N)
        for i in range(num_blocks):
            start = int(i * block_width)
            end = int((i + 1) * block_width)
            h[start:end] = hs[i]
        h = _fix_sum(h)
        return _score(h)

    # Try multiple initializations
    for init_type in range(5):
        if init_type == 0:
            h0 = np.zeros(num_blocks)
            for i in range(num_blocks):
                h0[i] = 0.5 + 0.3 * np.cos(2.0 * np.pi * i / num_blocks)
        elif init_type == 1:
            h0 = np.full(num_blocks, 0.5)
            h0[0] = 0.7
            h0[-1] = 0.3
        elif init_type == 2:
            h0 = np.linspace(0.3, 0.7, num_blocks)
        elif init_type == 3:
            h0 = np.full(num_blocks, 0.5)
            h0[::2] = 0.6
            h0[1::2] = 0.4
        else:
            h0 = np.random.uniform(0.2, 0.8, num_blocks)

        try:
            res = minimize(obj_heights, h0, method='Nelder-Mead',
                           bounds=[(0.0, 1.0)] * num_blocks,
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            hs_opt = np.clip(res.x, 0.0, 1.0)
            h = np.zeros(N)
            for i in range(num_blocks):
                start = int(i * block_width)
                end = int((i + 1) * block_width)
                h[start:end] = hs_opt[i]
            h = _fix_sum(h)
            s = _score(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
        except Exception:
            pass

    if best_h is None:
        # Fallback
        h = np.full(N, 0.5)
        return _score(h), h

    return best_score, best_h


def _logistic_h(r, x0):
    """
    Generate step heights from the logistic map.
    x_{n+1} = r * x_n * (1 - x_n)
    h_i = x_i
    """
    x = x0
    h = np.zeros(N)
    for i in range(N):
        h[i] = x
        x = r * x * (1.0 - x)
    return h


def _optimize_logistic():
    """
    Optimize logistic map parameters (r, x0) to generate a sequence h.
    """
    best_score = float('inf')
    best_h = None
    best_params = None

    # Grid search over r and x0
    r_grid = np.linspace(3.5, 4.0, 25)
    x0_grid = np.linspace(0.1, 0.9, 20)

    best_r = 3.9
    best_x0 = 0.5
    best_grid_score = float('inf')

    for r in r_grid:
        for x0 in x0_grid:
            h = _logistic_h(r, x0)
            h = np.clip(h, 0.0, 1.0)
            h = _fix_sum(h)
            s = _score(h)
            if s < best_grid_score:
                best_grid_score = s
                best_r = r
                best_x0 = x0

    # Local refinement with Nelder-Mead on (r, x0)
    def obj_log(params):
        r, x0 = params
        r = np.clip(r, 3.5, 4.0)
        x0 = np.clip(x0, 0.01, 0.99)
        h = _logistic_h(r, x0)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        return _score(h)

    x0_opt = np.array([best_r, best_x0])
    try:
        res = minimize(obj_log, x0_opt, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
        if res.fun < best_grid_score:
            best_grid_score = res.fun
            best_r = float(res.x[0])
            best_x0 = float(res.x[1])
    except Exception:
        pass

    # Try multiple starting points
    for r_init in [3.7, 3.8, 3.9, 3.95, 3.99]:
        for x0_init in [0.2, 0.4, 0.5, 0.6, 0.8]:
            x0_opt = np.array([r_init, x0_init])
            try:
                res = minimize(obj_log, x0_opt, method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                if res.fun < best_grid_score:
                    best_grid_score = res.fun
                    best_r = float(res.x[0])
                    best_x0 = float(res.x[1])
            except Exception:
                pass

    # Build best logistic h
    h_log = _logistic_h(best_r, best_x0)
    h_log = np.clip(h_log, 0.0, 1.0)
    h_log = _fix_sum(h_log)
    best_score = _score(h_log)
    best_h = h_log.copy()
    best_params = (best_r, best_x0)

    return best_score, best_h, best_params


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    # Phase 1: Sinusoidal optimization
    best_score = float('inf')
    best_h = None

    if time.time() < deadline:
        s_sin, h_sin, _ = _optimize_sinusoidal()
        if s_sin < best_score:
            best_score = s_sin
            best_h = h_sin.copy()

    # Phase 2: Multi-block optimization (3 and 4 blocks)
    for nb in [3, 4]:
        if time.time() > deadline - 10.0:
            break
        s_mb, h_mb = _optimize_multiblock(nb)
        if s_mb < best_score:
            best_score = s_mb
            best_h = h_mb.copy()

    # Phase 3: Logistic map optimization (new structural approach)
    if time.time() < deadline - 15.0:
        s_log, h_log, _ = _optimize_logistic()
        if s_log < best_score:
            best_score = s_log
            best_h = h_log.copy()

    # Phase 4: Direct optimization with analytic gradients from best candidate
    if best_h is not None and time.time() < deadline - 15.0:
        s_direct, h_direct = _optimize_direct(best_h, deadline - 15.0, beta_start=50.0, beta_end=500.0, max_iter=500)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 5: Local search polish
    if best_h is not None and time.time() < deadline - 5.0:
        s_polish, h_polish = _local_search_polish(best_h, max_iters=200, deadline=deadline - 5.0)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 6: Final high-precision direct optimization
    if best_h is not None and time.time() < deadline - 3.0:
        s_final, h_final = _optimize_direct(best_h, deadline - 3.0, beta_start=200.0, beta_end=1000.0, max_iter=300)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
