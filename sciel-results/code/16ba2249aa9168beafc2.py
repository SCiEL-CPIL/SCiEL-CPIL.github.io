"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 5.0  # seconds before 150s limit


def _overlap_at_shift(h, s):
    """Compute overlap of h with h shifted by s positions (circular)."""
    # Shift h right by s: new[i] = h[(i - s) % N]
    shifted = np.roll(h, s)
    # Overlap = sum of min(h[i], shifted[i])
    return np.sum(np.minimum(h, shifted))


def _max_overlap(h):
    """Maximum overlap over all shifts 1..N-1."""
    max_ov = 0.0
    for s in range(1, N):
        ov = _overlap_at_shift(h, s)
        if ov > max_ov:
            max_ov = ov
    return max_ov


def _objective_and_grad(x):
    """
    Objective: maximize t subject to t <= overlap(s) for all s, and constraints.
    We minimize -t + penalty for constraint violations.
    
    Actually, let's use a smooth max approximation:
    f(h) = soft_max of overlaps = (1/beta) * log(sum(exp(beta * ov_s)))
    We minimize this.
    """
    h = np.clip(x, 0.0, 1.0)
    
    # Compute all overlaps
    overlaps = np.zeros(N - 1)
    for s in range(1, N):
        shifted = np.roll(h, s)
        overlaps[s - 1] = np.sum(np.minimum(h, shifted))
    
    # Soft max with temperature beta
    beta = 50.0
    max_ov = np.max(overlaps)
    # Stable log-sum-exp
    shifted_exp = np.exp(beta * (overlaps - max_ov))
    soft_max = max_ov + np.log(np.sum(shifted_exp)) / beta
    
    # Gradient of soft max w.r.t. h
    # d(soft_max)/d(h[i]) = sum_s (exp(beta*(ov_s - max_ov)) / sum(exp(...))) * d(ov_s)/d(h[i])
    weights = shifted_exp / np.sum(shifted_exp)
    
    grad = np.zeros(N)
    for s in range(1, N):
        w = weights[s - 1]
        if w < 1e-12:
            continue
        shifted = np.roll(h, s)
        # d(ov_s)/d(h[i]) = 0.5 if h[i] == shifted[i], else 1 if h[i] < shifted[i], else -1 if h[i] > shifted[i]
        # More precisely: d min(a,b)/da = 1 if a < b, 0.5 if a == b, 0 if a > b
        # And d min(a,b)/db = 0 if a < b, 0.5 if a == b, 1 if a > b
        # ov_s = sum_i min(h[i], shifted[i])
        # d(ov_s)/d(h[i]) = 1 if h[i] < shifted[i], 0.5 if h[i] == shifted[i], 0 if h[i] > shifted[i]
        # d(ov_s)/d(shifted[i]) = 0 if h[i] < shifted[i], 0.5 if h[i] == shifted[i], 1 if h[i] > shifted[i]
        # shifted[i] = h[(i-s)%N], so d(ov_s)/d(h[j]) gets contribution from shifted[(j+s)%N]
        
        d_h = np.where(h < shifted, 1.0, np.where(h > shifted, 0.0, 0.5))
        d_shifted = np.where(h < shifted, 0.0, np.where(h > shifted, 1.0, 0.5))
        
        grad += w * d_h
        grad += w * np.roll(d_shifted, -s)
    
    # Add constraint gradient: sum(h) = N/2
    # We'll handle this via projection instead
    
    return soft_max, grad


def _project(x):
    """Project x onto the set {h in [0,1]^N : sum(h) = N/2}."""
    h = np.clip(x, 0.0, 1.0)
    # Adjust to satisfy sum constraint
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    
    # Distribute diff among elements not at bounds
    mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    if np.sum(mask) == 0:
        return h
    
    # Simple: add diff/n to all free elements, then clip and iterate
    for _ in range(100):
        h[mask] += diff / np.sum(mask)
        h = np.clip(h, 0.0, 1.0)
        current_sum = np.sum(h)
        diff = target - current_sum
        if abs(diff) < 1e-12:
            break
        mask = (h > 1e-10) & (h < 1.0 - 1e-10)
        if np.sum(mask) == 0:
            break
    
    return h


def _sturmian_sequence(alpha, rho, n):
    """Generate a Sturmian word of length n with slope alpha and intercept rho."""
    # Sturmian word: w[i] = floor((i+1)*alpha + rho) - floor(i*alpha + rho)
    # This gives 0 or 1 values
    i = np.arange(n)
    vals = np.floor((i + 1) * alpha + rho) - np.floor(i * alpha + rho)
    # Map to [0, 1] range with mean 0.5
    # The mean of vals is alpha, so we want mean 0.5
    # Scale: h[i] = (vals[i] - alpha) / (1 - 2*alpha) * 0.5 + 0.5 if alpha != 0.5
    # Simpler: just use vals directly as 0/1, then adjust
    return vals.astype(float)


def _random_initialization():
    """Generate a random valid initial point."""
    h = np.random.uniform(0, 1, N)
    return _project(h)


def _sinusoidal_initialization(freq, phase):
    """Generate a sinusoidal initial point."""
    x = np.arange(N)
    h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * x / N + phase)
    return _project(h)


def construct():
    start_time = time.time()
    deadline = 150.0 - DEADLINE_MARGIN
    
    best_h = np.full(N, 0.5)
    best_score = _max_overlap(best_h)
    
    # Try several initializations
    initializations = []
    
    # 1. Uniform
    initializations.append(np.full(N, 0.5))
    
    # 2. Sturmian with golden ratio
    alpha = (np.sqrt(5) - 1) / 2  # ~0.618
    for rho in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        h_sturm = _sturmian_sequence(alpha, rho, N)
        # Adjust to have mean 0.5
        h_sturm = (h_sturm - np.mean(h_sturm)) / (np.max(h_sturm - np.mean(h_sturm)) + np.min(np.max(h_sturm - np.mean(h_sturm)), 1e-10)) * 0.5 + 0.5
        h_sturm = np.clip(h_sturm, 0, 1)
        initializations.append(_project(h_sturm))
    
    # 3. Sinusoidal
    for freq in [1, 2, 3, 5, 7, 10, 15, 20]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            initializations.append(_sinusoidal_initialization(freq, phase))
    
    # 4. Random
    for _ in range(20):
        initializations.append(_random_initialization())
    
    # Evaluate all initializations
    for h_init in initializations:
        if time.time() > deadline:
            break
        score = _max_overlap(h_init)
        if score < best_score:
            best_score = score
            best_h = h_init.copy()
    
    # Optimize from the best initialization
    h = best_h.copy()
    
    # Gradient-based optimization using L-BFGS-B
    def obj(x):
        val, _ = _objective_and_grad(x)
        return val
    
    def obj_grad(x):
        val, grad = _objective_and_grad(x)
        return val, grad
    
    # Use multiple restarts from different points
    for restart in range(5):
        if time.time() > deadline:
            break
        
        if restart == 0:
            x0 = best_h.copy()
        else:
            x0 = _random_initialization()
            # Perturb
            x0 += 0.1 * np.random.normal(0, 1, N)
            x0 = _project(x0)
        
        try:
            result = minimize(
                obj_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            h_opt = _project(result.x)
            score = _max_overlap(h_opt)
            
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    # Local search: try swapping pairs of values
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                for j in range(i + 1, N):
                    if time.time() > deadline:
                        break
                    # Try swapping h[i] and h[j]
                    old_i, old_j = h_best = best_h[i], best_h[j]
                    best_h[i], best_h[j] = best_h[j], best_h[i]
                    new_score = _max_overlap(best_h)
                    if new_score < best_score - 1e-12:
                        best_score = new_score
                        improved = True
                    else:
                        best_h[i], best_h[j] = old_i, old_j
    
    # Final polish: use tighter optimization
    if time.time() < deadline - 2.0:
        h = best_h.copy()
        try:
            result = minimize(
                obj_grad,
                h,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            h_opt = _project(result.x)
            score = _max_overlap(h_opt)
            if score < best_score:
                best_score = score
                best_h = h_opt.copy()
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
