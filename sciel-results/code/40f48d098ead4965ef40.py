"""Search-based program for Erdos minimum overlap task."""
import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _max_overlap(h):
    """Compute max over shifts of mean(h[i]*(1-h[(i+s)%N]))."""
    N = len(h)
    inv_h = 1.0 - h
    best = 0.0
    for s in range(N):
        ov = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
        if ov > best:
            best = ov
    return best

def _overlap_vector(h):
    """Compute all shift overlaps as a vector."""
    N = len(h)
    inv_h = 1.0 - h
    overlaps = np.zeros(N)
    for s in range(N):
        overlaps[s] = np.dot(h, inv_h[np.roll(np.arange(N), -s)])
    return overlaps

def _smooth_max(overlaps, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = np.max(overlaps)
    if m < 1e-12:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (overlaps - m)))) / beta

def _smooth_max_grad(overlaps, beta=50.0):
    """Gradient of smooth max with respect to overlaps."""
    m = np.max(overlaps)
    exps = np.exp(beta * (overlaps - m))
    weights = exps / np.sum(exps)
    return weights

def _objective_and_grad(x, beta=50.0):
    """Objective: smooth max of overlaps + penalty for constraint violations.
    x is unconstrained; we map to [0,1] via sigmoid and enforce sum constraint."""
    # Map to [0,1] via sigmoid
    h = 1.0 / (1.0 + np.exp(-x))
    h = np.clip(h, 0.0, 1.0)
    
    # Enforce sum = N/2 by scaling
    target = N / 2.0
    current_sum = h.sum()
    if current_sum < 1e-10:
        return 1.0, np.zeros_like(x)
    h_scaled = h * (target / current_sum)
    h_scaled = np.clip(h_scaled, 0.0, 1.0)
    
    # Compute overlaps
    ov = _overlap_vector(h_scaled)
    sm = _smooth_max(ov, beta)
    
    # Penalty for being outside [0,1] after scaling (shouldn't happen much)
    penalty = 100.0 * (np.sum(np.maximum(0, h_scaled - 1.0)) + np.sum(np.maximum(0, -h_scaled)))
    
    val = sm + penalty
    
    # Gradient
    ov_grad = _smooth_max_grad(ov, beta)
    
    # d(ov_s)/d(h_i) = (1-h[(i+s)%N]) - h[i]*(1-h[(i+s)%N])... 
    # Actually ov_s = sum_i h[i]*(1-h[(i+s)%N])
    # d(ov_s)/d(h_j) = (1-h[(j+s)%N]) - h[j]*(-1) if j appears in second term
    # = (1-h[(j+s)%N]) + h[j] * delta(j, (i+s)%N for some i)
    # More carefully: ov_s = sum_i h[i]*(1-h[(i+s)%N])
    # d(ov_s)/d(h_j) = (1-h[(j+s)%N]) + sum_i h[i] * d(1-h[(i+s)%N])/d(h_j)
    # d(1-h[(i+s)%N])/d(h_j) = -1 if (i+s)%N == j, i.e., i = (j-s)%N
    # = -h[(j-s)%N]
    # So d(ov_s)/d(h_j) = (1-h[(j+s)%N]) - h[(j-s)%N]
    
    grad_ov = np.zeros_like(h_scaled)
    for s in range(N):
        # d(ov_s)/d(h_j) = (1-h[(j+s)%N]) - h[(j-s)%N]
        grad_ov += ov_grad[s] * ((1.0 - h_scaled[(np.arange(N) + s) % N]) - h_scaled[(np.arange(N) - s) % N])
    
    # Chain rule: h_scaled = h * (target / sum(h))
    # This is complex; approximate by treating scaling as fixed for gradient
    # d(h_scaled_j)/d(h_j) = target/sum(h) - h_j*target/sum(h)^2
    scale = target / current_sum
    dh_scaled_dh = scale * (1.0 - h_scaled / current_sum)
    
    # d(h_j)/d(x_j) = h_j*(1-h_j)
    dx = grad_ov * dh_scaled_dh * h_scaled * (1.0 - h_scaled)
    
    return val, dx

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
    # Fix residual
    diff = target - h.sum()
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
    return h

def _score(h):
    s = h.sum()
    if abs(s - N/2.0) > 1e-8:
        return float('inf')
    if h.min() < -1e-8 or h.max() > 1.0 + 1e-8:
        return float('inf')
    return _max_overlap(h)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 115.0  # safety margin before 150s kill
    rng = np.random.default_rng(42)
    
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)  # 0.25
    
    # Generate diverse initializations
    inits = []
    # Uniform
    inits.append(np.zeros(N))
    
    # Sinusoidal patterns
    for freq in [1, 2, 3, 4, 5, 7, 10]:
        for phase in [0, np.pi/2, np.pi, 3*np.pi/2]:
            x = np.arange(N)
            h = 0.5 + 0.4 * np.sin(2*np.pi*freq*x/N + phase)
            inits.append(np.log(h/(1-h)))  # inverse sigmoid
    
    # Block patterns
    for block in [1, 2, 3, 4, 5, 10, 25]:
        h = np.zeros(N)
        for i in range(N):
            if (i // block) % 2 == 0:
                h[i] = 1.0
        h = np.clip(h, 0.01, 0.99)
        inits.append(np.log(h/(1-h)))
    
    # Random
    for _ in range(20):
        h = rng.uniform(0.1, 0.9, N)
        inits.append(np.log(h/(1-h)))
    
    # Optimize from each initialization
    for init in inits:
        if time.time() > deadline:
            break
        
        try:
            res = minimize(
                _objective_and_grad,
                init,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            h_opt = _project_to_feasible(1.0 / (1.0 + np.exp(-res.x)))
            sc = _score(h_opt)
            if sc < best_score:
                best_score = sc
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Local search refinement
    h = best_h.copy()
    for _ in range(3000):
        if time.time() > deadline:
            break
        h_trial = h.copy()
        idxs = rng.choice(N, size=3, replace=False)
        for i in idxs:
            h_trial[i] = np.clip(h_trial[i] + rng.uniform(-0.05, 0.05), 0.0, 1.0)
        h_trial = _project_to_feasible(h_trial)
        sc = _score(h_trial)
        if sc < best_score - 1e-10:
            best_score = sc
            best_h = h_trial.copy()
            h = h_trial.copy()
    
    return best_h
# EVOLVE-BLOCK-END
