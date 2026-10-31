"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _smooth_max(x, alpha=50.0):
    """Smooth approximation of max using log-sum-exp."""
    x_max = np.max(x)
    return x_max + (1.0/alpha) * np.log(np.sum(np.exp(alpha * (x - x_max))))

def _smooth_max_grad(x, alpha=50.0):
    """Gradient of smooth max with respect to x."""
    x_max = np.max(x)
    exps = np.exp(alpha * (x - x_max))
    weights = exps / np.sum(exps)
    return weights

def _overlap_and_grad_h(h, alpha=50.0):
    """Return overlap vector and its gradient w.r.t. h (via smooth max weights).
    overlap_i = h_i*(1-h_{i+1}) for i<N-1; overlap_{N-1}=h_{N-1}*1 (shifted is 0).
    d(overlap_i)/d(h_j) is sparse.
    """
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    w = _smooth_max_grad(overlap, alpha)
    
    # Gradient of smooth_max w.r.t. h
    # d(smooth_max)/d(h_j) = sum_i w_i * d(overlap_i)/d(h_j)
    grad = np.zeros(n)
    for i in range(n):
        # overlap_i = h_i * (1 - h_{i+1}) for i < n-1
        # overlap_{n-1} = h_{n-1} * 1 (since shifted[n-1]=0)
        if i < n - 1:
            # d(overlap_i)/d(h_i) = (1 - h_{i+1})
            grad[i] += w[i] * (1.0 - h[i+1])
            # d(overlap_i)/d(h_{i+1}) = -h_i
            grad[i+1] += w[i] * (-h[i])
        else:
            # d(overlap_{n-1})/d(h_{n-1}) = 1
            grad[n-1] += w[i] * 1.0
    return overlap, grad

def _project_to_feasible(h):
    """Project h to satisfy sum=N/2 and 0<=h<=1."""
    target = N / 2.0
    h = np.clip(h, 0.0, 1.0).copy()
    s = h.sum()
    if s < 1e-10:
        return np.full(N, 0.5)
    # Water-filling style projection
    for _ in range(200):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0
        else:
            mask = h > 0.0
        if not mask.any():
            break
        h[mask] += diff / mask.sum()
        h = np.clip(h, 0.0, 1.0)
    return h

def _h_two_plateau(params):
    """Two-plateau with smooth transition.
    params: [low_val, high_val, trans_start, trans_width, trans_pos]
    h starts at low_val, transitions to high_val, stays at high_val.
    But we need integral = N/2, so we adjust.
    Actually, let's use a simpler parameterization:
    h[i] = base + amp * sigmoid((i - center)/width)
    Then project to satisfy constraints.
    """
    base, amp, center, width = params
    x = np.arange(N, dtype=float)
    # Sigmoid transition
    z = (x - center) / max(width, 1e-6)
    # Clip z to avoid overflow
    z = np.clip(z, -20, 20)
    sig = 1.0 / (1.0 + np.exp(-z))
    h = base + amp * sig
    return _project_to_feasible(h)

def _objective_two_plateau(params, alpha=50.0):
    h = _h_two_plateau(params)
    sm = _smooth_max(_overlap_vector(h), alpha)
    return sm

def _overlap_vector(h):
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    return h * (1.0 - shifted)

def _gradient_descent(h0, alpha=50.0, lr=0.01, n_iter=5000, deadline=None):
    """Projected gradient descent on h directly."""
    h = h0.copy()
    best_h = h.copy()
    best_score = _score(h)
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        overlap, grad = _overlap_and_grad_h(h, alpha)
        # Projected gradient step
        h_new = h - lr * grad
        h_new = _project_to_feasible(h_new)
        sc = _score(h_new)
        if sc < best_score:
            best_score = sc
            best_h = h_new.copy()
        h = h_new
        # Decay learning rate
        lr *= 0.999
    
    return best_h, best_score

def construct():
    deadline = time.time() + 130.0  # Safety margin
    best_h = np.full(N, 0.5)
    best_score = _score(best_h)
    
    # Phase 1: Optimize two-plateau parameterization
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        for trial in range(30):
            p0 = np.array([
                rng.uniform(0.0, 0.5),   # base
                rng.uniform(0.3, 1.0),   # amp
                rng.uniform(0.2*N, 0.8*N),  # center
                rng.uniform(2.0, 20.0)   # width
            ])
            # Simple gradient-free local search on params
            p = p0.copy()
            cur_score = _objective_two_plateau(p, alpha=50.0)
            for step in range(200):
                if time.time() > deadline - 60:
                    break
                # Random perturbation
                p_trial = p + rng.normal(0, 0.05, size=4)
                p_trial[2] = np.clip(p_trial[2], 0, N)
                p_trial[3] = np.clip(p_trial[3], 1.0, 50.0)
                s = _objective_two_plateau(p_trial, alpha=50.0)
                if s < cur_score:
                    p = p_trial
                    cur_score = s
            h = _h_two_plateau(p)
            sc = _score(h)
            if sc < best_score:
                best_score = sc
                best_h = h.copy()
    
    # Phase 2: Direct gradient descent on h coordinates
    if time.time() < deadline - 30:
        h, sc = _gradient_descent(best_h, alpha=50.0, lr=0.05, n_iter=3000, deadline=deadline - 30)
        if sc < best_score:
            best_score = sc
            best_h = h.copy()
    
    # Phase 3: Coordinate-wise refinement
    if time.time() < deadline - 10:
        rng = np.random.default_rng(123)
        h = best_h.copy()
        for _ in range(20000):
            if time.time() > deadline - 5:
                break
            idx = rng.integers(0, N)
            # Try several step sizes
            for step in [0.01, 0.05, 0.1, -0.01, -0.05, -0.1]:
                h_trial = h.copy()
                h_trial[idx] = np.clip(h_trial[idx] + step, 0.0, 1.0)
                h_trial = _project_to_feasible(h_trial)
                sc = _score(h_trial)
                if sc < best_score - 1e-12:
                    best_score = sc
                    h = h_trial.copy()
                    break
    
    return best_h

if __name__ == "__main__":
    h = construct()
    print(_score(h))
