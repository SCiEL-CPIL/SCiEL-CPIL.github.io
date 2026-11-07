"""
Model-and-optimize approach with smooth surrogate and gradient-based search.

Key insight: the objective l2/(norm_1 * norm_inf) is non-smooth due to
norm_inf = max(abs(g)). We use a log-sum-exp smooth approximation of norm_inf
to get a differentiable surrogate, then use L-BFGS-B with diverse starts and
coordinate descent polish.

Strategy:
1. Smooth surrogate: replace norm_inf with soft-max via log-sum-exp.
2. L-BFGS-B on the smooth objective from multiple diverse starts.
3. Coordinate descent polish on the exact objective.
4. Final high-precision polish pass.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
M = 2 * N - 1  # len(g) = 99
W = 1.0 / (M + 1)  # 1/100
TIME_LIMIT = 140.0  # safety margin before 150s kill

RNG = np.random.RandomState(12345)


def compute_value(h):
    """Exact objective value."""
    h = np.asarray(h, dtype=float).ravel()
    if h.size != N:
        h = h[:N]
        if h.size < N:
            h = np.pad(h, (0, N - h.size), 'constant')
    if not np.all(np.isfinite(h)):
        return 0.0
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 0.0
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 <= 0 or norm_inf <= 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)


def _smooth_neg_value(h, beta=50.0):
    """
    Negative of the smooth surrogate objective.
    Uses log-sum-exp to approximate norm_inf smoothly.
    We minimize -log(value) = -log(l2) + log(norm_1) + log(norm_inf_smooth)
    """
    h = np.asarray(h, dtype=float).ravel()
    h = np.maximum(0.0, h)
    if np.all(h == 0):
        return 1e10
    g = np.convolve(h, h)
    m = len(g)
    y = np.zeros(m + 2)
    y[1:-1] = g
    a = y[:m + 1]
    b = y[1:m + 2]
    l2 = (W / 3.0) * np.sum(a * a + a * b + b * b)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    # Smooth max via log-sum-exp
    abs_g = np.abs(g)
    max_abs = np.max(abs_g)
    if max_abs < 1e-15:
        return 1e10
    # log-sum-exp for stability
    shifted = abs_g - max_abs
    log_sum_exp = max_abs + np.log(np.sum(np.exp(shifted)))
    norm_inf_smooth = np.exp(log_sum_exp) / beta  # approximate
    # Actually use the standard soft-max: (1/beta)*log(sum(exp(beta*x)))
    # For better conditioning, use the standard formulation
    norm_inf_smooth = (1.0 / beta) * np.log(np.sum(np.exp(beta * abs_g / max_abs))) * max_abs
    # Simpler: just use the log-sum-exp directly
    norm_inf_smooth = max_abs + np.log(np.sum(np.exp(beta * (abs_g - max_abs)))) / beta
    
    if l2 <= 0 or norm_1 <= 0 or norm_inf_smooth <= 0:
        return 1e10
    
    return -(np.log(l2) - np.log(norm_1) - np.log(norm_inf_smooth))


def _smooth_neg_value_grad(h, beta=50.0):
    """
    Gradient of the smooth surrogate objective (negative log value).
    Uses finite differences for simplicity and robustness.
    """
    h = np.asarray(h, dtype=float).ravel().copy()
    h = np.maximum(0.0, h)
    eps = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        hp = h.copy()
        hp[i] += eps
        hm = h.copy()
        hm[i] = max(0.0, h[i] - eps)
        fp = _smooth_neg_value(hp, beta)
        fm = _smooth_neg_value(hm, beta)
        if hp[i] > h[i]:
            grad[i] = (fp - _smooth_neg_value(h, beta)) / eps
        else:
            grad[i] = (_smooth_neg_value(h, beta) - fm) / eps
    return grad


def _coordinate_descent(h, max_iter=500, step_init=0.1):
    """Local coordinate descent on exact value with adaptive step."""
    h = np.maximum(0.0, np.asarray(h, dtype=float).copy())
    best_val = compute_value(h)
    step = step_init
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            orig = h[i]
            for delta in [step, -step, step * 0.5, -step * 0.5, step * 2.0]:
                trial = h.copy()
                trial[i] = max(0.0, orig + delta)
                v = compute_value(trial)
                if v > best_val + 1e-12:
                    best_val = v
                    h = trial
                    improved = True
                    break
        if not improved:
            step *= 0.7
            if step < 1e-8:
                break
    return h, best_val


def _generate_diverse_starts(n_starts=15):
    """Generate diverse initial points for the optimizer."""
    x = np.linspace(0, N - 1, N)
    starts = []
    
    # Uniform
    starts.append(np.ones(N))
    
    # Centered Gaussian
    starts.append(np.exp(-0.5 * ((x - N / 2) / (N / 4)) ** 2))
    
    # Two bumps
    starts.append(np.exp(-0.5 * ((x - N * 0.25) / (N / 6)) ** 2) + 
                  np.exp(-0.5 * ((x - N * 0.75) / (N / 6)) ** 2))
    
    # Three bumps
    starts.append(np.exp(-0.5 * ((x - N * 0.2) / (N / 7)) ** 2) + 
                  np.exp(-0.5 * ((x - N * 0.5) / (N / 7)) ** 2) + 
                  np.exp(-0.5 * ((x - N * 0.8) / (N / 7)) ** 2))
    
    # Cosine window
    starts.append(0.5 * (1 + np.cos(2 * np.pi * x / N)))
    
    # Triangle
    starts.append(1 - 2 * np.abs(x - (N - 1) / 2) / (N - 1))
    
    # Sparse spikes
    for _ in range(4):
        h = np.zeros(N)
        k = RNG.randint(2, 8)
        idx = RNG.choice(N, size=k, replace=False)
        h[idx] = RNG.uniform(0.5, 2.0, k)
        starts.append(h)
    
    # Random smooth profiles
    for _ in range(n_starts - len(starts)):
        freqs = RNG.randint(1, 8, size=4)
        phases = RNG.uniform(0, 2 * np.pi, size=4)
        amps = RNG.uniform(0.3, 1.0, size=4)
        prof = np.zeros(N)
        for f_, p_, a_ in zip(freqs, phases, amps):
            prof += a_ * np.cos(2 * np.pi * f_ * x / N + p_)
        starts.append(np.maximum(0.05, prof))
    
    return starts


def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_val = 0.0
    best_h = np.ones(N)
    
    def track(h):
        nonlocal best_val, best_h
        h = np.maximum(0.0, np.asarray(h, dtype=float).ravel())
        if h.size != N:
            h = h[:N]
            if h.size < N:
                h = np.pad(h, (0, N - h.size), 'constant')
        v = compute_value(h)
        if v > best_val:
            best_val = v
            best_h = h.copy()
    
    # Generate diverse starts
    starts = _generate_diverse_starts(15)
    
    # Evaluate all starts
    for s in starts:
        track(s)
    
    # L-BFGS-B optimization from multiple starts
    n_lbfgs = 8
    lbfgs_starts = starts[:n_lbfgs]
    
    for i, s in enumerate(lbfgs_starts):
        if time.time() > deadline - 40:
            break
        
        # Use different beta values for diversity
        beta = 30.0 + 10.0 * (i % 3)
        
        try:
            res = minimize(
                _smooth_neg_value,
                s.copy(),
                args=(beta,),
                method='L-BFGS-B',
                bounds=[(0, None)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            h_opt = np.maximum(0.0, res.x)
            track(h_opt)
        except Exception:
            pass
    
    # Coordinate descent polish from best and nearby
    n_cd = 10
    for i in range(n_cd):
        if time.time() > deadline - 15:
            break
        if i == 0:
            h0 = best_h.copy()
        elif i == 1:
            h0 = starts[0].copy()
        elif i == 2:
            h0 = starts[1].copy()
        else:
            h0 = best_h + RNG.normal(0.0, 0.15, N)
            h0 = np.maximum(0.0, h0)
        h_opt, v_opt = _coordinate_descent(h0, max_iter=300, step_init=0.08)
        track(h_opt)
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        h_opt, v_opt = _coordinate_descent(best_h, max_iter=800, step_init=0.02)
        track(h_opt)
    
    # Very fine final polish
    if time.time() < deadline - 2:
        h_opt, v_opt = _coordinate_descent(best_h, max_iter=1000, step_init=0.005)
        track(h_opt)
    
    best_h = np.maximum(0.0, best_h)
    best_h = np.nan_to_num(best_h, nan=0.0, posinf=1e6, neginf=0.0)
    return best_h.tolist()


if __name__ == "__main__":
    h = construct()
    print("Score:", compute_value(h))
