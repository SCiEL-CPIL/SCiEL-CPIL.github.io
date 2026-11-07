import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
w = 1.0 / 100.0  # spacing on [-0.5, 0.5] with m+2=102 points

def compute_value(h):
    h = np.asarray(h, dtype=float)
    g = np.convolve(h, h)
    m = len(g)  # 99
    y = np.concatenate(([0.0], g, [0.0]))  # length m+2 = 101
    sq = y[:m+1]**2 + y[:m+1]*y[1:m+2] + y[1:m+2]**2
    l2 = (w / 3.0) * np.sum(sq)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_fast(h):
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    # Use the exact formula: l2 = (w/3) * sum(y[i]^2 + y[i]*y[i+1] + y[i+1]^2) for i=0..m
    # where y = [0, g[0], ..., g[m-1], 0]
    # This equals (w/3) * (g[0]^2 + sum_{k=1}^{m-1} (g[k-1]^2 + g[k-1]*g[k] + g[k]^2) + g[m-1]^2)
    # = (w/3) * (2*sum(g^2) + sum(g[k-1]*g[k] for k=1..m-1))
    g_sq_sum = np.dot(g, g)
    cross_sum = np.dot(g[:-1], g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def compute_value_precise(h):
    """High-precision evaluation using full formula."""
    h = np.asarray(h, dtype=np.float64)
    g = np.convolve(h, h)
    m = len(g)
    y = np.concatenate(([0.0], g, [0.0]))
    term1 = y[:m+1]**2
    term2 = y[:m+1] * y[1:m+2]
    term3 = y[1:m+2]**2
    sq = term1 + term2 + term3
    l2 = (w / 3.0) * np.sum(sq, dtype=np.float64)
    norm_1 = np.sum(np.abs(g), dtype=np.float64) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value(log_h):
    h = np.exp(log_h)
    return -compute_value_fast(h)

def neg_value_direct(h):
    return -compute_value_fast(h)

def coordinate_polish(h, max_rounds=50, tol=1e-10):
    """Cyclic coordinate descent in direct space with non-negativity constraint."""
    h = np.maximum(h.copy(), 0.0)
    best_val = compute_value_fast(h)
    for _ in range(max_rounds):
        improved = False
        for i in range(N):
            old = h[i]
            # Try small perturbations around current value
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001, 0.1, -0.1]:
                trial = h.copy()
                trial[i] = max(0.0, old + delta)
                v = compute_value_fast(trial)
                if v > best_val:
                    h = trial
                    best_val = v
                    improved = True
                    break
        if not improved:
            break
    return h, best_val

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 1.0

    rng = np.random.RandomState(42)

    def try_optimize_log(start_log_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value,
                start_log_h,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_optimize_direct(start_h, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            res = minimize(
                neg_value_direct,
                start_h,
                method='L-BFGS-B',
                bounds=[(1e-10, 100.0)] * N,
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
            )
            h_cand = np.maximum(res.x, 0.0)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_differential_evolution(bounds_log, maxiter):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            # DE in log-space
            def de_obj(log_h):
                h = np.exp(log_h)
                return -compute_value_fast(h)
            
            # Use bounds for log_h
            lb = bounds_log[0]
            ub = bounds_log[1]
            res = differential_evolution(
                de_obj,
                [(lb, ub)] * N,
                maxiter=maxiter,
                tol=1e-12,
                seed=42,
                polish=True,
                update_interval=10
            )
            h_cand = np.exp(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    def try_coordinate_descent(start_h, max_rounds=30):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            h_cand, v_cand = coordinate_polish(start_h, max_rounds=max_rounds)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0

    # Start from uniform
    try_optimize_log(np.zeros(N), maxiter=8000)
    
    # Try a broad differential evolution search in log-space
    if time.time() < deadline - 60.0:
        try_differential_evolution((-5.0, 5.0), maxiter=100)
    
    # Multi-start perturbations
    n_starts = 25
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        if i == 0:
            # Smooth random
            x = rng.randn(N)
            kernel = np.ones(7)/7.0
            x = np.convolve(x, kernel, mode='same')
            x = x - x.mean()
            x = 0.3 * x
        elif i == 1:
            x = np.sin(np.pi * np.arange(N) / N) * 0.5
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.4
        elif i == 3:
            x = 1 - 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean()
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/20.0)
            x = x - x.mean()
        elif i == 5:
            x = np.where(np.arange(N) < N//2, -0.3, 0.3) + 0.1*rng.randn(N)
        elif i == 6:
            x = (1 - 2*np.abs(np.arange(N)/N - 0.5))**2
            x = x - x.mean()
        elif i == 7:
            x = np.exp(-((np.arange(N)-N//4)**2)/8.0) + np.exp(-((np.arange(N)-3*N//4)**2)/8.0)
            x = x - x.mean()
        elif i == 8:
            x = np.linspace(-0.4, 0.4, N)
        elif i == 9:
            x = 0.3*np.cos(4*np.pi*np.arange(N)/N)
        elif i == 10:
            x = np.exp(-((np.arange(N)-N//2)**2)/5.0)
            x = x - x.mean()
        elif i == 11:
            x = np.exp(-((np.arange(N)-N//2)**2)/100.0)
            x = x - x.mean()
        elif i == 12:
            x = 2*np.abs(np.arange(N)/N - 0.5)
            x = x - x.mean()
        elif i == 13:
            x = 0.3*np.sin(3*np.pi*np.arange(N)/N)
        elif i == 14:
            x = np.where((np.arange(N) > N//4) & (np.arange(N) < 3*N//4), 0.4, -0.2)
        elif i == 15:
            # High frequency oscillation
            x = 0.2*np.sin(10*np.pi*np.arange(N)/N)
        elif i == 16:
            # Asymmetric
            x = np.sin(np.pi * np.arange(N) / (2*N)) - 0.5
        elif i == 17:
            # Two peaks offset
            x = np.exp(-((np.arange(N)-N//3)**2)/15.0) + np.exp(-((np.arange(N)-2*N//3)**2)/15.0)
            x = x - x.mean()
        elif i == 18:
            # S-shaped
            x = np.tanh(np.linspace(-3, 3, N))
            x = x - x.mean()
        elif i == 19:
            # Triple peak
            x = np.exp(-((np.arange(N)-N//6)**2)/10.0) + np.exp(-((np.arange(N)-N//2)**2)/10.0) + np.exp(-((np.arange(N)-5*N//6)**2)/10.0)
            x = x - x.mean()
        elif i == 20:
            # Plateau
            x = np.where((np.arange(N) > N//5) & (np.arange(N) < 4*N//5), 0.3, -0.1)
        elif i == 21:
            # Smooth bump with noise
            x = np.exp(-((np.arange(N)-N//2)**2)/30.0) + 0.05*rng.randn(N)
            x = x - x.mean()
        elif i == 22:
            # Cosine envelope
            x = np.cos(np.pi*np.arange(N)/N) * np.exp(-((np.arange(N)-N//2)**2)/50.0)
            x = x - x.mean()
        elif i == 23:
            # Step-like smooth
            x = np.tanh(5*(np.arange(N)/N - 0.5)) * 0.4
        else:
            x = 0.3*rng.randn(N)
        
        x = np.maximum(x, -6.0)
        try_optimize_log(x, maxiter=4000)
    
    # Coordinate descent polish after each major phase
    if time.time() < deadline - 35.0:
        try_coordinate_descent(best_h.copy(), max_rounds=40)
    
    # Final polish rounds
    if time.time() < deadline - 30.0:
        try_optimize_direct(best_h.copy(), maxiter=10000)
    
    if time.time() < deadline - 15.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-10)), maxiter=8000)
    
    # One more DE from around best
    if time.time() < deadline - 20.0:
        log_best = np.log(np.maximum(best_h, 1e-10))
        bounds = [(max(log_best[i]-3, -6.0), min(log_best[i]+3, 6.0)) for i in range(N)]
        # Convert to simple bounds for DE
        lb = min(b[0] for b in bounds)
        ub = max(b[1] for b in bounds)
        try_differential_evolution((lb, ub), maxiter=80)
    
    # Final coordinate descent + fine polish
    if time.time() < deadline - 10.0:
        try_coordinate_descent(best_h.copy(), max_rounds=60)
    
    if time.time() < deadline - 5.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-10)), maxiter=5000)
    
    # Very final coordinate polish
    if time.time() < deadline - 2.0:
        try_coordinate_descent(best_h.copy(), max_rounds=100)

    return list(np.asarray(best_h, dtype=float))
