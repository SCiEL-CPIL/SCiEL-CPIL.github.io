"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 138.0  # safety margin before 150s kill

def _time_remaining():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_cross_correlation_min(h):
    """Compute min over shifts of R(s) = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k] for k=1..n-1."""
    n = len(h)
    min_corr = float('inf')
    for k in range(1, n):
        corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
        if corr < min_corr:
            min_corr = corr
    return min_corr

def _project_box_sum(v, target_sum):
    """Project v onto {x : 0 <= x <= 1, sum(x) = target_sum} using iterative bisection on threshold."""
    n = len(v)
    # Bisection on Lagrange multiplier for sum constraint, with box clipping
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        w = np.clip(v - mid, 0.0, 1.0)
        s = np.sum(w)
        if s > target_sum:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - (lo + hi) / 2.0, 0.0, 1.0)
    # Fine-tune to hit exact sum
    diff = target_sum - np.sum(w)
    if abs(diff) > 1e-14:
        mask = (w > 0) & (w < 1)
        cnt = np.sum(mask)
        if cnt > 0:
            w[mask] += diff / cnt
            w = np.clip(w, 0.0, 1.0)
    return w

def _fista_optimize(h_init, target_sum, max_iters=2000, mu=None):
    """FISTA for minimizing max_k (1 - R_k) subject to 0<=h<=1, sum=h_target.
    
    Equivalent to: minimize t s.t. R_k >= 1-t for all k, 0<=h<=1, sum=h_target.
    We use the form: minimize f(h) + g(h) where
      f(h) = max_k (1 - R_k(h))  [convex, non-smooth]
      g(h) = indicator of {0<=h<=1, sum=h_target}
    
    FISTA: h_{k+1} = prox_g( y_k - (1/mu) * subgrad_f(y_k) )
    where y_k = h_k + (t_{k-1}-1)/t_{k+1} * (h_k - h_{k-1})
    
    For f(h) = max_k (1 - R_k), a subgradient is:
      -dR_k*/dh where k* = argmin R_k
    
    The Lipschitz constant of R_k is bounded by ||dR_k/dh||_2.
    We use a conservative mu estimate.
    """
    n = len(h_init)
    h = h_init.copy()
    h_prev = h_init.copy()
    t_curr = 1.0
    
    if mu is None:
        # Estimate Lipschitz constant: ||dR_k/dh||_2 for each k
        # R_k = (2/n) * sum_{i=0}^{n-k-1} h[i]*h[i+k]
        # dR_k/dh is a matrix; its operator norm is at most (2/n) * (n-k) * 2 (since h in [0,1])
        # More carefully: ||dR_k/dh||_2 <= (2/n) * 2 * sqrt(n-k) * ... 
        # Conservative bound: (2/n) * 2 * (n-1) for worst case
        mu = (2.0 / n) * 2.0 * n  # very conservative
        # Actually let's be more precise:
        # dR_k/dh[j] = (2/n)*h[j+k] (if j < n-k) + (2/n)*h[j-k] (if j >= k)
        # The Jacobian J has J[i,j] = (2/n)*h[j+k] or (2/n)*h[j-k]
        # ||J||_2 <= (2/n) * max(h) * sqrt(2*(n-k)) approximately
        # Let's use mu = (2/n) * 1.0 * sqrt(2*n) * 1.5 for safety
        mu = (2.0 / n) * 1.5 * np.sqrt(2.0 * n)
        mu = max(mu, 0.5)
    
    best_h = h.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(h)
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
        
        # FISTA momentum
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_curr * t_curr)) / 2.0
        y = h + ((t_curr - 1.0) / t_next) * (h - h_prev)
        
        # Compute subgradient of f(y) = max_k (1 - R_k(y))
        n_local = len(y)
        # Find k* = argmin R_k
        min_corr = float('inf')
        k_star = 1
        for k in range(1, n_local):
            corr = np.dot(y[:n_local-k], y[k:]) * (2.0 / n_local)
            if corr < min_corr:
                min_corr = corr
                k_star = k
        
        # Subgradient of f at y: -dR_{k_star}/dy
        # dR_k/dh[j] = (2/n)*h[j+k] for j < n-k, plus (2/n)*h[j-k] for j >= k
        subgrad = np.zeros(n_local)
        ks = k_star
        subgrad[:n_local-ks] -= (2.0 / n_local) * y[ks:]
        subgrad[ks:] -= (2.0 / n_local) * y[:n_local-ks]
        
        # Gradient step
        z = y - (1.0 / mu) * subgrad
        
        # Proximal step: project onto {0<=x<=1, sum=target_sum}
        h_new = _project_box_sum(z, target_sum)
        
        # Update
        h_prev = h
        h = h_new
        t_curr = t_next
        
        # Check improvement
        obj = 1.0 - _compute_cross_correlation_min(h)
        if obj < best_obj:
            best_obj = obj
            best_h = h.copy()
    
    return best_h, best_obj

def _local_polish(h_init, target_sum, max_iters=500):
    """Local polishing using subgradient descent with adaptive step size on the hard min."""
    n = len(h_init)
    h = h_init.copy()
    best_obj = 1.0 - _compute_cross_correlation_min(h)
    best_h = h.copy()
    step = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if _time_remaining() < 3.0:
            break
        
        # Find argmin correlation
        min_corr = float('inf')
        k_star = 1
        for k in range(1, n):
            corr = np.dot(h[:n-k], h[k:]) * (2.0 / n)
            if corr < min_corr:
                min_corr = corr
                k_star = k
        
        # Subgradient direction to increase R_{k_star}
        grad = np.zeros(n)
        ks = k_star
        grad[:n-ks] += (2.0 / n) * h[ks:]
        grad[ks:] += (2.0 / n) * h[:n-ks]
        
        # Move in +grad direction (to increase min correlation)
        h_new = h + step * grad
        h_new = _project_box_sum(h_new, target_sum)
        
        new_obj = 1.0 - _compute_cross_correlation_min(h_new)
        if new_obj < best_obj - 1e-10:
            best_obj = new_obj
            best_h = h_new.copy()
            h = h_new
            no_improve = 0
            step = min(step * 1.05, 0.05)
        else:
            h = h_new  # still update to keep momentum
            no_improve += 1
            step *= 0.85
            if step < 1e-5:
                break
    
    return best_h, best_obj

def construct():
    """Construct step heights minimizing maximum overlap."""
    n = N
    target_sum = n / 2.0
    
    best_x = np.full(n, 0.5)
    best_obj = 1.0 - _compute_cross_correlation_min(best_x)
    
    np.random.seed(42)
    
    # Generate diverse initializations
    restarts = []
    restarts.append(np.full(n, 0.5))
    for frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        k = max(1, int(n * frac))
        if k < n:
            val = target_sum / (n - k)
            if 0 <= val <= 1:
                restarts.append(np.concatenate([np.full(k, 0.0), np.full(n-k, val)]))
                restarts.append(np.concatenate([np.full(k, val), np.full(n-k, 0.0)]))
    # Alternating
    restarts.append(np.array([0.0 if i % 2 == 0 else 1.0 for i in range(n)]))
    restarts.append(np.array([1.0 if i % 2 == 0 else 0.0 for i in range(n)]))
    for period in [3, 4, 5, 6, 7]:
        for offset in range(period):
            pattern = np.array([1.0 if (i + offset) % period < period // 2 else 0.0 for i in range(n)])
            mean_val = np.mean(pattern)
            if 0.01 < mean_val < 0.99:
                pattern = pattern / (2.0 * mean_val)
                if np.all((pattern >= 0) & (pattern <= 1)):
                    restarts.append(pattern)
    # Random
    for _ in range(10):
        restarts.append(np.random.uniform(0, 1, n))
    # Sinusoidal
    x = np.linspace(0, 2 * np.pi, n)
    for freq in [1, 2, 3, 4, 5, 6, 7, 8]:
        for amp in [0.2, 0.3, 0.4]:
            restarts.append(np.clip(0.5 + amp * np.sin(freq * x), 0, 1))
            restarts.append(np.clip(0.5 + amp * np.cos(freq * x), 0, 1))
    # Multi-frequency
    for _ in range(5):
        freqs = np.random.choice([1,2,3,4,5,6,7,8], size=3, replace=False)
        amps = np.random.uniform(0.1, 0.3, size=3)
        phases = np.random.uniform(0, 2*np.pi, size=3)
        sig = np.zeros(n)
        for f, a, p in zip(freqs, amps, phases):
            sig += a * np.sin(f * x + p)
        restarts.append(np.clip(0.5 + sig, 0, 1))
    
    # Run FISTA on each restart
    for x0 in restarts:
        if _time_remaining() < 8.0:
            break
        
        x0_proj = _project_box_sum(x0, target_sum)
        try:
            h_opt, obj = _fista_optimize(x0_proj, target_sum, max_iters=1500)
            if obj < best_obj:
                best_obj = obj
                best_x = h_opt
        except Exception:
            pass
    
    # Polish the best solution
    if _time_remaining() > 8.0:
        h_pol, obj_pol = _local_polish(best_x, target_sum, max_iters=500)
        if obj_pol < best_obj:
            best_obj = obj_pol
            best_x = h_pol
    
    # Final FISTA polish with more iterations
    if _time_remaining() > 8.0:
        h_opt2, obj2 = _fista_optimize(best_x, target_sum, max_iters=3000)
        if obj2 < best_obj:
            best_obj = obj2
            best_x = h_opt2
    
    # Ensure exact sum and bounds
    best_x = np.clip(best_x, 0, 1)
    diff = target_sum - np.sum(best_x)
    if abs(diff) > 1e-12:
        mask = (best_x > 0) & (best_x < 1)
        cnt = np.sum(mask)
        if cnt > 0:
            best_x[mask] += diff / cnt
            best_x = np.clip(best_x, 0, 1)
    
    return best_x
# EVOLVE-BLOCK-END
