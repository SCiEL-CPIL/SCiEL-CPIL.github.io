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
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_inf == 0 or norm_1 == 0:
        return 0.0
    return l2 / (norm_1 * norm_inf)

def neg_value_smooth(log_h):
    """Smoothed objective in log-space using log-sum-exp for norm_inf."""
    h = np.exp(log_h)
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g) + 1e-15
    log_max = np.logaddexp.reduce(np.log(abs_g))
    norm_inf_smooth = np.exp(log_max)
    if norm_inf_smooth < 1e-30 or norm_1 < 1e-30:
        return 0.0
    return -l2 / (norm_1 * norm_inf_smooth)

def neg_value_direct(h):
    return -compute_value_fast(h)

def neg_value_direct_smooth(h):
    """Smoothed objective in direct space."""
    h = np.maximum(h, 1e-12)
    g = np.convolve(h, h)
    m = len(g)
    g_sq_sum = np.sum(g**2)
    cross_sum = np.sum(g[:-1] * g[1:]) if m > 1 else 0.0
    l2 = (w / 3.0) * (2.0 * g_sq_sum + cross_sum)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    abs_g = np.abs(g) + 1e-15
    log_max = np.logaddexp.reduce(np.log(abs_g))
    norm_inf_smooth = np.exp(log_max)
    if norm_inf_smooth < 1e-30 or norm_1 < 1e-30:
        return 0.0
    return -l2 / (norm_1 * norm_inf_smooth)

def spectral_to_h(params):
    """Convert spectral parameters to nonnegative step heights.
    
    params layout:
      - First K*3 entries: for each of K sinusoids, (amplitude, frequency, phase)
      - Last entry: DC offset (bias)
    
    h_i = max(0, bias + sum_k amp_k * sin(freq_k * i + phase_k))
    """
    K = len(params) // 3 - 1  # number of sinusoids
    bias = params[-1]
    h = np.full(N, bias)
    for k in range(K):
        base = k * 3
        amp = params[base]
        freq = params[base + 1]
        phase = params[base + 2]
        i = np.arange(N)
        h += amp * np.sin(freq * i + phase)
    return np.maximum(h, 0.0)

def neg_value_spectral(params):
    h = spectral_to_h(params)
    return -compute_value_fast(h)

def neg_value_spectral_smooth(params):
    h = spectral_to_h(params)
    return -compute_value_direct_smooth(h)

def _hump_init():
    """Concentrated hump in the center."""
    x = np.arange(N)
    center = N / 2.0
    h = np.exp(-((x - center)**2) / (2.0 * (N/8.0)**2))
    return h

def _two_peak_init():
    """Two peaks at quarter and three-quarter positions."""
    x = np.arange(N)
    h = np.exp(-((x - N/4.0)**2) / 8.0) + np.exp(-((x - 3*N/4.0)**2) / 8.0)
    return h

def _ramp_init():
    """Linear ramp."""
    return np.linspace(0.1, 1.0, N)

def _bump_init():
    """Single broad bump."""
    x = np.arange(N)
    h = np.exp(-((x - N/2.0)**2) / (2.0 * (N/4.0)**2))
    return h

def _triangle_init():
    """Triangle wave."""
    x = np.arange(N) / N
    h = 1.0 - 2.0 * np.abs(x - 0.5)
    return h

def _uniform_init():
    return np.ones(N)

def _random_smooth(rng):
    """Random smooth function."""
    x = rng.randn(N)
    kernel = np.ones(9) / 9.0
    x = np.convolve(x, kernel, mode='same')
    x = x - x.mean()
    return np.maximum(0.1 + x, 0.0)

def construct():
    deadline = time.time() + 130.0  # safety margin before 150s kill
    rng = np.random.RandomState(42)
    
    best_h = np.ones(N)
    best_val = compute_value(best_h)  # should be 1.0
    
    def try_optimize_log(start_log_h, maxiter, method='L-BFGS-B', use_smooth=True):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            if use_smooth:
                obj = neg_value_smooth
            else:
                obj = lambda lh: -compute_value_fast(np.exp(lh))
            
            res = minimize(
                obj,
                start_log_h,
                method=method,
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
                neg_value_direct_smooth,
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
    
    def try_spectral_optimize(K, maxiter):
        """Optimize using spectral parametrization with K sinusoids."""
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            # Parameters: K*(amp, freq, phase) + bias
            n_params = K * 3 + 1
            # Initialize
            params0 = np.zeros(n_params)
            for k in range(K):
                params0[k*3] = 0.3  # amplitude
                params0[k*3+1] = (k+1) * 2 * np.pi / N  # frequency
                params0[k*3+2] = rng.uniform(0, 2*np.pi)  # phase
            params0[-1] = 1.0  # bias
            
            res = minimize(
                neg_value_spectral_smooth,
                params0,
                method='L-BFGS-B',
                bounds=[(-5, 5)] * (K*3) + [(0.01, 10.0)],
                options={'maxiter': maxiter, 'maxfun': maxiter * 5, 'ftol': 1e-14, 'gtol': 1e-12, 'eps': 1e-8}
            )
            h_cand = spectral_to_h(res.x)
            v_cand = compute_value_fast(h_cand)
            if v_cand > best_val:
                best_val = v_cand
                best_h = h_cand.copy()
        except Exception:
            pass
        return time.time() - t0
    
    def try_differential_evolution(bounds_log, maxiter, popsize=15):
        nonlocal best_h, best_val
        t0 = time.time()
        try:
            def de_obj(log_h):
                h = np.exp(log_h)
                return -compute_value_fast(h)
            
            lb = bounds_log[0]
            ub = bounds_log[1]
            res = differential_evolution(
                de_obj,
                [(lb, ub)] * N,
                maxiter=maxiter,
                tol=1e-12,
                seed=42,
                polish=True,
                popsize=popsize,
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
    
    def custom_local_search(h, deadline, max_rounds=300):
        """Coordinate descent with adaptive step sizes."""
        nonlocal best_h, best_val
        best_h = h.copy()
        best_val = compute_value_fast(h)
        n = len(h)
        
        for rnd in range(max_rounds):
            if time.time() > deadline:
                break
            
            improved = False
            # Adaptive step sizes
            step_sizes = [0.1, 0.05, 0.01, 0.005, 0.001, -0.1, -0.05, -0.01, -0.005, -0.001]
            
            for i in range(n):
                if time.time() > deadline:
                    break
                
                orig = best_h[i]
                for delta in step_sizes:
                    cand = best_h.copy()
                    cand[i] = max(0.0, orig + delta)
                    v = compute_value_fast(cand)
                    if v > best_val + 1e-14:
                        best_val = v
                        best_h = cand
                        improved = True
                        break
            
            if not improved:
                # Try Nelder-Mead on random subset
                subset_size = min(8, n)
                subset = np.random.choice(n, size=subset_size, replace=False)
                sub_h = best_h[subset]
                try:
                    def sub_obj(x):
                        full = best_h.copy()
                        full[subset] = np.maximum(x, 0.0)
                        return -compute_value_fast(full)
                    
                    res = minimize(
                        sub_obj,
                        sub_h,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    h_cand = best_h.copy()
                    h_cand[subset] = np.maximum(res.x, 0.0)
                    v_cand = compute_value_fast(h_cand)
                    if v_cand > best_val:
                        best_val = v_cand
                        best_h = h_cand.copy()
                        improved = True
                except Exception:
                    pass
            
            if not improved and rnd > 15:
                break
        
        return best_h, best_val
    
    # Phase 1: Try various initializations with log-space optimization
    inits = [
        _uniform_init,
        _hump_init,
        _two_peak_init,
        _ramp_init,
        _bump_init,
        _triangle_init,
    ]
    
    for init_func in inits:
        if time.time() > deadline - 80.0:
            break
        h0 = init_func()
        log_h0 = np.log(np.maximum(h0, 1e-8))
        try_optimize_log(log_h0, maxiter=5000)
    
    # Phase 2: Spectral optimization with different numbers of sinusoids
    for K in [3, 5, 7]:
        if time.time() > deadline - 65.0:
            break
        try_spectral_optimize(K, maxiter=3000)
    
    # Phase 3: Differential evolution for global search
    if time.time() < deadline - 50.0:
        try_differential_evolution((-4.0, 4.0), maxiter=60, popsize=12)
    
    # Phase 4: More diverse random starts
    n_starts = 12
    for i in range(n_starts):
        if time.time() > deadline - 40.0:
            break
        
        if i == 0:
            x = _random_smooth(rng)
        elif i == 1:
            x = np.sin(np.pi * np.arange(N) / N) * 0.5 + 0.5
        elif i == 2:
            x = np.cos(2*np.pi*np.arange(N)/N) * 0.4 + 0.6
        elif i == 3:
            x = (1 - 2*np.abs(np.arange(N)/N - 0.5))**2 + 0.1
        elif i == 4:
            x = np.exp(-((np.arange(N)-N//2)**2)/20.0) + 0.1
        elif i == 5:
            x = np.where(np.arange(N) < N//2, 0.3, 0.7) + 0.05*rng.randn(N)
        elif i == 6:
            x = np.exp(-((np.arange(N)-N//4)**2)/8.0) + np.exp(-((np.arange(N)-3*N//4)**2)/8.0) + 0.1
        elif i == 7:
            x = np.linspace(0.2, 0.8, N)
        elif i == 8:
            x = 0.3*np.cos(4*np.pi*np.arange(N)/N) + 0.5
        elif i == 9:
            x = np.exp(-((np.arange(N)-N//2)**2)/5.0) + 0.1
        elif i == 10:
            x = 0.3*np.sin(3*np.pi*np.arange(N)/N) + 0.5
        else:
            x = 0.3*rng.randn(N) + 0.5
        
        x = np.maximum(x, 1e-6)
        log_x = np.log(x)
        try_optimize_log(log_x, maxiter=4000)
    
    # Phase 5: Fine-tuning with local search
    if time.time() < deadline - 25.0:
        best_h, best_val = custom_local_search(best_h, deadline - 20.0, max_rounds=200)
    
    # Phase 6: Final polish with direct-space optimization
    if time.time() < deadline - 18.0:
        try_optimize_direct(best_h.copy(), maxiter=10000)
    
    # Phase 7: Another round of log-space polish
    if time.time() < deadline - 12.0:
        try_optimize_log(np.log(np.maximum(best_h, 1e-10)), maxiter=12000)
    
    # Phase 8: Spectral refinement from current best
    if time.time() < deadline - 8.0:
        # Try to fit current best with spectral representation
        for K in [4, 6]:
            if time.time() > deadline - 5.0:
                break
            try_spectral_optimize(K, maxiter=2000)
    
    # Phase 9: Final local search
    if time.time() < deadline - 3.0:
        best_h, best_val = custom_local_search(best_h, deadline - 2.0, max_rounds=100)
    
    # Ensure nonnegativity and finiteness
    best_h = np.maximum(np.asarray(best_h, dtype=float), 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return list(best_h)
