import numpy as np
import time
import scipy.optimize

N = 400

def _autoconv_lin(x):
    """Compute linear autoconvolution using FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]

def _metric_std(x):
    """Standard metric: max |c_k| / S^2 including k=0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c))) / (s * s)

def _metric_offdiag(x):
    """Off-diagonal metric: max |c_k| for k!=0 / S^2."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:]))) / (s * s)
    return 0.0

def _zadoff_chu_real(n, q):
    """
    Generate the real part of a Zadoff-Chu-like sequence with quadratic phase.
    x_k = cos(2*pi*q*k*(k+1)/(2*n) + phase0) for k=0..n-1
    We optimize q and phase0 to minimize autoconvolution peaks.
    """
    k = np.arange(n)
    # Quadratic chirp: phase = 2*pi*q*k*(k+1)/(2*n)
    # For Zadoff-Chu: z_k = exp(-j*pi*q*k*(k+1)/n)
    # Real part: cos(-pi*q*k*(k+1)/n)
    phase = np.pi * q * k * (k + 1) / n
    x = np.cos(phase)
    return x

def _optimize_zadoff_chu_params(n, q_init=0.5, phase_init=0.0, max_iters=100, deadline=None):
    """
    Optimize q and phase0 to minimize the autoconvolution metric.
    x_k = cos(pi * q * k * (k+1) / n + phase0)
    """
    def objective(params):
        q, phase0 = params
        k = np.arange(n)
        phase = np.pi * q * k * (k + 1) / n + phase0
        x = np.cos(phase)
        # Project to binary for evaluation? Or use continuous?
        # The task asks for signed step heights, which can be continuous.
        # But the metric is max|c|/S^2.
        # Let's evaluate the continuous version first.
        return _metric_std(x)
    
    # Use Nelder-Mead or L-BFGS-B
    # q is in [0, 1] (mod 1 effectively, due to periodicity)
    # phase0 is in [0, 2*pi]
    
    best_x = None
    best_val = float('inf')
    
    # Grid search for initial q and phase
    q_vals = np.linspace(0, 1, 50)
    phase_vals = np.linspace(0, 2*np.pi, 20)
    
    for q in q_vals:
        if deadline is not None and time.time() > deadline:
            break
        for ph in phase_vals:
            if deadline is not None and time.time() > deadline:
                break
            k = np.arange(n)
            phase = np.pi * q * k * (k + 1) / n + ph
            x = np.cos(phase)
            v = _metric_std(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    
    if best_x is None:
        return np.ones(n), float('inf')
    
    # Refine with local optimization
    k = np.arange(n)
    
    def obj(params):
        q, ph = params
        phase = np.pi * q * k * (k + 1) / n + ph
        x = np.cos(phase)
        return _metric_std(x)
    
    # Find initial best q, ph from grid
    # Re-derive best q, ph
    best_q, best_ph = q_init, phase_init
    # Actually, let's just use the best_x found and try to refine it?
    # No, we need to optimize q and ph.
    # Let's find the best q, ph from the grid search above properly.
    
    best_q, best_ph = 0.5, 0.0
    min_v = float('inf')
    for q in q_vals:
        for ph in phase_vals:
            k = np.arange(n)
            phase = np.pi * q * k * (k + 1) / n + ph
            x = np.cos(phase)
            v = _metric_std(x)
            if v < min_v:
                min_v = v
                best_q = q
                best_ph = ph
    
    x0 = np.array([best_q, best_ph])
    
    try:
        res = scipy.optimize.minimize(obj, x0, method='Nelder-Mead', 
                                       options={'maxiter': max_iters, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < min_v:
            q_opt, ph_opt = res.x
            k = np.arange(n)
            phase = np.pi * q_opt * k * (k + 1) / n + ph_opt
            x_opt = np.cos(phase)
            return x_opt, res.fun
    except Exception:
        pass
    
    k = np.arange(n)
    phase = np.pi * best_q * k * (k + 1) / n + best_ph
    x_opt = np.cos(phase)
    return x_opt, min_v

def _local_search_binary(x, max_iters=2000, deadline=None):
    """
    Local search on binary sequence.
    """
    x = np.sign(x).astype(float)
    x[x == 0] = 1.0
    current_val = _metric_std(x)
    n = len(x)
    rng = np.random.default_rng(123)
    
    # Precompute autoconvolution
    c = _autoconv_lin(x)
    s = np.sum(x)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random flip
        i = rng.integers(0, n)
        x_trial = x.copy()
        x_trial[i] *= -1.0
        
        v_trial = _metric_std(x_trial)
        if v_trial < current_val:
            x = x_trial
            current_val = v_trial
            c = _autoconv_lin(x)
            s = np.sum(x)
    
    return x, current_val

def _simulated_annealing(x, max_iters=5000, deadline=None):
    """
    Simulated annealing for binary sequence optimization.
    """
    x = np.sign(x).astype(float)
    x[x == 0] = 1.0
    current_val = _metric_std(x)
    best_x = x.copy()
    best_val = current_val
    n = len(x)
    rng = np.random.default_rng(456)
    
    T = 0.1
    T_min = 0.001
    cooling = 0.999
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        i = rng.integers(0, n)
        x_trial = x.copy()
        x_trial[i] *= -1.0
        
        v_trial = _metric_std(x_trial)
        delta = v_trial - current_val
        
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            x = x_trial
            current_val = v_trial
            if v_trial < best_val:
                best_val = v_trial
                best_x = x.copy()
        
        T = max(T * cooling, T_min)
    
    return best_x, best_val

def construct():
    deadline = time.time() + 140.0
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Phase 1: Zadoff-Chu optimization
    if time.time() < deadline - 60:
        x_zc, v_zc = _optimize_zadoff_chu_params(N, deadline=deadline - 60)
        if v_zc < best_val:
            best_val = v_zc
            best_x = x_zc.copy()
    
    # Phase 2: Generate other candidates
    candidates = []
    
    # Random signs
    for _ in range(5):
        x_rand = rng.choice([-1.0, 1.0], size=N).astype(float)
        candidates.append(x_rand)
    
    # Alternating
    x_alt = np.ones(N)
    x_alt[1::2] = -1.0
    candidates.append(x_alt)
    
    # Block patterns
    for block_size in [10, 20, 50, 100]:
        n_blocks = N // block_size
        x_block = np.tile(rng.choice([-1.0, 1.0], size=n_blocks), N // block_size)
        if len(x_block) < N:
            x_block = np.concatenate([x_block, np.ones(N - len(x_block))])
        candidates.append(x_block[:N])
    
    # Evaluate candidates
    for cand in candidates:
        if time.time() > deadline - 30:
            break
        v = _metric_std(cand)
        if v < best_val:
            best_val = v
            best_x = cand.copy()
    
    if best_x is None:
        return np.ones(N)
    
    # Phase 3: Local search and simulated annealing on best candidates
    # Take top few candidates
    cand_scores = []
    if best_x is not None:
        cand_scores.append((best_x, best_val))
    for cand in candidates:
        if time.time() > deadline - 20:
            break
        v = _metric_std(cand)
        cand_scores.append((cand, v))
    
    cand_scores.sort(key=lambda x: x[1])
    
    for cand, v in cand_scores[:3]:
        if time.time() > deadline - 10:
            break
        
        # Local search
        x_ls, v_ls = _local_search_binary(cand, max_iters=2000, deadline=deadline - 5)
        if v_ls < best_val:
            best_val = v_ls
            best_x = x_ls.copy()
        
        # Simulated annealing
        if time.time() < deadline - 5:
            x_sa, v_sa = _simulated_annealing(cand, max_iters=5000, deadline=deadline - 2)
            if v_sa < best_val:
                best_val = v_sa
                best_x = x_sa.copy()
    
    return best_x
