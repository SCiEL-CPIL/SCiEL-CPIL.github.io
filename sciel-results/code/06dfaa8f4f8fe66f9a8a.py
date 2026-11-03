import numpy as np
import time
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_fft_size(n):
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _soft_max_abs(ac, beta=200.0):
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=200.0):
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _generate_candidate(rng, strategy=0):
    h = np.zeros(N)
    
    if strategy == 0:
        h = rng.standard_normal(N)
    elif strategy == 1:
        h = rng.choice([1.0, -1.0], N)
    elif strategy == 2:
        mask = rng.random(N) < 0.2
        h[mask] = rng.choice([1.0, -1.0], mask.sum())
    elif strategy == 3:
        num_blocks = 10
        block_size = N // num_blocks
        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size if i < num_blocks - 1 else N
            sign = rng.choice([1.0, -1.0])
            h[start:end] = sign
    elif strategy == 4:
        noise = rng.standard_normal(N)
        H = np.fft.rfft(noise)
        cutoff = 10
        H[cutoff:] = 0
        h = np.fft.irfft(H, N)
        if np.sum(np.abs(h)) < 1e-6:
             h = rng.standard_normal(N)
    elif strategy == 5:
        x = np.arange(N)
        env = np.cos(np.pi * x / N) ** 2
        signs = rng.choice([1.0, -1.0], N)
        h = env * signs
    elif strategy == 6:
        # Chirp-like structure
        x = np.arange(N)
        freq = rng.uniform(1.0, 5.0)
        h = np.sin(2 * np.pi * freq * x / N + rng.uniform(0, 2*np.pi))
        # Add random noise
        h += 0.1 * rng.standard_normal(N)
    elif strategy == 7:
        # Low-pass filtered random signs with smooth envelope
        signs = rng.choice([1.0, -1.0], N)
        x = np.arange(N)
        env = np.exp(-((x - N/2)**2) / (2 * (N/4)**2))
        h = signs * env
        # Smooth it a bit
        kernel = np.ones(5) / 5
        h = np.convolve(h, kernel, mode='same')
    elif strategy == 8:
        # Random walk with mean zero
        steps = rng.standard_normal(N)
        h = np.cumsum(steps)
        h -= np.mean(h)
    else:
        h = rng.standard_normal(N)
        
    return h

def _gradient_descent(h_init, deadline, rng, lr=0.01, momentum=0.9, beta=200.0):
    """
    Gradient descent with momentum on the smooth objective.
    """
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = rng.standard_normal(N)
        s = np.sum(h)
    h = h / s
    
    current_score = _objective_smooth(h, beta)
    best_score = current_score
    best_h = h.copy()
    
    v = np.zeros(N)
    t = 0
    
    while time.time() < deadline:
        t += 1
        # Compute gradient numerically or analytically?
        # Numerical is safer but slower. Let's try a small step size for numerical grad.
        # Or just use the fact that we can compute the gradient of the soft max.
        # For now, let's use a simple stochastic gradient or just perturb.
        # Actually, let's stick to a robust local search since gradient of max is tricky.
        
        # Let's do a hybrid: small random perturbations + acceptance based on exact score
        # This is essentially simulated annealing with very small steps.
        
        # Perturb a few elements
        n_perturb = rng.integers(1, 10)
        indices = rng.choice(N, n_perturb, replace=False)
        
        orig_vals = h[indices].copy()
        h[indices] += rng.standard_normal(n_perturb) * 0.1
        
        new_score = _objective_smooth(h, beta)
        
        if new_score < current_score:
            current_score = new_score
            v = momentum * v + (1-momentum) * (h[indices] - orig_vals)
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            # Revert
            h[indices] = orig_vals
            v = momentum * v
            
        # Occasionally normalize to prevent drift
        if t % 100 == 0:
            s = np.sum(h)
            if abs(s) > 1e-9:
                h = h / s
                current_score = _objective_smooth(h, beta)
                if current_score < best_score:
                    best_score = current_score
                    best_h = h.copy()
                    
    # Final exact score evaluation
    exact_score = _score(best_h)
    if exact_score < best_score:
        best_score = exact_score
        
    return best_h, best_score

def _simulated_annealing(h_init, deadline, rng, T0=1.0, cooling=0.999, min_T=1e-6):
    """
    Simulated Annealing to refine a candidate.
    Uses multi-bit flips and exact score evaluation.
    """
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = rng.standard_normal(N)
        s = np.sum(h)
    h = h / s
    
    current_score = _score(h)
    best_score = current_score
    best_h = h.copy()
    
    T = T0
    i = 0
    while time.time() < deadline and T > min_T:
        i += 1
        n_flips = rng.integers(1, 6)
        indices = rng.choice(N, n_flips, replace=False)
        
        orig_vals = h[indices].copy()
        h[indices] = -h[indices]
        
        new_score = _score(h)
        
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h.copy()
        else:
            if T > 0:
                delta = new_score - current_score
                if delta / T < 700:
                    p = np.exp(-delta / T)
                    if rng.random() < p:
                        current_score = new_score
                    else:
                        h[indices] = orig_vals
                else:
                    h[indices] = orig_vals
            else:
                h[indices] = orig_vals
        
        T *= cooling
        
    return best_h, best_score

def _bfgs_optimize(h_init, deadline, rng):
    """
    Use BFGS with a smooth objective to find a local minimum.
    """
    h = h_init.copy()
    s = np.sum(h)
    if abs(s) < 1e-9:
        h = rng.standard_normal(N)
        s = np.sum(h)
    h = h / s
    
    def obj(x):
        return _objective_smooth(x, beta=200.0)
        
    res = minimize(obj, h, method='BFGS', options={'maxiter': 50, 'gtol': 1e-5})
    
    if res.success or res.fun < _score(h):
        h_opt = res.x
        s_opt = np.sum(h_opt)
        if abs(s_opt) > 1e-9:
            h_opt = h_opt / s_opt
        score_opt = _score(h_opt)
        return h_opt, score_opt
    else:
        return h, _score(h)

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin for 150s limit
    
    best_score = np.inf
    best_h = np.ones(N)
    
    rng = np.random.default_rng(42)
    
    strategies = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    # Generate initial candidates
    num_initial = 30
    candidates = []
    for i in range(num_initial):
        if time.time() > deadline - 30:
            break
        strat = strategies[i % len(strategies)]
        h = _generate_candidate(rng, strat)
        s = np.sum(h)
        if abs(s) < 1e-9:
            continue
        h = h / s
        val = _score(h)
        candidates.append((val, h))
        
    candidates.sort(key=lambda x: x[0])
    
    # Run different optimization methods on top candidates
    # Allocate time based on rank
    for i, (init_val, h_init) in enumerate(candidates[:10]):
        if time.time() > deadline - 10:
            break
        
        remaining = deadline - time.time()
        
        # Choose method based on index
        if i % 3 == 0:
            # BFGS
            local_deadline = min(deadline - 5.0, time.time() + 10.0)
            h_opt, score_opt = _bfgs_optimize(h_init, local_deadline, rng)
        elif i % 3 == 1:
            # Gradient descent
            local_deadline = min(deadline - 5.0, time.time() + 10.0)
            h_opt, score_opt = _gradient_descent(h_init, local_deadline, rng)
        else:
            # Simulated annealing
            local_deadline = min(deadline - 5.0, time.time() + 10.0)
            h_opt, score_opt = _simulated_annealing(h_init, local_deadline, rng)
            
        if score_opt < best_score:
            best_score = score_opt
            best_h = h_opt
    
    # Final polish on the best found with SA
    if time.time() < deadline - 5.0 and best_score < np.inf:
        final_deadline = deadline
        h_polished, score_polished = _simulated_annealing(best_h, final_deadline, rng, T0=0.1, cooling=0.9999)
        
        if score_polished < best_score:
            best_score = score_polished
            best_h = h_polished
            
    # Ensure valid return
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
