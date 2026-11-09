import numpy as np
import time

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _score(f):
    """Compute the actual C3' score."""
    ac = _autoconv(f)
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs_ac(f, beta=80.0):
    """
    Smooth approximation of max |ac| using log-sum-exp.
    Returns (value, gradient) using vectorized computation.
    """
    F = np.fft.rfft(f, n=N_FFT)
    ac = np.fft.irfft(F * F, n=N_FFT)
    abs_ac = np.abs(ac)
    max_ac = np.max(abs_ac)
    if max_ac < 1e-30:
        return max_ac, np.zeros_like(f)
    
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    log_sum = np.log(np.sum(exp_shifted)) / beta + max_ac
    
    weights = exp_shifted / np.sum(exp_shifted)
    signs = np.sign(ac)
    
    g = weights * signs
    # Vectorized gradient: grad[i] = 2 * sum_k g[k] * f[k-i] for valid k-i
    # This is a cross-correlation of g with f
    grad = 2.0 * np.correlate(g, f, mode='full')[N-1:2*N-1]
    
    return log_sum, grad

def _objective_and_grad(f, beta=80.0):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f, beta=beta)
    s = np.sum(f)
    
    if abs(s) < 1e-15 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _gradient_optimize(f, deadline, max_iter=3000):
    """
    L-BFGS-B gradient optimization with adaptive beta.
    """
    from scipy.optimize import minimize
    
    best_f = f.copy()
    best_obj = _objective_and_grad(best_f)[0]
    
    # Adaptive beta: start high, decrease if stuck
    betas = [80.0, 60.0, 40.0, 30.0, 20.0, 15.0, 10.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        current = best_f.copy()
        obj_prev = _objective_and_grad(current, beta)[0]
        
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta)[0],
                current,
                jac=lambda x: _objective_and_grad(x, beta)[1],
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            
            candidate = res.x
            obj_new = _objective_and_grad(candidate, beta)[0]
            
            if obj_new < best_obj - 1e-10:
                best_f = candidate.copy()
                best_obj = obj_new
                
        except Exception:
            pass
    
    return best_f

def _greedy_peak_shave(f, max_iters=3000, deadline=None):
    """
    Greedy peak shaving: iteratively flip the sign of the element most
    correlated with the current maximum autoconvolution peak.
    """
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _autoconv(best)
        abs_ac = np.abs(ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = ac[peak_idx]
        
        if peak_idx < N:
            i_start = max(0, peak_idx - (N - 1))
            i_end = min(N - 1, peak_idx)
        else:
            i_start = max(0, peak_idx - N + 1)
            i_end = min(N - 1, peak_idx)
        
        if i_start > i_end:
            break
        
        valid_i = np.arange(i_start, i_end + 1)
        k_vals = peak_idx - valid_i
        
        products = best[valid_i] * best[k_vals]
        same_sign_mask = np.sign(products) == np.sign(peak_val)
        
        if not np.any(same_sign_mask):
            # Try a random flip
            idx = np.random.randint(N)
            candidate = best.copy()
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        # Try top few candidates
        order = np.argsort(np.abs(candidates_prod))[::-1]
        improved = False
        for trial in range(min(8, len(order))):
            fi = candidates_idx[order[trial]]
            candidate = best.copy()
            candidate[fi] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
                improved = True
                break
        
        if not improved:
            # Try random flip as fallback
            if np.random.random() < 0.1:
                idx = np.random.randint(N)
                candidate = best.copy()
                candidate[idx] *= -1
                sc = _score(candidate)
                if sc < best_score:
                    best = candidate
                    best_score = sc
    
    return best, best_score

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips
        n_flips = np.random.randint(1, 4)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        sc = _score(candidate)
        if sc < best_score:
            best = candidate
            best_score = sc
        
        # Try single flip of the most correlated element with peak
        if it % 3 == 0:
            ac = _autoconv(best)
            abs_ac = np.abs(ac)
            peak_idx = np.argmax(abs_ac)
            peak_val = ac[peak_idx]
            
            if peak_idx < N:
                i_start = max(0, peak_idx - (N - 1))
                i_end = min(N - 1, peak_idx)
            else:
                i_start = max(0, peak_idx - N + 1)
                i_end = min(N - 1, peak_idx)
            
            if i_start <= i_end:
                valid_i = np.arange(i_start, i_end + 1)
                k_vals = peak_idx - valid_i
                products = best[valid_i] * best[k_vals]
                same_sign_mask = np.sign(products) == np.sign(peak_val)
                
                if np.any(same_sign_mask):
                    candidates_idx = valid_i[same_sign_mask]
                    candidates_prod = products[same_sign_mask]
                    best_local = np.argmax(np.abs(candidates_prod))
                    flip_idx = candidates_idx[best_local]
                    
                    candidate = best.copy()
                    candidate[flip_idx] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
    
    return best, best_score

def _spectral_init(seed=0):
    """
    Spectral Barker-like initialization: inverse FFT of phase-only mask with flat magnitude.
    """
    rng = np.random.RandomState(seed)
    n_freqs = N_FFT // 2 + 1
    
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    mags = np.ones(n_freqs)
    mags[0] = 0
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    s = np.sum(f)
    if s < 0:
        f = -f
    elif abs(s) < 1e-12:
        f = rng.randn(N)
    
    return f

def _barker_like_init(seed=0):
    """
    Generate Barker-like sequences using known Barker codes and their extensions.
    Barker codes have ideal autocorrelation properties.
    """
    rng = np.random.RandomState(seed)
    
    # Known Barker codes (lengths 2, 3, 4, 5, 7, 11, 13)
    barker_codes = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, 1, -1],
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, 1, -1, 1],
        11: [1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    }
    
    # Try to construct a sequence by repeating/concatenating Barker codes
    candidates = []
    
    # Option 1: Repeat a Barker code
    for code_len in [11, 13, 7]:
        if code_len in barker_codes:
            code = np.array(barker_codes[code_len])
            n_repeats = N // code_len
            remainder = N % code_len
            seq = np.tile(code, n_repeats)
            if remainder > 0:
                seq = np.concatenate([seq, code[:remainder]])
            candidates.append(seq)
    
    # Option 2: Random phase mask with spectral constraints
    n_freqs = N_FFT // 2 + 1
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    mags = np.ones(n_freqs)
    mags[0] = 0
    
    F = mags * np.exp(1j * phases)
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    s = np.sum(f)
    if s < 0:
        f = -f
    candidates.append(f)
    
    # Option 3: Random ±1 sequence
    f = rng.choice([-1.0, 1.0], size=N)
    if np.sum(f) < 0:
        f = -f
    candidates.append(f)
    
    # Pick best candidate
    best_f = None
    best_sc = np.inf
    for c in candidates:
        sc = _score(c)
        if sc < best_sc:
            best_sc = sc
            best_f = c.copy()
    
    return best_f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer)
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Phase 1: Generate diverse initializations
    inits = []
    
    # Barker-like sequences
    for r in range(5):
        if time.time() > deadline - 50.0:
            break
        f0 = _barker_like_init(seed=r * 131 + 7)
        if f0 is not None:
            inits.append(f0)
    
    # Spectral initializations
    for r in range(5):
        if time.time() > deadline - 50.0:
            break
        f0 = _spectral_init(seed=r * 137 + 1)
        inits.append(f0)
    
    # Other diverse starts
    inits.append(np.ones(N))
    inits.append(np.random.randn(N))
    inits.append(np.random.choice([-1.0, 1.0], size=N))
    s = np.sum(inits[-1])
    if s < 0:
        inits[-1] = -inits[-1]
    
    t = np.linspace(0, 10, N)
    inits.append(np.cos(t**2))
    inits.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Evaluate all inits
    for f0 in inits:
        if time.time() > deadline - 45.0:
            break
        sc = _score(f0)
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Phase 2: Gradient optimization from best and other starts
    starts = []
    if best_f is not None:
        starts.append(best_f.copy())
    
    # Add a few more diverse starts
    for f0 in inits[:5]:
        if len(starts) < 4:
            starts.append(f0.copy())
    
    for f0 in starts:
        if time.time() > deadline - 15.0:
            break
        
        polished = _gradient_optimize(f0, deadline - 15.0, max_iter=2000)
        sc = _score(polished)
        if sc < best_score:
            best_f = polished.copy()
            best_score = sc
    
    # Phase 3: Greedy peak shaving polish
    if best_f is not None and time.time() < deadline - 10.0:
        polished, polished_score = _greedy_peak_shave(best_f, max_iters=5000, deadline=deadline - 10.0)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Phase 4: Local search polish
    if best_f is not None and time.time() < deadline - 3.0:
        polished, polished_score = _local_search(best_f, deadline - 3.0, max_iters=30000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Final: try a few more gradient optimizations with remaining time
    if best_f is not None and time.time() < deadline - 5.0:
        polished = _gradient_optimize(best_f, deadline - 5.0, max_iter=1000)
        sc = _score(polished)
        if sc < best_score:
            best_f = polished
            best_score = sc
    
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
