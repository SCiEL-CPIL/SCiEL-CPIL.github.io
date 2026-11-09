import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import remez

N = 400
N_FFT = 2 * N - 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _soft_max_abs_ac(f, beta=50.0):
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

def _objective_and_grad(f, beta=50.0):
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

def _score(f):
    """Compute the actual C3' score."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return np.inf
    ac = _autoconv(f)
    return np.max(np.abs(ac)) / (s * s)

def _greedy_peak_shave(f, max_iters=2000, deadline=None):
    """
    Greedy peak shaving: iteratively flip the sign of the element most
    correlated with the current maximum autoconvolution peak.
    Uses fast incremental score update for speed.
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
        
        # Determine valid indices for this peak
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
            candidate = best.copy()
            idx = np.random.randint(N)
            candidate[idx] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
            continue
        
        candidates_idx = valid_i[same_sign_mask]
        candidates_prod = products[same_sign_mask]
        
        # Try top-3 candidates by |product|
        n_try = min(3, len(candidates_idx))
        order = np.argsort(np.abs(candidates_prod))[::-1][:n_try]
        
        improved = False
        for trial in range(n_try):
            fi = candidates_idx[order[trial]]
            candidate = best.copy()
            candidate[fi] *= -1
            sc = _score(candidate)
            if sc < best_score:
                best = candidate
                best_score = sc
                improved = True
                break
    
    return best, best_score

def _spectral_barker_init(seed=0):
    """
    Barker-like initialization: inverse FFT of phase-only mask with flat magnitude.
    Then apply greedy peak shaving.
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
    
    f, _ = _greedy_peak_shave(f, max_iters=100)
    
    return f

def _remez_init(n_taps, num_ri, num_rl, seed=0, deadline=None):
    """
    Generate equiripple FIR filter coefficients using Parks-McClellan (remez).
    Design various filter types whose impulse responses have mixed signs,
    distributing autoconvolution energy more uniformly across lags.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    rng = np.random.RandomState(seed)
    candidates = []
    
    def _add_candidate(h):
        if len(h) < N:
            h = np.pad(h, (0, N - len(h)))
        else:
            h = h[:N]
        s = np.sum(h)
        if abs(s) < 1e-12:
            h = h + rng.randn(N) * 0.01
            s = np.sum(h)
        if s < 0:
            h = -h
        candidates.append(h)
    
    n_taps_odd = N if N % 2 == 1 else N - 1
    
    # Type 1: Bandstop filters with varied parameters
    for i in range(3):
        try:
            f1 = 0.1 + 0.08 * (seed % 4) + 0.02 * i
            f2 = f1 + 0.08 + 0.04 * (seed % 3)
            if f2 > 0.45:
                continue
            bands = [0.0, f1, f2, 0.5]
            desired = [1.0, 0.0, 1.0]
            weights = [1.0, 1.0, 1.0]
            h = remez(n_taps_odd - 1, bands, desired, weights)
            _add_candidate(h)
        except Exception:
            pass
    
    # Type 2: Lowpass with varied cutoffs
    for i in range(3):
        try:
            cutoff = 0.15 + 0.05 * (seed % 6) + 0.02 * i
            if cutoff > 0.45:
                continue
            h = remez(n_taps_odd - 1, [0.0, cutoff], [1.0, 0.0])
            _add_candidate(h)
        except Exception:
            pass
    
    # Type 3: Highpass with varied cutoffs
    for i in range(2):
        try:
            cutoff = 0.2 + 0.05 * (seed % 4) + 0.02 * i
            if cutoff > 0.45:
                continue
            h = remez(n_taps_odd - 1, [0.0, cutoff], [0.0, 1.0])
            _add_candidate(h)
        except Exception:
            pass
    
    # Type 4: Bandpass filters
    for i in range(2):
        try:
            f1 = 0.1 + 0.05 * (seed % 5)
            f2 = f1 + 0.1 + 0.03 * (seed % 3)
            if f2 > 0.45:
                continue
            bands = [0.0, f1, f2, 0.5]
            desired = [0.0, 1.0, 0.0]
            weights = [1.0, 1.0, 1.0]
            h = remez(n_taps_odd - 1, bands, desired, weights)
            _add_candidate(h)
        except Exception:
            pass
    
    if not candidates:
        return None
    
    # Pick the candidate with best initial score
    best_h = None
    best_sc = np.inf
    for h in candidates:
        sc = _score(h)
        if sc < best_sc:
            best_sc = sc
            best_h = h
    
    return best_h

def _local_search(f, deadline, max_iters=50000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = _score(best)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips (1-3 flips)
        n_flips = np.random.randint(1, 4)
        indices = np.random.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        sc = _score(candidate)
        if sc < best_score:
            best = candidate
            best_score = sc
        
        # Try targeted peak-reduction flips
        if it % 2 == 0:
            ac = _autoconv(best)
            abs_ac = np.abs(ac)
            
            # Try top-3 peaks
            top_peaks = np.argsort(abs_ac)[::-1][:3]
            
            for peak_idx in top_peaks:
                peak_val = ac[peak_idx]
                
                if peak_idx < N:
                    i_start = max(0, peak_idx - (N - 1))
                    i_end = min(N - 1, peak_idx)
                else:
                    i_start = max(0, peak_idx - N + 1)
                    i_end = min(N - 1, peak_idx)
                
                if i_start > i_end:
                    continue
                
                valid_i = np.arange(i_start, i_end + 1)
                k_vals = peak_idx - valid_i
                products = best[valid_i] * best[k_vals]
                same_sign_mask = np.sign(products) == np.sign(peak_val)
                
                if not np.any(same_sign_mask):
                    continue
                
                candidates_idx = valid_i[same_sign_mask]
                candidates_prod = products[same_sign_mask]
                
                # Try top-3 candidates by |product|
                n_try = min(3, len(candidates_idx))
                order = np.argsort(np.abs(candidates_prod))[::-1][:n_try]
                
                for trial in range(n_try):
                    flip_idx = candidates_idx[order[trial]]
                    candidate = best.copy()
                    candidate[flip_idx] *= -1
                    sc = _score(candidate)
                    if sc < best_score:
                        best = candidate
                        best_score = sc
                        break
                else:
                    continue
                break
    
    return best, best_score

def _gradient_optimize(f0, deadline, maxiter=600, beta_schedule=None):
    """
    Run L-BFGS-B with adaptive beta schedule for soft-max.
    Starts with lower beta (smoother) and increases for sharper max approximation.
    """
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    current = f0.copy()
    best_local = current.copy()
    best_local_score = _score(current)
    
    for beta in beta_schedule:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _objective_and_grad(x, beta=beta)[0]
        
        def jac(x):
            return _objective_and_grad(x, beta=beta)[1]
        
        try:
            res = minimize(
                obj,
                current,
                jac=jac,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 40}
            )
            
            candidate = res.x
            sc = _score(candidate)
            if sc < best_local_score:
                best_local = candidate.copy()
                best_local_score = sc
            current = candidate.copy()
            
        except Exception:
            pass
    
    return best_local, best_local_score

def construct():
    start_time = time.time()
    deadline = start_time + 147.0  # 3s safety margin
    
    best_f = None
    best_score = np.inf
    
    np.random.seed(42)
    
    # Phase 1: Parks-McClellan (remez) equiripple initializations
    n_remez = 16
    for r in range(n_remez):
        if time.time() > deadline - 50.0:
            break
        f0 = _remez_init(N, 1, 1, seed=r * 131 + 7, deadline=deadline - 50.0)
        if f0 is not None:
            sc = _score(f0)
            if sc < best_score:
                best_f = f0.copy()
                best_score = sc
    
    # Phase 1b: Spectral Barker initializations with greedy shaving
    n_spectral = 10
    for r in range(n_spectral):
        if time.time() > deadline - 45.0:
            break
        f0 = _spectral_barker_init(seed=r * 137 + 1)
        sc = _score(f0)
        if sc < best_score:
            best_f = f0.copy()
            best_score = sc
    
    # Phase 2: Gradient optimization from best and other starts
    starts = []
    if best_f is not None:
        starts.append(best_f.copy())
    
    # Add diverse starts
    starts.append(np.ones(N))
    starts.append(np.random.randn(N))
    starts.append(np.random.choice([-1.0, 1.0], size=N))
    s = np.sum(starts[2])
    if s < 0:
        starts[2] = -starts[2]
    
    t = np.linspace(0, 10, N)
    starts.append(np.cos(t**2))
    
    starts.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # Chirp signal
    t2 = np.linspace(0, 5, N)
    starts.append(np.sin(2 * np.pi * t2**2))
    
    # Add a remez-based start if available
    f_remez_extra = _remez_init(N, 1, 1, seed=999, deadline=deadline - 35.0)
    if f_remez_extra is not None:
        starts.append(f_remez_extra)
    
    # Run gradient optimization on each start
    for f0 in starts:
        if time.time() > deadline - 30.0:
            break
        
        candidate, sc = _gradient_optimize(f0, deadline - 30.0, maxiter=500)
        if sc < best_score:
            best_f = candidate.copy()
            best_score = sc
    
    # Phase 3: Greedy peak shaving polish
    if best_f is not None and time.time() < deadline - 20.0:
        polished, polished_score = _greedy_peak_shave(best_f, max_iters=10000, deadline=deadline - 20.0)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Phase 4: Local search polish
    if best_f is not None and time.time() < deadline - 8.0:
        polished, polished_score = _local_search(best_f, deadline - 8.0, max_iters=60000)
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Phase 5: Final high-precision gradient polish with high beta
    if best_f is not None and time.time() < deadline - 3.0:
        polished, polished_score = _gradient_optimize(
            best_f, deadline - 3.0, maxiter=300,
            beta_schedule=[200.0, 500.0, 1000.0]
        )
        if polished_score < best_score:
            best_f = polished
            best_score = polished_score
    
    # Final: try a few more remez inits with remaining time
    if time.time() < deadline - 4.0:
        for r in range(4):
            if time.time() > deadline - 4.0:
                break
            f0 = _remez_init(N, 1, 1, seed=2000 + r * 89, deadline=deadline - 4.0)
            if f0 is not None:
                sc = _score(f0)
                if sc < best_score:
                    best_f = f0.copy()
                    best_score = sc
                    # Quick polish
                    if time.time() < deadline - 2.0:
                        polished, polished_score = _greedy_peak_shave(best_f, max_iters=1500, deadline=deadline - 2.0)
                        if polished_score < best_score:
                            best_f = polished
                            best_score = polished_score
    
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
