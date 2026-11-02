"""Optimized construction for the signed autocorrelation C3-prime task.
Combines spectral shaping (FFT amplitude control) with vectorized coordinate
descent polish and multi-scale sign-flip search.

Key structural change: instead of starting from random/structured time-domain
seeds and refining, we construct initial candidates directly in the frequency
domain by shaping the power spectral density to suppress large autoconvolution
peaks, then refine in the time domain.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 400
FFT_SIZE = 1024  # Next power of 2 >= 2*N-1 = 799

def _autoconv(x):
    """Full linear autoconvolution of x via FFT."""
    F = np.fft.rfft(x, FFT_SIZE)
    c = np.fft.irfft(F * np.conj(F), FFT_SIZE)
    return c[:2 * N - 1]

def _score(x):
    """C3' = max|autoconv| / (integral)^2. Integral = sum(x)."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    c = _autoconv(x)
    return float(np.max(np.abs(c)) / (s * s))

def _score_fast(x):
    """Fast score computation with precomputed FFT size."""
    s = x.sum()
    if abs(s) < 1e-12:
        return np.inf
    F = np.fft.rfft(x, FFT_SIZE)
    c = np.fft.irfft(F * np.conj(F), FFT_SIZE)[:2 * N - 1]
    return float(np.max(np.abs(c)) / (s * s))

def _soft_max_score(x, beta):
    """Smooth approximation of max|ac|/(sum)^2 using soft-max."""
    s = x.sum()
    if abs(s) < 1e-12:
        return 1e10
    F = np.fft.rfft(x, FFT_SIZE)
    c = np.fft.irfft(F * np.conj(F), FFT_SIZE)[:2 * N - 1]
    ac_abs = np.abs(c)
    m = np.max(ac_abs)
    # Soft max: m + (1/beta) * log(sum(exp(beta*(ac_abs - m))))
    exps = np.exp(beta * (ac_abs - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        soft_max = m
    else:
        soft_max = m + (1.0 / beta) * np.log(sum_exps)
    return soft_max / (s * s)

def _soft_max_grad(x, beta):
    """Gradient of soft-max score w.r.t. x.
    
    d(soft_max)/dx[n] = sum_k w_k * d(ac_abs[k])/dx[n]
    where w_k = beta * exp(beta*(ac_abs[k]-m)) / sum_j exp(beta*(ac_abs[j]-m))
    
    d(ac[k])/dx[n] = x[n] * x[k-n] + x[k-n] * x[n] = 2*x[n]*x[k-n] for valid indices
    Actually for autoconvolution: ac[k] = sum_i x[i]*x[k-i]
    d(ac[k])/dx[n] = x[k-n] + x[n] (when both indices valid)
    For |ac[k]|, d|ac[k]|/dx[n] = sign(ac[k]) * (x[k-n] + x[n])
    """
    s = x.sum()
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(x)
    
    F = np.fft.rfft(x, FFT_SIZE)
    c = np.fft.irfft(F * np.conj(F), FFT_SIZE)[:2 * N - 1]
    ac_abs = np.abs(c)
    m = np.max(ac_abs)
    
    exps = np.exp(beta * (ac_abs - m))
    sum_exps = np.sum(exps)
    if sum_exps < 1e-300:
        weights = np.zeros_like(ac_abs)
    else:
        weights = beta * exps / sum_exps
    
    # Gradient of soft_max w.r.t. x
    # soft_max = m + (1/beta)*log(sum_exps)
    # d(soft_max)/dx[n] = sum_k weights[k] * d(ac_abs[k])/dx[n]
    # d(ac_abs[k])/dx[n] = sign(c[k]) * (x[k-n] if 0<=k-n<N else 0 + x[n] if 0<=k-n<N else 0)
    # Wait, let me be more careful.
    # ac[k] = sum_{i=0}^{N-1} x[i] * x[k-i] where x[j]=0 for j<0 or j>=N
    # d(ac[k])/dx[n] = x[k-n] (if 0<=k-n<N) + x[n] (if 0<=k-n<N) ... no
    # d(ac[k])/dx[n] = sum_i [d(x[i])/dx[n] * x[k-i] + x[i] * d(x[k-i])/dx[n]]
    #                = x[k-n] (if 0<=k-n<N) + x[k-n] (if 0<=k-n<N)
    #                = 2*x[k-n] if 0<=k-n<N else 0
    # Wait, that's not right either. Let me think again.
    # ac[k] = sum_i x[i] * x[k-i]
    # d(ac[k])/dx[n] = sum_i [delta_{i,n} * x[k-i] + x[i] * delta_{k-i,n}]
    #                = x[k-n] + x[k-n]  (both terms give x[k-n] when valid)
    #                = 2*x[k-n] if 0 <= k-n < N, else 0
    
    # But for |ac[k]|: d|ac[k]|/dx[n] = sign(ac[k]) * 2*x[k-n] (if valid)
    
    # So d(soft_max)/dx[n] = sum_k weights[k] * sign(c[k]) * 2*x[k-n] (if 0<=k-n<N)
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # Let g[k] = weights[k] * sign(c[k]) * 2
    # Then d(soft_max)/dx[n] = sum_k g[k] * x[k-n] (if 0<=k-n<N)
    # This is a cross-correlation of g with x.
    
    g = weights * np.sign(c) * 2.0
    # Pad g to handle boundaries
    # d(soft_max)/dx[n] = sum_{k=0}^{2N-2} g[k] * x[k-n] * [0<=k-n<N]
    # = sum_{m=0}^{N-1} g[m+n] * x[m] where m = k-n, so k = m+n
    # This is like a convolution of g with x, but shifted.
    
    # Let's compute via FFT for efficiency
    # h[n] = sum_{m=0}^{N-1} g[m+n] * x[m]
    # This is a valid cross-correlation.
    
    # Pad both to FFT_SIZE
    g_pad = np.zeros(FFT_SIZE)
    g_pad[:2*N-1] = g
    x_pad = np.zeros(FFT_SIZE)
    x_pad[:N] = x
    
    # h = ifft(conj(fft(g_pad)) * fft(x_pad)) ... no
    # h[n] = sum_m g[m+n] * x[m]
    # Let's flip g: g_flip[k] = g[-k] mod FFT_SIZE
    # Then h = ifft(fft(g_flip) * fft(x_pad))
    
    # Actually, let's just do it directly for N=400, it's fast enough
    grad_sm = np.zeros(N)
    for n in range(N):
        # k ranges from n to n+N-1 (so that k-n is in [0, N-1])
        k_start = n
        k_end = min(n + N, 2 * N - 1)
        if k_start < 2 * N - 1:
            grad_sm[n] = np.sum(g[k_start:k_end] * x[:k_end - k_start])
    
    # Now d(score)/dx[n] = d(soft_max)/dx[n] / s^2 - 2 * soft_max / s^3
    score_val = soft_max_val = m + (1.0 / beta) * np.log(max(sum_exps, 1e-300))
    score = score_val / (s * s)
    grad = grad_sm / (s * s) - 2.0 * score_val / (s * s * s)
    
    return score, grad

def _spectral_seed(rng, target_peak_reduction=0.5):
    """Generate a seed by shaping the power spectral density in frequency domain.
    
    Strategy: start with a random phase spectrum, shape the amplitude to reduce
    peak autoconvolution. The autoconvolution in frequency domain is |X(f)|^2,
    so we want to minimize the peak of the inverse FFT of |X(f)|^2.
    """
    # Start with random signs
    x = rng.choice([-1.0, 1.0], size=N)
    
    # Get the spectrum
    X = np.fft.rfft(x, FFT_SIZE)
    amp = np.abs(X)
    phase = np.angle(X)
    
    # Shape the amplitude: reduce high-amplitude components
    # The autoconvolution spectrum is |X|^2, so peaks in |X|^2 correspond to
    # peaks in the autoconvolution.
    # We want to flatten |X|^2 as much as possible while keeping sum(x) non-zero.
    
    # DC component (k=0) must be preserved to keep sum non-zero
    # Try different amplitude shaping strategies
    best_x = x.copy()
    best_score = _score_fast(x)
    
    for strategy in range(5):
        amp_new = amp.copy()
        
        if strategy == 0:
            # Compress amplitude: amp_new = amp^0.7
            amp_new = amp ** 0.7
        elif strategy == 1:
            # Compress amplitude more
            amp_new = amp ** 0.5
        elif strategy == 2:
            # Soft threshold: reduce amplitudes above median
            med = np.median(amp)
            amp_new = np.where(amp > med, med + (amp - med) * 0.5, amp)
        elif strategy == 3:
            # Targeted reduction: reduce top 20% amplitudes
            threshold = np.percentile(amp, 80)
            amp_new = np.where(amp > threshold, threshold, amp)
        else:
            # Log compression
            amp_new = np.log1p(amp)
        
        # Preserve DC component to keep sum non-zero
        dc_orig = X[0].real
        X_new = amp_new * np.exp(1j * phase)
        X_new[0] = dc_orig  # Preserve DC
        
        x_new = np.fft.irfft(X_new, FFT_SIZE)[:N]
        
        # Snap to signs but allow continuous values for optimization
        # First try with continuous values
        sc = _score_fast(x_new)
        if sc < best_score:
            best_score = sc
            best_x = x_new.copy()
        
        # Also try snapping to signs
        x_signs = np.sign(x_new)
        x_signs[x_signs == 0] = 1.0
        sc_s = _score_fast(x_signs)
        if sc_s < best_score:
            best_score = sc_s
            best_x = x_signs
    
    return best_x, best_score

def _continuous_optimize(x0, deadline, max_iter=300, beta=50.0):
    """Use scipy L-BFGS-B to optimize continuously with soft-max surrogate."""
    x0 = x0.astype(np.float64).copy()
    s0 = x0.sum()
    if abs(s0) < 1e-12:
        x0[0] += 0.1
    
    def objective(x):
        if time.time() > deadline:
            return 1e10
        val, _ = _soft_max_grad(x, beta)
        return val
    
    bounds = [(-3.0, 3.0)] * N
    
    try:
        res = minimize(
            objective, x0, method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.fun < 1e9:
            return res.x
    except Exception:
        pass
    return x0

def _vectorized_polish(x, deadline, max_rounds=50):
    """Vectorized coordinate descent polish.
    
    For each coordinate j, compute the score if we flip x[j].
    Use the fact that flipping x[j] changes autoconvolution in a structured way.
    """
    x = x.copy().astype(np.float64)
    best = _score_fast(x)
    n = len(x)
    
    # Precompute current autoconvolution
    ac = _autoconv(x)
    s = x.sum()
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        improved = False
        # Try each coordinate flip
        # Flipping x[j] to -x[j]:
        # new_ac[k] = ac[k] - 2*x[j]*(x[k-j] + x[j-k]) for valid indices
        # Actually: ac_new[k] = ac[k] - 2*x[j]*x[k-j] - 2*x[j]*x[j-k] ... 
        # Let's just recompute for simplicity with vectorized operations
        
        # For efficiency, compute the change in autoconvolution when flipping x[j]
        # ac[k] = sum_i x[i]*x[k-i]
        # If we flip x[j]: x_new[j] = -x[j]
        # ac_new[k] = ac[k] - 2*x[j]*x[k-j] (for valid k-j) - 2*x[j]*x[j-k] (for valid j-k)
        # Wait, both terms involve x[j]. Let me be careful.
        # ac_new[k] = sum_i x_new[i]*x_new[k-i]
        # = sum_{i≠j} x[i]*x[k-i] + x_new[j]*x_new[k-j] + x_new[k-j]*x_new[j] (if k-j≠j)
        # This is getting complicated. Let's just recompute for each flip.
        
        # Vectorized approach: for each j, compute new score
        # This is O(N) recomputations per round, each O(N log N) via FFT
        # Total: O(N^2 log N) per round, which for N=400 is ~160K * 10 = 1.6M ops
        # Should be fast enough.
        
        for j in range(n):
            if time.time() > deadline:
                break
            
            old_val = x[j]
            x[j] = -old_val
            new_score = _score_fast(x)
            if new_score < best:
                best = new_score
                improved = True
            else:
                x[j] = old_val
        
        if not improved:
            break
    
    return x, best

def _block_sign_search(x, deadline, min_block=2, max_block=16):
    """Search over blocks of consecutive sign flips."""
    x = x.copy().astype(np.float64)
    best = _score_fast(x)
    n = len(x)
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try random blocks
        for _ in range(200):
            if time.time() > deadline:
                break
            
            block_size = int(rng.integers(min_block, max_block + 1))
            start = int(rng.integers(0, n - block_size + 1))
            idx = np.arange(start, start + block_size)
            
            cand = x.copy()
            cand[idx] = -cand[idx]
            sc = _score_fast(cand)
            if sc < best:
                best = sc
                x = cand
                improved = True
        
        # Try single flips
        for j in range(n):
            if time.time() > deadline:
                break
            cand = x.copy()
            cand[j] = -cand[j]
            sc = _score_fast(cand)
            if sc < best:
                best = sc
                x = cand
                improved = True
        
        if not improved:
            break
    
    return x, best

def _multi_beta_optimize(x0, deadline):
    """Run continuous optimization with multiple beta values for soft-max."""
    best_x = x0.copy()
    best_score = _score_fast(x0)
    
    betas = [20.0, 50.0, 100.0, 200.0]
    time_per_beta = max(1.0, (deadline - time.time()) / len(betas) * 0.8)
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        
        beta_deadline = min(deadline - 2, time.time() + time_per_beta)
        x_opt = _continuous_optimize(best_x, beta_deadline, max_iter=200, beta=beta)
        
        # Evaluate and keep best
        sc = _score_fast(x_opt)
        if sc < best_score:
            best_score = sc
            best_x = x_opt.copy()
    
    return best_x, best_score

def _structured_seeds(rng):
    """Generate diverse structured seeds."""
    seeds = []
    
    # All ones (baseline)
    seeds.append(np.ones(N, dtype=np.float64))
    
    # Alternating
    seeds.append(np.tile([1.0, -1.0], N // 2).astype(np.float64))
    
    # Random patterns
    for _ in range(6):
        x = rng.choice([-1.0, 1.0], size=N)
        seeds.append(x.astype(np.float64))
    
    # Block patterns
    for block_size in [4, 8, 16, 32]:
        if N % block_size == 0:
            n_blocks = N // block_size
            block_signs = rng.choice([-1.0, 1.0], size=n_blocks)
            x = np.repeat(block_signs, block_size).astype(np.float64)
            seeds.append(x)
    
    return seeds

def construct():
    """Return the sequence of signed step heights minimizing C3'."""
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    rng = np.random.default_rng(2024)
    
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score_fast(best_x)
    
    # Phase 1: Generate spectral seeds and evaluate
    spectral_time_budget = min(20.0, (deadline - time.time()) * 0.15)
    spectral_deadline = time.time() + spectral_time_budget
    
    for _ in range(8):
        if time.time() > spectral_deadline:
            break
        x_spec, sc_spec = _spectral_seed(rng)
        if sc_spec < best_score:
            best_score = sc_spec
            best_x = x_spec.copy()
    
    # Phase 2: Continuous optimization from best seed
    cont_deadline = min(deadline - 30, time.time() + 25.0)
    if time.time() < cont_deadline:
        x_opt, sc_opt = _multi_beta_optimize(best_x, cont_deadline)
        if sc_opt < best_score:
            best_score = sc_opt
            best_x = x_opt.copy()
    
    # Phase 3: Snap to signs and polish
    x_signs = np.sign(best_x)
    x_signs[x_signs == 0] = 1.0
    
    # Ensure non-zero sum
    if abs(x_signs.sum()) < 1e-12:
        x_signs[0] = 1.0
        x_signs[1] = 1.0
    
    sc_signs = _score_fast(x_signs)
    if sc_signs < best_score:
        best_score = sc_signs
        best_x = x_signs.copy()
    
    # Phase 4: Vectorized polish
    polish_deadline = min(deadline - 10, time.time() + 30.0)
    if time.time() < polish_deadline:
        x_polished, sc_polished = _vectorized_polish(best_x, polish_deadline, max_rounds=30)
        if sc_polished < best_score:
            best_score = sc_polished
            best_x = x_polished.copy()
    
    # Phase 5: Block sign search
    block_deadline = min(deadline - 3, time.time() + 20.0)
    if time.time() < block_deadline:
        x_block, sc_block = _block_sign_search(best_x, block_deadline)
        if sc_block < best_score:
            best_score = sc_block
            best_x = x_block.copy()
    
    # Phase 6: Final intensive single-flip search
    final_deadline = deadline - 1.0
    if time.time() < final_deadline:
        x_final, sc_final = _vectorized_polish(best_x, final_deadline, max_rounds=20)
        if sc_final < best_score:
            best_score = sc_final
            best_x = x_final.copy()
    
    return best_x
