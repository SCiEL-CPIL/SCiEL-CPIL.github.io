import numpy as np
import time
from scipy.optimize import minimize

N = 400
DEADLINE = 145.0  # safety margin before 150s kill


def _value(h):
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))


def _smooth_objective_and_grad(h, beta=50.0):
    """Smooth surrogate using log-sum-exp for max in autocorrelation, with gradient."""
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e18, np.zeros(N)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -700, 700)
    weights = np.exp(shifted)
    w_sum = np.sum(weights)
    weights /= w_sum
    soft_max = m + np.log(w_sum) / beta
    
    val = 2.0 * N * soft_max / (s * s)
    obj = abs(val)
    
    # Gradient computation
    # d(conv[k])/d(h[j]) = h[k-j] + h[k-j] (symmetric) = 2*h[k-j] if 0 <= k-j < N
    # d(soft_max)/d(h[j]) = sum_k weights[k] * d(conv[k])/d(h[j])
    # = sum_k weights[k] * 2 * h[k-j] for valid k-j in [0, N-1]
    # This is a correlation: d_soft_max[j] = 2 * sum_k weights[k] * h[k-j]
    # Using np.correlate for efficiency
    
    # d_soft_max[j] = 2 * (weights * h)[j] where * is correlation
    # np.correlate(weights, h, mode='full') gives correlation
    # But we need: for each j, sum over k of weights[k] * h[k-j]
    # This is cross-correlation of weights with h
    
    # Actually, let's think more carefully:
    # conv[k] = sum_i h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] (but since conv is symmetric, and we're taking derivative)
    # Wait, conv = np.convolve(h, h) where h has length N
    # conv[k] = sum_i h[i] * h[k-i] for valid i
    # d(conv[k])/d(h[j]) = h[k-j] if 0 <= k-j < N, plus h[j-k] if 0 <= j-k < N
    # But since conv is symmetric (conv[k] = conv[2N-2-k]), and h is real:
    # Actually np.convolve(h,h) gives: conv[k] = sum_{i} h[i]*h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] (from the first h) + h[k-j] (from the second h, by symmetry)
    # Wait no. Let's be precise:
    # conv[k] = sum_i h[i] * h[k-i]
    # d(conv[k])/d(h[j]) = h[k-j] + h[j-k] ... no
    # d(conv[k])/d(h[j]) = sum_i [delta(i,j)*h[k-i] + h[i]*delta(k-i,j)]
    # = h[k-j] + h[k-j] = 2*h[k-j] if 0 <= k-j < N and 0 <= k-j < N
    # But we need 0 <= i < N and 0 <= k-i < N for the term to exist
    # So d(conv[k])/d(h[j]) = 2*h[k-j] if 0 <= k-j < N, else 0
    
    # d(soft_max)/d(h[j]) = sum_k weights[k] * 2*h[k-j] for 0 <= k-j < N
    # = 2 * sum_{k=j}^{j+N-1} weights[k] * h[k-j]
    # = 2 * (weights ⊛ h)[j] where ⊛ is cross-correlation
    
    # np.correlate(a, v) computes sum_k a[k] * v[k-j] for various j
    # We want sum_k weights[k] * h[k-j]
    # This is np.correlate(weights, h, mode='full')[N-1+N-1-j]... let me think
    # Actually, let's use FFT for speed
    
    n_fft = 2 * N
    H_fft = np.fft.rfft(h, n=n_fft)
    W_fft = np.fft.rfft(weights, n=n_fft)
    
    # We want d_soft_max[j] = 2 * sum_k weights[k] * h[k-j]
    # In FFT terms, if we define the correlation as:
    # corr[j] = sum_k weights[k] * h[k-j]
    # This is the cross-correlation. Using FFT:
    # corr = ifft(conj(fft(weights)) * fft(h)) ... let me verify
    # Actually: corr[j] = sum_k weights[k] * h[k-j]
    # Let's use: ifft(fft(weights) * conj(fft(h))) gives sum_k weights[k] * h[k-j]? 
    # No. The standard result: ifft(fft(a) * conj(fft(b))) gives the cross-correlation
    # where (a ⊛ b)[j] = sum_k a[k] * b[k-j]
    
    corr = np.fft.irfft(H_fft * np.conj(W_fft), n=n_fft)
    # corr[j] = sum_k h[k] * weights[k-j] for the linear case (with zero-padding)
    # But we want sum_k weights[k] * h[k-j]
    # By symmetry of correlation for real signals: sum_k weights[k]*h[k-j] = sum_k h[k]*weights[k-j]
    # So corr[j] from above should work, but let's verify the indexing
    
    # Actually, np.fft.irfft(H_fft * np.conj(W_fft)) computes:
    # c[j] = sum_k H[k] * conj(W[k]) in frequency -> time domain
    # This is the circular correlation of h and weights
    # With zero-padding to 2N, this gives linear correlation
    # c[j] = sum_k h[k] * weights[(k-j) mod 2N]
    # For linear correlation with zero-padding, this equals sum_k h[k] * weights[k-j] 
    # for the valid range
    
    # We want d_soft_max[j] = 2 * sum_k weights[k] * h[k-j]
    # Note that sum_k weights[k] * h[k-j] = sum_k h[k] * weights[k-j] (by substitution m=k-j)
    # So this equals 2 * corr[j] where corr is as computed above
    
    d_soft_max = 2.0 * corr[:N]
    
    # Now d(obj)/d(h[j]) where obj = abs(2*N*soft_max/(s*s))
    # Let f = 2*N*soft_max/(s*s)
    # obj = abs(f)
    # d(obj)/d(h[j]) = sign(f) * d(f)/d(h[j])
    # d(f)/d(h[j]) = 2*N * (d(soft_max)/d(h[j]) * s*s - soft_max * 2*s * ds/dh[j]) / (s*s)^2
    # ds/dh[j] = 1
    # d(f)/d(h[j]) = 2*N * (d_soft_max[j] * s - 2*soft_max) / (s^3)
    
    sign_f = np.sign(2.0 * N * soft_max / (s * s))
    if sign_f == 0:
        sign_f = 1.0
    
    grad = sign_f * 2.0 * N * (d_soft_max * s - 2.0 * soft_max) / (s ** 3)
    
    return obj, grad


def _diverse_init(max_time=15.0):
    """Generate diverse initial candidates."""
    t0 = time.time()
    best_h = None
    best_val = 1e18
    rng = np.random.default_rng(123)
    
    x = np.arange(N)
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Single cosine modes
    for k in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100]:
        for amp in [0.5, 1.0, 1.5, 2.0, 3.0]:
            h = np.ones(N) + amp * np.cos(2 * np.pi * k * x / N)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Chirp signals
    for freq in [5, 10, 20, 30, 50, 75]:
        for chirp in [-100, -50, 0, 50, 100]:
            t = x / N
            phase = 2 * np.pi * freq * t + 0.5 * chirp * t * t
            h = np.cos(phase)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Random Gaussian
    for trial in range(20):
        if time.time() - t0 > max_time * 0.5:
            break
        h = rng.normal(0, 1, N)
        s = np.sum(h)
        if abs(s) > 1e-12:
            candidates.append(h / s)
    
    # Block structure
    for nblocks in [4, 8, 16, 32]:
        block_size = N // nblocks
        for seed in range(5):
            if time.time() - t0 > max_time * 0.8:
                break
            rng_block = np.random.default_rng(seed + 1000)
            h = np.zeros(N)
            for b in range(nblocks):
                val = rng_block.normal(0, 1)
                h[b*block_size:(b+1)*block_size] = val
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Multi-frequency cosine
    for n_freq in range(2, 5):
        if time.time() - t0 > max_time * 0.85:
            break
        for trial in range(10):
            freqs = sorted(rng.uniform(1, 40, n_freq))
            phases = rng.uniform(0, 2*np.pi, n_freq)
            amps = rng.uniform(0.2, 1.0, n_freq)
            dc = rng.uniform(-0.1, 0.1)
            h = np.ones(N) * dc
            for f, p, a in zip(freqs, phases, amps):
                h += a * np.cos(2*np.pi*f*x/N + p)
            s = np.sum(h)
            if abs(s) > 1e-12:
                candidates.append(h / s)
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        v = _value(h)
        if v < best_val:
            best_val = v
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = 2.0
    
    return best_h, best_val


def _local_refine(h0, max_time=25.0):
    """Local refinement using smooth objective with multiple betas."""
    t0 = time.time()
    best_h = h0.copy()
    best_val = _value(h0)
    
    for beta in [30.0, 50.0, 80.0]:
        if time.time() - t0 > max_time * 0.8:
            break
        try:
            def obj_fn(h):
                return _smooth_objective_and_grad(h, beta=beta)[0]
            
            def grad_fn(h):
                return _smooth_objective_and_grad(h, beta=beta)[1]
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def _multi_start(h_best, n_starts=8, max_time=30.0):
    """Multi-start local refinement."""
    t0 = time.time()
    best_h = h_best.copy()
    best_val = _value(h_best)
    rng = np.random.default_rng(42)
    
    for trial in range(n_starts):
        if time.time() - t0 > max_time * 0.85:
            break
        
        scale = 0.1 + 0.05 * trial
        perturbation = rng.normal(0, scale, N)
        perturbation -= np.mean(perturbation)
        
        h_init = best_h + perturbation
        s = np.sum(h_init)
        if abs(s) < 1e-12:
            continue
        h_init = h_init / s
        
        try:
            def obj_fn(h):
                return _smooth_objective_and_grad(h, beta=40.0)[0]
            
            def grad_fn(h):
                return _smooth_objective_and_grad(h, beta=40.0)[1]
            
            res = minimize(
                obj_fn,
                h_init,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 300, 'ftol': 1e-14, 'gtol': 1e-11}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    return best_h, best_val


def construct():
    t0 = time.time()
    
    def elapsed():
        return time.time() - t0
    
    def time_left():
        return DEADLINE - elapsed()
    
    # Phase 1: Diverse initialization
    h_div, val_div = _diverse_init(max_time=12.0)
    
    best_h = h_div.copy()
    best_val = val_div
    
    # Include seed
    h_seed = np.ones(N)
    v_seed = _value(h_seed)
    if v_seed < best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Phase 2: Local refinement with multiple betas
    if time_left() > 30:
        best_h, best_val = _local_refine(best_h, max_time=min(25.0, time_left() - 15.0))
    
    # Phase 3: Multi-start refinement
    if time_left() > 25:
        best_h, best_val = _multi_start(best_h, n_starts=8, max_time=min(20.0, time_left() - 12.0))
    
    # Phase 4: Final high-precision polish
    if time_left() > 12:
        try:
            def obj_fn(h):
                return _smooth_objective_and_grad(h, beta=100.0)[0]
            
            def grad_fn(h):
                return _smooth_objective_and_grad(h, beta=100.0)[1]
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 600, 'maxfun': 1200, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Phase 5: Additional polish if time allows
    if time_left() > 8:
        try:
            def obj_fn(h):
                return _smooth_objective_and_grad(h, beta=150.0)[0]
            
            def grad_fn(h):
                return _smooth_objective_and_grad(h, beta=150.0)[1]
            
            res = minimize(
                obj_fn,
                best_h,
                jac=grad_fn,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 800, 'ftol': 1e-16, 'gtol': 1e-15}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                v = _value(res.x)
                if v < best_val:
                    best_val = v
                    best_h = res.x.copy()
        except Exception:
            pass
    
    # Safety: ensure valid output
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        best_h = np.ones(N)
    
    return best_h


if __name__ == '__main__':
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Min: {np.min(h)}, Max: {np.max(h)}")
