"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Precompute FFT size for linear autoconvolution of length N
N_FFT = 1
while N_FFT < 2 * N - 1:
    N_FFT <<= 1

def _autoconv(f):
    """Computes the full linear autoconvolution of f using FFT."""
    F = np.fft.rfft(f, n=N_FFT)
    return np.fft.irfft(F * F, n=N_FFT)

def _score(f):
    """Compute max(|autoconv|) / (sum(f))^2. Lower is better."""
    s = np.sum(f)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(f)
    max_abs = np.max(np.abs(ac))
    return max_abs / (s * s)

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
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        return max_ac, np.zeros_like(f)
        
    log_sum = np.log(sum_exp) / beta + max_ac
    
    weights = exp_shifted / sum_exp
    signs = np.sign(ac)
    
    # Vectorized gradient: grad[i] = 2 * sum_k weights[k] * sign(ac[k]) * f[k-i] for valid k-i
    # Let g[k] = weights[k] * signs[k] for k in [0, 2N-2)
    # grad[i] = 2 * sum_j g[i+j] * f[j]
    g = weights * signs
    # For N=400, this is 400 dots of length 400 = 160K ops, very fast
    # Using a loop is acceptable for this small N, but we can try to vectorize if needed.
    # Actually, simple loop is fine and less error-prone for the boundary conditions.
    grad = np.zeros(N)
    for i in range(N):
        grad[i] = 2.0 * np.dot(g[i:i+N], f)
    
    return log_sum, grad

def _objective_and_grad(f):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum f|)
    This is scale invariant and smooth.
    """
    smax, grad_smax = _soft_max_abs_ac(f)
    s = np.sum(f)
    
    if s == 0 or smax < 1e-30:
        return 1e10, np.ones(N) * 1e-6
    
    # d/d f log(smax) = grad_smax / smax
    # d/d f (-2 log|s|) = -2 * ones / s
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones(N) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _local_search(f, deadline, max_iters=20000):
    """Local search: try flipping signs and small perturbations"""
    best = f.copy()
    best_score = _score(best)
    if not np.isfinite(best_score):
        return best, best_score

    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Try random sign flips (more aggressive)
        n_flips = rng.integers(1, 6)
        indices = rng.choice(N, n_flips, replace=False)
        candidate = best.copy()
        candidate[indices] *= -1
        
        score = _score(candidate)
        if score < best_score:
            best = candidate
            best_score = score
        else:
            # Try small random perturbations occasionally
            if it % 5 == 0:
                candidate = best + rng.standard_normal(N) * 0.01
                score = _score(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
    
    return best, best_score

def _spectral_init(seed=0):
    """
    Spectral shaping initialization: design frequency response to suppress
    large autoconvolution peaks, then invert to time domain.
    """
    rng = np.random.RandomState(seed)
    # Design a frequency response that is flat (white) to minimize peaks
    # Start with random phase, flat magnitude, then shape to reduce peaks
    n_freqs = N_FFT // 2 + 1
    # Random initial phases
    phases = rng.uniform(0, 2 * np.pi, size=n_freqs)
    # Start with flat magnitude
    mags = np.ones(n_freqs)
    mags[0] = 0  # zero DC to ensure sum is controlled
    
    F = mags * np.exp(1j * phases)
    # Make it conjugate symmetric for real signal
    F_full = np.zeros(N_FFT, dtype=complex)
    F_full[:n_freqs] = F
    if n_freqs > 1:
        F_full[n_freqs:] = np.conj(F[1:n_freqs][::-1])
    
    f = np.fft.irfft(F_full, n=N_FFT)[:N]
    
    # Iteratively suppress peaks in autoconvolution
    for _ in range(20):
        ac = _autoconv(f)
        s = np.sum(f)
        if abs(s) < 1e-12:
            break
        # Find where peaks are
        peak_idx = np.argmax(np.abs(ac))
        # Reduce contribution at this lag by adjusting signal
        # Simple approach: subtract a small multiple of the signal shifted
        shift = peak_idx % N
        if shift == 0:
            f -= 0.01 * f
        else:
            # Adjust to reduce correlation at this lag
            f -= 0.005 * np.roll(f, shift)
    
    # Normalize
    s = np.sum(f)
    if abs(s) > 1e-12:
        f = f / s  # scale so sum = 1
    else:
        f = rng.randn(N)
    
    return f

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_f = None
    best_score = float('inf')
    
    # Initial guesses - more restarts since gradient is now fast
    n_restarts = 15
    np.random.seed(42)
    
    # List of initializers
    initializers = []
    
    # 1. All ones
    initializers.append(np.ones(N))
    
    # 2. Random Gaussian
    initializers.append(np.random.randn(N))
    
    # 3. Random ±1
    f_rand = np.random.choice([-1.0, 1.0], size=N)
    if np.sum(f_rand) < 0:
        f_rand = -f_rand
    initializers.append(f_rand)
    
    # 4. Chirp
    t = np.linspace(0, 10, N)
    initializers.append(np.cos(t**2))
    
    # 5. Alternating blocks
    f_alt = np.ones(N)
    f_alt[::2] = -1.0
    initializers.append(f_alt)
    
    # 6. Random with structure (repeated blocks)
    f_struct = np.random.randn(N // 2)
    f_struct = np.repeat(f_struct, 2)
    initializers.append(f_struct)
    
    # 7. Sine wave
    t_sin = np.linspace(0, 4 * np.pi, N)
    initializers.append(np.sin(t_sin))
    
    # 8-12. Spectral inits
    for i in range(5):
        initializers.append(_spectral_init(seed=i * 100 + 1))
        
    # 13-15. More random with block structure
    for i in range(3):
        block_size = np.random.randint(5, 20)
        n_blocks = N // block_size
        blocks = np.random.randn(n_blocks)
        initializers.append(np.repeat(blocks, block_size)[:N])
    
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if r < len(initializers):
            f0 = initializers[r].copy()
        else:
            f0 = np.random.randn(N)
        
        # Ensure non-zero sum
        s = np.sum(f0)
        if abs(s) < 1e-6:
            f0 += 1e-3 * np.random.randn(N)
        
        # Optimization using L-BFGS-B with soft-max objective
        try:
            res = minimize(
                lambda x: _objective_and_grad(x)[0],
                f0,
                jac=lambda x: _objective_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            f_opt = res.x
            # Snap to discrete ±1 if close, or keep continuous?
            # The task says "signed step heights", which implies real values are allowed,
            # but "signed step functions" often implies discrete.
            # However, the score is max_abs / sum^2. 
            # If we allow continuous values, we can potentially do better?
            # Let's check the objective. 
            # If we scale f, the objective is invariant (numerator scales by c^2, denominator by c^2).
            # So we can normalize sum to 1.
            # The problem is to minimize max |ac|.
            # With continuous values, we might get lower max |ac| than ±1?
            # Actually, for ±1, ac values are integers.
            # For continuous, we can smooth out peaks.
            # Let's keep the continuous result from optimization, but maybe refine with local search on ±1?
            # The prompt says "sequence of signed step heights". It doesn't strictly say ±1.
            # But the seed program used ±1.
            # Let's try to use the continuous result if it's better, but also try snapping to ±1.
            
            score_cont = _score(f_opt)
            
            # Try snapping to ±1
            f_snap = np.sign(f_opt)
            # Handle zeros
            f_snap[f_snap == 0] = 1.0
            score_snap = _score(f_snap)
            
            # Also try local search on the continuous result? 
            # Local search defined above flips signs, which is discrete.
            # Let's run local search on the snapped version and the continuous version (perturbed).
            
            # 1. Local search on snapped
            f_ls_snap, score_ls_snap = _local_search(f_snap, deadline - 5.0, max_iters=5000)
            
            # 2. Local search on continuous (using small perturbations)
            # The current _local_search flips signs. Let's make a variant or just use it on snapped.
            # For continuous, we can just do a few more gradient steps or random perturbations.
            # Let's stick to discrete local search for now as it's more robust for "step functions".
            
            # Compare candidates
            candidates = [
                (f_opt, score_cont),
                (f_snap, score_snap),
                (f_ls_snap, score_ls_snap)
            ]
            
            for f_cand, s_cand in candidates:
                if s_cand < best_score:
                    best_score = s_cand
                    best_f = f_cand.copy()
                    
        except Exception:
            continue
            
    # Final polish: intensive local search on the best found
    if best_f is not None and time.time() < deadline - 5.0:
        best_f, best_score = _local_search(best_f, deadline, max_iters=20000)
        
    # If no valid result found, return a default
    if best_f is None:
        best_f = np.ones(N)
        
    return best_f
# EVOLVE-BLOCK-END
