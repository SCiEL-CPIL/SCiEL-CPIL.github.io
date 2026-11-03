"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_fft(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _soft_max_abs(ac, beta=50.0):
    """
    Smooth approximation of max(|ac|) using log-sum-exp.
    This is differentiable and allows gradient-based optimization.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2
    We optimize this to find good candidates, then evaluate the true objective.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _grad_smooth(h, beta=50.0):
    """
    Analytical gradient of the smooth objective.
    f = S / s^2 where S = soft_max_abs(ac), s = sum(h)
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        S = 0.0
        w = np.zeros_like(ac)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        log_sum = np.log(np.sum(exp_scaled))
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / np.sum(exp_scaled)
    
    # Gradient of S w.r.t. h
    # S = (1/beta) log(sum(exp(beta * |ac_k|)))
    # dS/dh_i = (1/beta) * sum_k [ beta * sign(ac_k) * ac_k / |ac_k| * w_k * dac_k/dh_i ]
    # dac_k/dh_i = h_{k-i} (cyclic, but we use linear conv so it's h_{k-i} if 0<=k-i<N)
    # Note: ac_k = sum_j h_j h_{k-j}
    # dac_k/dh_i = h_{k-i} + h_i h_{k-i} ... wait.
    # ac_k = sum_j h_j h_{k-j}
    # d/dh_i (sum_j h_j h_{k-j}) = h_{k-i} + h_i h_{k-i} ? No.
    # If i == j, term is h_i h_{k-i}. If i == k-j, term is h_j h_i.
    # So dac_k/dh_i = h_{k-i} + h_{k-i} = 2 h_{k-i} (if i != k-i)
    # If i == k-i (k=2i), then term is h_i^2, derivative is 2 h_i.
    # So dac_k/dh_i = 2 h_{k-i} is correct for linear convolution if we treat it as product rule.
    # Actually, for linear convolution, the gradient of sum_k w_k |ac_k| w.r.t h_i
    # is sum_k w_k sign(ac_k) 2 h_{k-i} (with appropriate bounds).
    
    # Let's compute the gradient of S w.r.t h.
    # dS/dh_i = sum_k w_k sign(ac_k) dac_k/dh_i
    # dac_k/dh_i = 2 h_{k-i} if 0 <= k-i < N, else 0.
    # So dS/dh = 2 * (w * sign(ac)) convolved with h (linear).
    
    # Let v_k = w_k * sign(ac_k)
    # dS/dh_i = 2 * sum_k v_k h_{k-i}
    # This is 2 * (v conv h)_i (linear convolution)
    
    v = w * np.sign(ac)
    
    # Linear convolution of v and h
    # v has length 2N-1, h has length N.
    # We need (v conv h)_i for i in 0..N-1.
    # (v conv h)_i = sum_k v_k h_{i-k} ? Or sum_k v_{i-k} h_k?
    # Let's stick to the definition: dac_k/dh_i = 2 h_{k-i}.
    # dS/dh_i = sum_k w_k sign(ac_k) 2 h_{k-i} = 2 sum_k v_k h_{k-i}.
    # This is 2 * (v * h)_i where * is cross-correlation? No.
    # sum_k v_k h_{k-i} is the cross-correlation of v and h at lag i.
    # Let's compute it via FFT or direct sum if small. N=400, so 2N-1=799.
    # Direct sum is O(N^2) = 160k ops, which is fine.
    
    # Efficient way: FFT
    # corr = np.correlate(v, h, mode='full')
    # We want indices 0..N-1 of the correlation?
    # Let's check indices.
    # ac_k = sum_j h_j h_{k-j}.
    # dS/dh_i = 2 sum_k v_k h_{k-i}.
    # Let m = k-i. k = m+i.
    # sum_m v_{m+i} h_m.
    # This is the cross-correlation of v and h.
    
    # Using FFT for correlation:
    # R = ifft(fft(v) * conj(fft(h)))
    # But v is length 799, h is 400.
    # Let's just use np.correlate.
    # np.correlate(v, h, mode='full')
    # The full correlation has length N + 799 - 1 = 1198.
    # We need the part corresponding to i=0..399.
    # In np.correlate(a, v, mode='full'), a is the data, v is the kernel.
    # r[k] = sum_n a[n+k] conj(v[n])
    # We want sum_m v_{m+i} h_m.
    # This is sum_m h_m v_{m+i}.
    # This is the cross-correlation of h and v at lag -i?
    # Let's just compute the gradient numerically or use the FFT trick carefully.
    
    # Simpler: compute g_i = 2 * sum_k v_k h_{k-i}
    # This is the convolution of v and h, but with v reversed?
    # (v * h)_i = sum_k v_k h_{i-k}.
    # We have sum_k v_k h_{k-i} = sum_k v_{k} h_{-(i-k)} ?
    # Let j = k-i. k = j+i.
    # sum_j v_{j+i} h_j.
    # This is the cross-correlation <v, h> at lag i.
    
    # Let's use FFT.
    # fsize = 2*N
    # V = fft(v, fsize)
    # H = fft(h, fsize)
    # C = ifft(V * conj(H))
    # C[i] = sum_n v[n] conj(h[n-i]) ?
    # Standard: ifft(fft(v) * conj(fft(h))) gives cross-correlation.
    # c[k] = sum_n v[n+k] conj(h[n])
    # We want sum_m v_{m+i} h_m.
    # This matches c[i] if h is real.
    
    fsize = 1
    while fsize < 2 * (2 * N - 1):
        fsize *= 2
        
    V = np.fft.rfft(v, fsize)
    H = np.fft.rfft(h, fsize)
    C = np.fft.irfft(V * np.conj(H), fsize)
    
    # C[i] should be sum_n v[n+i] h[n]
    # We need i from 0 to N-1.
    dS_dh = 2 * C[:N]
    
    # Now, f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s * 1) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (dS_dh * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_flip(h, max_passes=5, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    """
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3,5,7]):
    """
    Block flip local search: flip contiguous blocks of signs.
    """
    h = h.copy()
    best_val = _score(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 50)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = _score(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # Safety margin before 150s limit

    best_score = np.inf
    best_h = np.ones(N) / N  # Baseline: constant sequence

    base_score = _score(best_h)
    if base_score < best_score:
        best_score = base_score
        best_h = best_h.copy()

    # Initialization strategies
    inits = []
    
    # 1. All ones
    inits.append(np.ones(N))
    
    # 2. Cosine windows
    x = np.arange(N)
    inits.append(np.cos(np.pi * x / N))
    inits.append(np.cos(np.pi * x / (2*N))**2)
    
    # 3. Gaussian window
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-(x - mu)**2 / (2*sigma**2)))
    
    # 4. Triangle window
    inits.append(np.maximum(0, 1 - np.abs(x - mu)/mu))
    
    # 5. Chirp sequence
    phase = np.pi * x**2 / N
    inits.append(np.cos(phase))
    
    # 6. Barker-like sequence
    inits.append(np.array([1.0 if i % 4 in [0, 2] else -1.0 for i in range(N)]))
    
    # 7. Random smoothed sequences
    rng = np.random.default_rng(42)
    for i in range(5):
        noise = rng.standard_normal(N)
        kernel = np.ones(10)/10
        smooth = np.convolve(noise, kernel, mode='same')
        inits.append(smooth)
        
    # 8. Random binary sequences
    for i in range(5):
        inits.append(rng.choice([1.0, -1.0], N))

    # Optimization loop using smooth surrogate
    for h_init in inits:
        if time.time() > deadline - 20:
            break
            
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        # Normalize to sum 1 to simplify optimization landscape
        h_norm = h_init / s0
        
        # Use smooth objective for gradient-based optimization
        try:
            # Start with moderate beta, increase for refinement
            for beta in [20, 50, 100, 200]:
                if time.time() > deadline - 10:
                    break
                
                def obj(x):
                    val, grad = _grad_smooth(x, beta)
                    return val, grad
                
                res = minimize(obj, h_norm, method='L-BFGS-B', jac=True,
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-5})
                h_opt = res.x
                
                # Check validity
                s_opt = np.sum(h_opt)
                if abs(s_opt) < 1e-6:
                    continue
                    
                # Evaluate the actual objective
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
                
                # Also try sign-binarized version
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = _score(h_bin)
                if val_bin < best_score:
                    best_score = val_bin
                    best_h = h_bin.copy()
                    
        except Exception:
            pass

    # Final local search refinement
    if time.time() < deadline - 10:
        # First, try to refine the continuous solution with sign flips
        h_work = best_h.copy()
        h_refined, val_refined = _local_search_flip(h_work, max_passes=3, deadline=deadline-5)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()
            
        # Try block flips
        if time.time() < deadline - 5:
            h_work = best_h.copy()
            h_refined, val_refined = _local_search_block(h_work, deadline=deadline-2)
            if val_refined < best_score:
                best_score = val_refined
                best_h = h_refined.copy()

    # Final check against baseline
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N) / N

    return best_h

# EVOLVE-BLOCK-END
