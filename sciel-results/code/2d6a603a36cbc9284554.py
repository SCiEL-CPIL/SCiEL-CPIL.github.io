import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _ac_fft(x):
    """Full linear autoconvolution via FFT (real)."""
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    ac = np.fft.irfft(X * X, fsize)
    return ac[:size]

def _waterfill(h, max_iters=400, deadline=None):
    """
    Thresholded projected gradient descent on the autoconvolution.
    Iteratively suppresses the largest |ac_k| by nudging h in the
    direction that reduces that peak (conjugate of the peak index).
    After each step, re-binarize to {-1,1} (sign projection) to keep
    the sequence in the discrete feasible set.
    """
    h = np.sign(h).astype(float)
    h[h == 0] = 1.0
    best_h = h.copy()
    best_val = compute_raw_value(h)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        ac = _ac_fft(h)
        abs_ac = np.abs(ac)
        m = np.max(abs_ac)
        if m < 1e-12:
            break
        
        # Find all near-max peaks (within 5% of max)
        thresh = 0.95 * m
        peak_idx = np.where(abs_ac >= thresh)[0]
        
        # For each peak at lag k, the gradient of |ac_k| w.r.t. h
        # is sign(ac_k) * (shifted h). We want to DECREASE |ac_k|,
        # so we move h opposite to that gradient.
        # ac_k = sum_i h_i * h_{i+k} (for k >= 0)
        # d|ac_k|/dh_i = sign(ac_k) * (h_{i+k} for i in range where valid)
        
        # Build a correction vector: for each peak lag k,
        # the "conjugate" signal is sign(ac_k) * h shifted by -k
        # We subtract a small multiple of this from h to reduce the peak.
        
        # Efficient: compute the aggregate gradient
        # For lag k >= 0: ac_k = sum_{i=0}^{N-1-k} h_i * h_{i+k}
        # grad_k[i] = sign(ac_k) * h_{i+k} for i in [0, N-1-k]
        #           = sign(ac_k) * h_i for i in [k, N-1]  (from ac_{-k})
        
        # Simpler approach: use the FFT to compute the "score" vector
        # that, when subtracted from h, most reduces the max peak.
        # The subgradient of max_k |ac_k| is sign(ac_k*) * conj(h shifted)
        # where k* is argmax.
        
        # Use a soft version: weight peaks by their magnitude
        weights = abs_ac / (m + 1e-15)  # normalized weights, peaks near 1
        # Only keep significant peaks
        weights[weights < 0.5] = 0
        
        # Compute the weighted sum of shifted conjugates
        # This is essentially a filtered version of h
        # Using FFT: the "gradient" in h-space of sum_k w_k |ac_k|
        # is related to the IFFT of W * H (in frequency domain)
        
        fsize = 1
        while fsize < 2 * N - 1:
            fsize *= 2
        H = np.fft.rfft(h, fsize)
        # The autoconvolution in freq domain is H^2
        # d/dh of |ac_k| ~ sign(ac_k) * IFFT(H * delta_k) where delta_k is at freq k
        # More directly: the gradient of ac = h*h w.r.t. h is 2 * (circular conv of h with reversed h)
        # For linear: grad[i] = sum_k w_k * sign(ac_k) * h_{i-k} (with appropriate shifts)
        
        # Practical approach: for the dominant peak, compute the conjugate signal
        k_star = np.argmax(abs_ac)
        sign_k = np.sign(ac[k_star])
        
        # The signal to subtract: for lag k_star,
        # if k_star >= 0: contrib[i] = sign_k * h[i + k_star] (for i < N - k_star)
        #                  contrib[i] = sign_k * h[i - k_star] (for i >= k_star, from negative lag)
        # For autoconvolution, ac_k = ac_{-k}, so we get contributions from both sides.
        
        contrib = np.zeros(N)
        if k_star < N:
            # Positive lag contribution
            if k_star > 0:
                contrib[:N - k_star] += sign_k * h[k_star:]
            # Negative lag contribution (same as positive by symmetry)
            contrib[k_star:] += sign_k * h[:N - k_star]
        
        # Normalize the correction
        cnorm = np.linalg.norm(contrib)
        if cnorm < 1e-12:
            break
        
        # Adaptive step size
        step = 0.3 * (m / (np.sum(abs_ac) / (2 * N - 1) + 1e-15))
        step = min(step, 1.0)
        
        # Apply correction and re-binarize
        h_new = h - step * (contrib / cnorm)
        h_bin = np.sign(h_new)
        h_bin[h_bin == 0] = 1.0
        
        new_val = compute_raw_value(h_bin)
        if new_val < best_val:
            best_val = new_val
            best_h = h_bin.copy()
            h = h_bin
        else:
            # Try smaller step
            h_new2 = h - 0.1 * step * (contrib / cnorm)
            h_bin2 = np.sign(h_new2)
            h_bin2[h_bin2 == 0] = 1.0
            new_val2 = compute_raw_value(h_bin2)
            if new_val2 < best_val:
                best_val = new_val2
                best_h = h_bin2.copy()
                h = h_bin2
            else:
                # No improvement, break or try different peak
                break
    
    return best_h, best_val

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _max_abs_ac(h):
    """Maximum absolute value of autoconvolution."""
    return np.max(np.abs(_autoconv_full(h)))

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    n = len(x)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    X = np.fft.rfft(x, fsize)
    conv = np.fft.irfft(X * X, fsize)
    conv = conv[:size]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs(ac, beta=50.0):
    """Smooth approximation of max(|ac|) using log-sum-exp."""
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    log_sum = np.log(np.sum(np.exp(scaled)))
    return m + log_sum / beta

def _smooth_obj(h, beta=50.0):
    """Smooth objective: soft_max_abs(autoconv(h)) / (sum(h))^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    sm = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _peak_suppression_obj(h, beta=50.0, peak_weight=2.0):
    """
    Objective that specifically penalizes the peak of the autoconvolution.
    This helps escape local minima where a single dominant peak remains.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0
    # Weight the peak more heavily
    peak_mask = abs_ac > 0.9 * m
    peak_penalty = np.sum(abs_ac[peak_mask]) * peak_weight
    other_penalty = np.sum(abs_ac[~peak_mask])
    total_penalty = peak_penalty + other_penalty
    return total_penalty / (s * s)

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Works on binary {-1, 1} sequences. Vectorized evaluation.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    
    # Precompute FFT for fast updates
    fsize = 1
    while fsize < 2 * N - 1:
        fsize *= 2
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = compute_raw_value(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3, 5, 7, 10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    Helps escape local minima of single-bit flips.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            for start in range(0, N - bs + 1, max(1, (N - bs) // 50)):
                if deadline is not None and time.time() > deadline:
                    break
                h[start:start+bs] = -h[start:start+bs]
                new_val = compute_raw_value(h)
                if new_val < best_val:
                    best_val = new_val
                    improved = True
                else:
                    h[start:start+bs] = -h[start:start+bs]  # revert
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    # Chirp sequences
    t = np.arange(N)
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
    
    # Linear chirp (constant frequency)
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
    
    # Barker-like structured
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(15):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
    
    # Golden ratio based
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # 4. Smooth surrogate optimization from best candidates
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 5 diverse starts
    top_starts = scored_candidates[:5]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        # Normalize to sum=1 for optimization
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Multi-stage beta optimization with adaptive schedule
        for beta in [60, 120, 250]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x):
                    return _smooth_obj(x, beta=beta)
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-7})
                h_opt = res.x
                
                # Evaluate the actual objective
                val = compute_raw_value(h_opt)
                if val < best_val:
                    best_val = val
                    best_x = h_opt.copy()
                
                # Also try sign-binarized version
                h_bin = np.sign(h_opt)
                h_bin[h_bin == 0] = 1
                val_bin = compute_raw_value(h_bin)
                if val_bin < best_val:
                    best_val = val_bin
                    best_x = h_bin.copy()
                    
            except Exception:
                pass
    
    # 5. Peak suppression phase: specifically target the argmax of autoconvolution
    if time.time() < deadline - 30:
        h_work = best_x.copy()
        s0 = np.sum(h_work)
        if abs(s0) > 1e-6:
            h_norm = h_work / s0
            
            for beta in [100, 200]:
                if time.time() > deadline - 15:
                    break
                try:
                    def peak_obj(x):
                        return _peak_suppression_obj(x, beta=beta, peak_weight=3.0)
                    
                    res = minimize(peak_obj, h_norm, method='L-BFGS-B',
                                   options={'maxiter': 200, 'disp': False, 'gtol': 1e-7})
                    h_opt = res.x
                    
                    val = compute_raw_value(h_opt)
                    if val < best_val:
                        best_val = val
                        best_x = h_opt.copy()
                    
                    h_bin = np.sign(h_opt)
                    h_bin[h_bin == 0] = 1
                    val_bin = compute_raw_value(h_bin)
                    if val_bin < best_val:
                        best_val = val_bin
                        best_x = h_bin.copy()
                        
                except Exception:
                    pass
    
    # 6. WATER-FILLING PHASE: thresholded projected gradient on autoconvolution peaks
    # This directly attacks the maximum absolute autoconvolution value by
    # iteratively finding the dominant peak and nudging h to reduce it.
    if time.time() < deadline - 25:
        # Run water-filling from multiple starting points
        wf_starts = [best_x.copy()]
        # Add a few diverse starts
        for val_c, x_c in scored_candidates[:3]:
            if x_c is not best_x:
                wf_starts.append(x_c.copy())
        
        for x_start in wf_starts:
            if time.time() > deadline - 15:
                break
            h_wf, val_wf = _waterfill(x_start, max_iters=300, deadline=deadline - 10)
            if val_wf < best_val:
                best_val = val_wf
                best_x = h_wf.copy()
    
    # 7. Local search refinement on best result
    if time.time() < deadline - 12:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Single-bit flip local search
        h_work, val_work = _local_search_flip(h_work, max_passes=5, deadline=deadline - 6)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # Block flip local search
        if time.time() < deadline - 4:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 2, 
                                                    block_sizes=[3, 5, 7])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 8. Final water-filling polish after local search
    if time.time() < deadline - 8:
        h_wf, val_wf = _waterfill(best_x, max_iters=200, deadline=deadline - 3)
        if val_wf < best_val:
            best_val = val_wf
            best_x = h_wf.copy()
    
    # 9. Final random restarts with local search if time remains
    if time.time() < deadline - 15:
        for _ in range(3):
            if time.time() > deadline - 8:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_work, val_work = _local_search_flip(x_rand, max_passes=2, deadline=deadline - 5)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            # Also try water-filling from this random start
            if time.time() < deadline - 5:
                h_wf, val_wf = _waterfill(h_work, max_iters=100, deadline=deadline - 3)
                if val_wf < best_val:
                    best_val = val_wf
                    best_x = h_wf.copy()
    
    # Ensure we return a valid array
    return best_x

if __name__ == "__main__":
    # Quick self-test
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
