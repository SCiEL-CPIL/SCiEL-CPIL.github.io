import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute frequency indices for incremental FFT updates
_FREQ_IDX = np.arange(_FSIZE // 2 + 1)
_EXP_TABLE = np.exp(-2j * np.pi * _FREQ_IDX / _FSIZE)

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def _max_abs_ac(h):
    return np.max(np.abs(_autoconv_full(h)))

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
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

def _smooth_obj_grad(h, beta=50.0):
    """Smooth objective with gradient using FFT-based computation."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        return 0.0, np.zeros_like(h)
    
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    log_sum = np.log(np.sum(exp_scaled))
    sm = m + log_sum / beta
    
    # Gradient of soft-max w.r.t. ac
    weights = exp_scaled / np.sum(exp_scaled)
    grad_ac = beta * weights * np.sign(ac)
    
    # FFT-based gradient: grad_h[j] = sum_k grad_ac[k] * (h[j+k] + h[j-k]) for valid indices
    # term1[j] = sum_k grad_ac[k] * h[j+k] for 0<=j+k<N
    # term2[j] = sum_k grad_ac[k] * h[j-k] for 0<=j-k<N
    # term2 is standard linear convolution (grad_ac * h)
    # term1 is cross-correlation
    
    _FSIZE2 = 1
    while _FSIZE2 < 2*N:
        _FSIZE2 *= 2
    
    GA = np.fft.rfft(grad_ac, _FSIZE2)
    HA = np.fft.rfft(h, _FSIZE2)
    
    # term2: standard linear convolution
    conv2 = np.fft.irfft(GA * HA, _FSIZE2)[:N]
    
    # term1: cross-correlation, use reversed grad_ac
    grad_ac_rev = grad_ac[::-1]
    GAR = np.fft.rfft(grad_ac_rev, _FSIZE2)
    conv1_full = np.fft.irfft(GAR * HA, _FSIZE2)
    # conv1_full[n] = sum_m grad_ac[m] * h[n-N+1+m]
    # We want term1[j] = sum_m grad_ac[m] * h[j+m], so n = j+N-1
    term1 = conv1_full[N-1:N-1+N]
    
    grad_h = term1 + conv2
    
    # Gradient of sm/(s^2)
    grad_obj = grad_h / (s*s) - 2 * sm / (s**3)
    
    return sm / (s*s), grad_obj

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses incremental FFT updates.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            old_val = h[i]
            h[i] = -old_val
            new_sum = current_sum - 2 * old_val
            
            # Update FFT incrementally
            delta = -2 * old_val * _EXP_TABLE[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            
            if abs(new_sum) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                current_sum = new_sum
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=[3,5,7,10]):
    """
    Block flip local search: flip contiguous blocks of signs.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    current_sum = np.sum(h)
    improved = True
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        for bs in block_sizes:
            if bs <= 4:
                step = max(1, (N - bs) // 150)
            elif bs <= 7:
                step = max(1, (N - bs) // 100)
            else:
                step = max(1, (N - bs) // 60)
            
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                new_sum = current_sum - 2 * np.sum(block)
                
                inner_sum = np.sum(block * _EXP_TABLE[:bs])
                delta_vec = -2.0 * _EXP_TABLE[start] * inner_sum
                H_new = H + delta_vec
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                
                if abs(new_sum) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    current_sum = new_sum
                    improved = True
                else:
                    h[start:start+bs] = block  # revert
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # 1. Baseline: All ones
    best_x = np.ones(N, dtype=float)
    best_val = compute_raw_value(best_x)
    
    # 2. Generate diverse candidates
    candidates = [best_x.copy()]
    
    t = np.arange(N)
    # Chirp sequences
    for freq in [1.0, 2.0, 0.5, 3.0]:
        phase = 2 * np.pi * freq * (t**2) / N
        x_chirp = np.sign(np.cos(phase))
        x_chirp[x_chirp == 0] = 1
        candidates.append(x_chirp)
        
    # Linear chirp
    for freq in [1.0, 2.0, 0.5]:
        phase = 2 * np.pi * freq * t / N
        x_lin = np.sign(np.cos(phase))
        x_lin[x_lin == 0] = 1
        candidates.append(x_lin)
        
    # Barker-like
    x_barker = np.array([1 if i % 4 in [0, 2] else -1 for i in range(N)], dtype=float)
    candidates.append(x_barker)
    
    # Random sequences
    rng = np.random.default_rng(42)
    for _ in range(20):
        x_rand = rng.choice([1.0, -1.0], N)
        candidates.append(x_rand)
        
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    x_gold = np.sign(np.cos(2 * np.pi * phi * t))
    x_gold[x_gold == 0] = 1
    candidates.append(x_gold)
    
    # Welch-like sequences based on quadratic residues
    def _welch_seq(N_len, p):
        """Generate sequence based on quadratic residues mod p."""
        quad_res = set([(j*j) % p for j in range(1, p)])
        seq = np.zeros(N_len)
        for i in range(N_len):
            r = (i * i) % p
            if r == 0:
                seq[i] = 1
            elif r in quad_res:
                seq[i] = 1
            else:
                seq[i] = -1
        return seq
    
    for p in [7, 11, 13, 17, 19, 23, 31, 37, 41, 43]:
        if p <= N:
            x_welch = _welch_seq(N, p)
            candidates.append(x_welch)
    
    # 3. Evaluate all candidates
    for x in candidates:
        val = compute_raw_value(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
            
    # 4. Smooth surrogate optimization with adaptive beta schedule
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    top_starts = scored_candidates[:6]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Adaptive beta schedule
        for beta in [30, 80, 150, 300]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj(x, b=beta):
                    return _smooth_obj(x, beta=b)
                
                max_iter = 300 if beta <= 80 else 800
                
                res = minimize(obj, h_norm, method='L-BFGS-B',
                               options={'maxiter': max_iter, 'disp': False, 'gtol': 1e-10,
                                        'ftol': 1e-12, 'maxls': 50})
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
                
                h_norm = h_bin / np.sum(h_bin)
                
            except Exception:
                pass
    
    # 5. Peak-suppression phase
    if time.time() < deadline - 25:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        H = np.fft.rfft(h_work, _FSIZE)
        current_sum = np.sum(h_work)
        best_val_work = best_val
        
        for round_num in range(6):
            if time.time() > deadline - 10:
                break
            
            ac = np.fft.irfft(H * H, _FSIZE)[:2*N-1]
            abs_ac = np.abs(ac)
            
            peak_indices = np.argsort(abs_ac)[::-1][:20]
            
            improved_this_round = False
            for peak_idx in peak_indices:
                if time.time() > deadline - 8:
                    break
                
                center = peak_idx % N
                
                for flip_offset in range(-4, 5):
                    if time.time() > deadline - 8:
                        break
                    
                    j = (center + flip_offset) % N
                    old_val = h_work[j]
                    h_work[j] = -old_val
                    new_sum = current_sum - 2 * old_val
                    
                    delta = -2 * old_val * _EXP_TABLE[j]
                    H_new = H + delta
                    conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                    
                    if abs(new_sum) < 1e-12:
                        new_val = float('inf')
                    else:
                        new_val = np.max(np.abs(conv)) / (new_sum * new_sum)
                    
                    if new_val < best_val_work:
                        best_val_work = new_val
                        H = H_new
                        current_sum = new_sum
                        improved_this_round = True
                    else:
                        h_work[j] = old_val
                
                if improved_this_round:
                    break
            
            if not improved_this_round:
                break
        
        if best_val_work < best_val:
            best_val = best_val_work
            best_x = h_work.copy()
    
    # 6. Local search refinement
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        h_work, val_work = _local_search_flip(h_work, max_passes=20, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
            
        if time.time() < deadline - 6:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 4, 
                                                    block_sizes=[2, 3, 4, 5, 7, 9, 13])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 7. Random restarts with structured perturbations
    if time.time() < deadline - 20:
        for _ in range(12):
            if time.time() > deadline - 10:
                break
            
            perturbation_type = rng.integers(0, 3)
            if perturbation_type == 0:
                x_rand = rng.choice([1.0, -1.0], N)
            elif perturbation_type == 1:
                x_rand = best_x.copy()
                n_flips = rng.integers(5, 30)
                flip_idx = rng.choice(N, n_flips, replace=False)
                x_rand[flip_idx] = -x_rand[flip_idx]
            else:
                x_rand = best_x.copy()
                bs = rng.integers(3, 20)
                start = rng.integers(0, N - bs)
                x_rand[start:start+bs] = -x_rand[start:start+bs]
            
            h_work, val_work = _local_search_flip(x_rand, max_passes=10, deadline=deadline - 5)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
                
    # 8. Final high-precision polish
    if time.time() < deadline - 5:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        for _ in range(5):
            if time.time() > deadline - 2:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=10, deadline=deadline - 2)
            if val_work >= best_val:
                break
            best_val = val_work
            best_x = h_work.copy()
        
        if time.time() < deadline - 3:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 1,
                                                    block_sizes=[2, 3, 4, 5, 7])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
            
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
