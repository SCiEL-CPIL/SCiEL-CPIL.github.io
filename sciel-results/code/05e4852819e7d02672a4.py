import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# Precompute basis for FFT updates
_freqs = np.arange(_FSIZE // 2 + 1)
_basis = np.exp(-2j * np.pi * _freqs[None, :] * np.arange(N)[:, None] / _FSIZE)

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

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

def _soft_max_abs_and_grad(h, beta=50.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        scaled = beta * (abs_ac - m)
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # v_k = w_k * sign(ac_k)
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h
        # dS/dh_i = 2 * sum_k v_k h_{k-i}
        # This is 2 * cross_correlation(v, h)
        
        # Use FFT for cross-correlation
        fsize_corr = 1
        while fsize_corr < 3 * N:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        C = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        grad_S = 2 * C[:N]
    
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _local_search_flip(h, max_passes=10, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            # Update FFT incrementally
            delta = -2 * old_val * _basis[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val

def _local_search_block(h, deadline=None, block_sizes=None):
    """
    Block flip local search: flip contiguous blocks of signs.
    Uses FFT incrementally for speedup with precomputed basis.
    """
    if block_sizes is None:
        block_sizes = [2, 3, 5, 7, 11, 15]
    
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    improved = True
    pass_count = 0
    while improved and (deadline is None or time.time() < deadline):
        improved = False
        pass_count += 1
        # Adaptive: use larger steps in later passes
        step_base = max(1, (N - 15) // (100 - min(pass_count * 10, 60)))
        
        for bs in block_sizes:
            step = max(1, step_base)
            for start in range(0, N - bs + 1, step):
                if deadline is not None and time.time() > deadline:
                    break
                # Flip block
                block = h[start:start+bs].copy()
                h[start:start+bs] = -h[start:start+bs]
                # Update FFT using precomputed basis
                delta = np.zeros(_FSIZE//2+1, dtype=complex)
                for j in range(bs):
                    idx = start + j
                    delta += -2 * block[j] * _basis[idx]
                H_new = H + delta
                conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
                integral = np.sum(h)
                if abs(integral) < 1e-12:
                    new_val = float('inf')
                else:
                    new_val = np.max(np.abs(conv)) / (integral * integral)
                
                if new_val < best_val:
                    best_val = new_val
                    H = H_new
                    improved = True
                else:
                    h[start:start+bs] = block  # revert
    return h, best_val

def _simulated_annealing(h_init, deadline):
    """
    Structural change: Simulated Annealing to escape local optima.
    Uses incremental FFT updates for speed.
    """
    h = h_init.copy()
    h = np.sign(h)
    h[h == 0] = 1
    
    T_start = 0.5
    T_end = 0.01
    current_val = compute_raw_value(h)
    best_val = current_val
    best_h = h.copy()
    
    H = np.fft.rfft(h, _FSIZE)
    
    # Number of steps to fit in time
    # Estimate time per step
    t0 = time.time()
    # Dummy step
    i = 0
    old_val = h[i]
    h[i] = -old_val
    delta = -2 * old_val * _basis[i]
    H_new = H + delta
    conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        new_val = float('inf')
    else:
        new_val = np.max(np.abs(conv)) / (integral * integral)
    h[i] = old_val
    t1 = time.time()
    time_per_step = max(1e-6, (t1 - t0))
    
    # Total steps allowed
    time_remaining = deadline - time.time()
    max_steps = int(time_remaining / time_per_step)
    
    if max_steps < 100:
        return best_h, best_val
        
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        # Cooling schedule
        alpha = T_end / T_start
        T = T_start * (alpha ** (step / max_steps))
        
        # Pick random index
        i = np.random.randint(0, N)
        
        old_val = h[i]
        h[i] = -old_val
        
        # Update FFT
        delta = -2 * old_val * _basis[i]
        H_new = H + delta
        conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
        integral = np.sum(h)
        if abs(integral) < 1e-12:
            new_val = float('inf')
        else:
            new_val = np.max(np.abs(conv)) / (integral * integral)
        
        delta_val = new_val - current_val
        
        # Acceptance criterion
        if delta_val < 0 or np.random.rand() < np.exp(-delta_val / T):
            current_val = new_val
            H = H_new
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            # Revert
            h[i] = old_val
            
    return best_h, best_val

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
    
    # 4. Smooth surrogate optimization from best candidates using analytical gradient
    scored_candidates = []
    for x in candidates:
        val = compute_raw_value(x)
        scored_candidates.append((val, x))
    scored_candidates.sort(key=lambda t: t[0])
    
    # Optimize from top 8 diverse starts
    top_starts = scored_candidates[:8]
    
    for init_val, h_init in top_starts:
        if time.time() > deadline - 40:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        h_norm = h_init / s0
        
        # Multi-stage optimization: use analytical gradient
        for beta in [20, 50, 100]:
            if time.time() > deadline - 20:
                break
            
            try:
                def obj_and_grad(x, b=beta):
                    return _soft_max_abs_and_grad(x, beta=b)
                
                # Use L-BFGS-B with analytical gradient
                res = minimize(obj_and_grad, h_norm, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': 300, 'disp': False, 'gtol': 1e-8})
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
    
    # 5. Local search refinement on best result
    if time.time() < deadline - 15:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        
        # Single-bit flip local search
        h_work, val_work = _local_search_flip(h_work, max_passes=8, deadline=deadline - 8)
        if val_work < best_val:
            best_val = val_work
            best_x = h_work.copy()
        
        # Block flip local search with adaptive sizes
        if time.time() < deadline - 5:
            h_work, val_work = _local_search_block(h_work, deadline=deadline - 2,
                                                    block_sizes=[2, 3, 5, 7, 11])
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # 6. NEW: Simulated Annealing for global exploration
    if time.time() < deadline - 30:
        # Run SA from the current best
        h_sa, val_sa = _simulated_annealing(best_x, deadline - 10)
        if val_sa < best_val:
            best_val = val_sa
            best_x = h_sa.copy()
            
        # Run SA from a few random starts
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            x_rand = rng.choice([1.0, -1.0], N)
            h_sa, val_sa = _simulated_annealing(x_rand, deadline - 5)
            if val_sa < best_val:
                best_val = val_sa
                best_x = h_sa.copy()
                
    # 7. Final high-precision polish: multiple passes of single-bit flips
    if time.time() < deadline - 3:
        h_work = np.sign(best_x)
        h_work[h_work == 0] = 1
        # Run several short passes to squeeze out last improvements
        for polish_pass in range(3):
            if time.time() > deadline - 1:
                break
            h_work, val_work = _local_search_flip(h_work, max_passes=2, deadline=deadline - 1)
            if val_work < best_val:
                best_val = val_work
                best_x = h_work.copy()
    
    # Ensure we return a valid array
    return best_x

if __name__ == "__main__":
    x = construct()
    val = compute_raw_value(x)
    print(f"Best raw value: {val}")
    print(f"Sum: {np.sum(x)}")
